"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        config: Any | None = None,
        fallback_models: list | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # Model fallback state
        self._config = config                           # kept for runtime provider factory
        self._fallback_models = list(fallback_models or [])
        self._fallback_index = 0                        # next fallback to try
        self._primary_model = self.model                # remember original for /model reset
        self._primary_provider = provider

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None,
                          metadata: dict | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "message":
                        tool.set_context(channel, chat_id, message_id, metadata=metadata)
                    else:
                        tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            # Local models (ollama CPU): no tools, timeout, minimal context already set
            is_local = self.context.local_mode
            tools_for_call = None if is_local else self.tools.get_definitions()
            local_timeout = 80.0  # seconds before we give up on a slow local model

            try:
                chat_coro = self.provider.chat(
                    messages=messages,
                    tools=tools_for_call,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 512) if is_local else self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                if is_local:
                    response = await asyncio.wait_for(chat_coro, timeout=local_timeout)
                else:
                    response = await chat_coro
            except asyncio.TimeoutError:
                logger.warning("Local model '{}' timed out after {}s, trying fallback...", self.model, local_timeout)
                switched = await self._try_switch_to_fallback()
                if switched:
                    logger.info("Switched to fallback model '{}' after timeout", self.model)
                    # Rebuild messages with new (non-local) context if we left local mode
                    continue
                final_content = (
                    f"The local model timed out and no further fallbacks are available.\n"
                    f"Try `/model reset` to go back to the primary model."
                )
                break

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Rate-limit: try automatic fallback to next model in list.
                if response.finish_reason == "rate_limit":
                    logger.warning("Rate limited on model '{}', trying fallback...", self.model)
                    switched = await self._try_switch_to_fallback()
                    if switched:
                        logger.info("Switched to fallback model '{}', retrying...", self.model)
                        continue  # retry with new model
                    # No more fallbacks available
                    logger.error("All fallback models exhausted. Last error: {}", (clean or "")[:200])
                    final_content = (
                        f"All models are currently rate limited or unavailable.\n"
                        f"Last error: {clean or 'unknown error'}\n"
                        f"Try /model <name> to switch manually, or try again later."
                    )
                    break
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                # Reset fallback index on successful response — primary model may be available again
                self._fallback_index = 0
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/voice — Reply with voice message\n/model [name|save|reset] — Switch or inspect active model\n/help — Show available commands")
        if cmd.startswith("/model"):
            return await self._handle_model_command(msg)

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"),
                               metadata=msg.metadata)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {} (voice_requested={})", msg.channel, msg.sender_id, preview, msg.metadata.get("voice_requested"))
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _try_switch_to_fallback(self) -> bool:
        """Try to switch to the next fallback model. Returns True if switched, False if exhausted."""
        if not self._fallback_models or not self._config:
            return False

        # Cycle through all fallbacks; wrap around once to retry from start next time
        attempts = 0
        while attempts < len(self._fallback_models):
            fb = self._fallback_models[self._fallback_index % len(self._fallback_models)]
            self._fallback_index += 1
            attempts += 1
            try:
                from nanobot.providers.factory import make_provider
                new_provider = make_provider(self._config, model=fb.model, provider_name=fb.provider)
                self._switch_model(fb.model, new_provider)
                logger.info("Auto-fallback: switched to '{}' (provider: {})", fb.model, fb.provider)
                return True
            except Exception as e:
                logger.warning("Fallback model '{}' unavailable: {}", fb.model, e)
                continue

        # All fallbacks failed — reset index so next rate-limit starts fresh
        self._fallback_index = 0
        return False

    def _is_local_provider(self) -> bool:
        """Return True if the current provider is a slow local model (ollama CPU etc.)."""
        if not self._config:
            return False
        from nanobot.providers.registry import find_by_name
        provider_name = self._config.get_provider_name(self.model)
        spec = find_by_name(provider_name) if provider_name else None
        return bool(spec and spec.is_local)

    def _switch_model(self, model: str, provider: LLMProvider | None = None) -> None:
        """Switch active model (and optionally provider) at runtime.

        Also toggles context.local_mode based on whether the new provider is local.
        """
        self.model = model
        if provider is not None:
            self.provider = provider
        # Keep subagents in sync
        self.subagents.model = model
        if provider is not None:
            self.subagents.provider = provider
        # Switch context to minimal mode for slow local models
        self.context.local_mode = self._is_local_provider()

    async def _handle_model_command(self, msg: "InboundMessage") -> "OutboundMessage":
        """Handle /model [name|save|save <name>] command."""
        # Parse: preserve original case for model names, but command keyword is lowercased
        raw = msg.content.strip()
        parts = raw.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if len(parts) == 1:
            # /model — show current + fallback list
            lines = [f"**Active model:** `{self.model}`"]
            if self._fallback_models:
                lines.append("\n**Fallback list:**")
                for i, fb in enumerate(self._fallback_models, 1):
                    marker = " ← next" if (i - 1) == (self._fallback_index % len(self._fallback_models)) else ""
                    lines.append(f"  {i}. `{fb.model}` ({fb.provider}){marker}")
                lines.append("\nUse `/model <number>` or `/model <name>` to switch.")
            else:
                lines.append("_No fallback models configured._")
            content = "\n".join(lines)

        elif sub.isdigit():
            # /model <number> — pick from fallback list by index
            idx = int(sub) - 1
            if 0 <= idx < len(self._fallback_models):
                fb = self._fallback_models[idx]
                # Pass provider_name explicitly so auto-detection doesn't pick the wrong one
                await self._switch_model_by_name(fb.model, provider_name=fb.provider)
                content = f"Switched to `{self.model}` ({fb.provider}) for this session. Use `/model save` to persist."
            else:
                content = f"Invalid number. Choose 1–{len(self._fallback_models)}. Use `/model` to see the list."

        elif sub == "save" and len(parts) == 2:
            # /model save — persist current active model to config.json
            self._save_model_to_config(self.model)
            content = f"Saved `{self.model}` as default model in config.json."

        elif sub == "save" and len(parts) == 3:
            # /model save <name> — switch + persist
            new_model = parts[2]
            await self._switch_model_by_name(new_model)
            self._save_model_to_config(self.model)
            content = f"Switched to `{self.model}` and saved as default in config.json."

        elif sub == "reset":
            # /model reset — back to primary (startup) model
            if self._config:
                try:
                    from nanobot.providers.factory import make_provider
                    new_provider = make_provider(self._config, model=self._primary_model)
                    self._switch_model(self._primary_model, new_provider)
                    self._fallback_index = 0
                    content = f"Reset to primary model: `{self.model}`"
                except Exception as e:
                    content = f"Failed to reset to primary model: {e}"
            else:
                self._switch_model(self._primary_model)
                self._fallback_index = 0
                content = f"Reset to primary model: `{self.model}`"

        else:
            # /model <name> — session switch (not persisted)
            new_model = parts[1]  # use original case
            await self._switch_model_by_name(new_model)
            content = f"Switched to `{self.model}` for this session. Use `/model save` to persist."

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    async def _switch_model_by_name(self, model: str, provider_name: str | None = None) -> None:
        """Switch to a named model, instantiating the right provider from config.

        Args:
            model: Model name to switch to.
            provider_name: Force a specific provider (e.g. "ollama", "lmstudio").
                           If None, auto-detects from model name — which can be wrong
                           for locally-named models like "qwen2.5:3b".
        """
        if self._config:
            try:
                from nanobot.providers.factory import make_provider
                new_provider = make_provider(self._config, model=model, provider_name=provider_name)
                self._switch_model(model, new_provider)
                return
            except Exception as e:
                logger.warning("Could not instantiate provider '{}' for '{}': {}. Reusing current provider.", provider_name, model, e)
        # Fallback: keep current provider, just swap model name
        self._switch_model(model)

    def _save_model_to_config(self, model: str) -> None:
        """Persist model name to ~/.nanobot/config.json."""
        try:
            from nanobot.config.loader import load_config, save_config
            cfg = load_config()
            cfg.agents.defaults.model = model
            save_config(cfg)
            # Keep our local _config in sync too
            if self._config:
                self._config.agents.defaults.model = model
        except Exception as e:
            logger.error("Failed to save model to config: {}", e)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
