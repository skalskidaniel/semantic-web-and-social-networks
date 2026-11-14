"""LLM client that shells out to the Claude Code CLI in headless mode (``claude -p``).

Useful where Claude Code is authenticated but no provider API key is configured. Tools are
disabled and the CLI runs in an empty temporary directory, so it behaves as a plain
completion endpoint. Temperature is not exposed by the CLI and is therefore ignored.
"""

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence

from ontoagents.llm.client import LLMError, LLMResponse, Message


def _flatten(messages: Sequence[Message]) -> tuple[str, str]:
    system_parts = [m.content for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role != "system"]
    if len(turns) == 1:
        prompt = turns[0].content
    else:
        prompt = "\n\n".join(f"<{m.role}>\n{m.content}\n</{m.role}>" for m in turns)
    return "\n\n".join(system_parts), prompt


class ClaudeCodeClient:
    """Completion via ``claude -p --output-format json``."""

    def __init__(
        self,
        model: str,
        *,
        executable: str = "claude",
        timeout_s: float = 600.0,
        retries: int = 3,
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise LLMError(f"{executable!r} not found on PATH")
        self._executable = resolved
        self._model = model
        self._timeout = timeout_s
        self._retries = retries
        self._workdir = tempfile.mkdtemp(prefix="ontoagents-claude-")

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one headless completion, retrying transient failures with backoff."""
        system_prompt, prompt = _flatten(messages)
        cmd = [
            self._executable,
            "-p",
            "--model",
            self._model,
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
            "--system-prompt",
            system_prompt or "You are a helpful assistant.",
        ]
        last_error = ""
        for attempt in range(self._retries):
            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=self._workdir,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_error = f"timeout after {self._timeout}s"
            else:
                try:
                    payload = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    last_error = (
                        f"exit {proc.returncode}: {proc.stderr[-500:] or proc.stdout[-500:]}"
                    )
                else:
                    if not payload.get("is_error") and isinstance(payload.get("result"), str):
                        usage = payload.get("usage") or {}
                        input_tokens = (
                            int(usage.get("input_tokens", 0))
                            + int(usage.get("cache_creation_input_tokens", 0))
                            + int(usage.get("cache_read_input_tokens", 0))
                        )
                        return LLMResponse(
                            text=payload["result"],
                            model=self._model,
                            input_tokens=input_tokens,
                            output_tokens=int(usage.get("output_tokens", 0)),
                        )
                    last_error = str(payload.get("result") or payload)[:500]
            time.sleep(2 ** (attempt + 1))
        raise LLMError(f"claude CLI failed after {self._retries} attempts: {last_error}")
