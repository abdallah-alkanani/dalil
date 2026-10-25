"""Talk to a model through the local `opencode` CLI, using its signed-in providers.

`opencode run` reaches the same Ollama Cloud models as `ollama_cli.py`, through
whatever account opencode is logged in to (`opencode auth login`), so it is one
more transport with the same `ask` / `ask_json` interface.

Three things this is careful about:

**No tools.** opencode is a coding agent: its default agent can run shell
commands and edit files, and a user config that allows every permission lets it
do so without asking. A boundary prompt only needs text back, so each call runs
with `XDG_CONFIG_HOME` pointed at a private config whose only rule denies every
tool, in an empty working directory. Measured: asked to `touch` a file, the model
then gets "No tool named ... is currently available" and the file is not created.
The login lives in opencode's data directory, not its config, so it still works.
`--standalone` keeps the call off the background service, which has the user's
config loaded.

**The reply is read from `--format json`.** Plain output carries a banner and
terminal escapes; the JSON event stream has the reply as `text` parts.

The prompt goes in on **stdin**, and the child runs in its **own process group**,
killed whole on timeout, for the same reasons as `ollama_cli.py`.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import LLMError, parse_json_reply

DEFAULT_MODEL = "ollama-cloud/deepseek-v4.1-flash"
NO_TOOLS_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
}
_sandbox: Path | None = None
_sandbox_lock = threading.Lock()


class OpencodeCLIError(LLMError):
    pass


def _sandbox_dir() -> Path:
    """An empty directory holding the no-tools config, made once per process."""
    global _sandbox
    with _sandbox_lock:  # boundary workers call this from several threads at once
        if _sandbox is None:
            path = Path(tempfile.mkdtemp(prefix="dalil-opencode-"))
            atexit.register(shutil.rmtree, path, ignore_errors=True)
            (path / "opencode").mkdir()
            (path / "opencode" / "opencode.json").write_text(json.dumps(NO_TOOLS_CONFIG))
            _sandbox = path
    return _sandbox


def ask(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 300,
        retries: int = 2) -> str:
    """Send one prompt through `opencode run`, return the reply text."""
    sandbox = _sandbox_dir()
    env = {**os.environ, "XDG_CONFIG_HOME": str(sandbox)}
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    last: Exception | None = None
    for attempt in range(retries + 1):
        proc = subprocess.Popen(
            ["opencode", "run", "--standalone", "-m", model, "--format", "json"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=sandbox, env=env, start_new_session=True,
        )
        try:
            out, err = proc.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            last = TimeoutError(f"no reply within {timeout}s")
        else:
            texts, errors = [], []
            for line in out.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                part = event.get("part") or {}
                if event.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])
                elif event.get("type") == "error" or event.get("error"):
                    errors.append(json.dumps(event.get("error") or event)[:200])
            if proc.returncode == 0 and texts:
                return "\n".join(texts).strip()
            message = "; ".join(errors) or (err or out).strip()[-200:]
            last = OpencodeCLIError(f"rc={proc.returncode}: {message}")
        if attempt < retries:
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    raise OpencodeCLIError(str(last))


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group: SIGTERM, then SIGKILL after 5 s."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def ask_json(prompt: str, **kw) -> dict:
    """Send one prompt and parse the JSON object out of the reply."""
    return parse_json_reply(ask(prompt, **kw), OpencodeCLIError)
