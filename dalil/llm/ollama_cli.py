"""Talk to a cloud model through the local `ollama` CLI instead of the HTTP API.

`ollama_cloud.py` posts to https://ollama.com/api/chat directly; a 2,500-word
chunk measured anywhere from 40 s to 500 s there. `ollama run` signs the request
with the local ollama credentials and returned promptly for the same prompt, so
the CLI is a second transport with the same `ask` / `ask_json` interface.

Three things this is careful about:

`--nowordwrap` is mandatory, not cosmetic. Without it `ollama run` word-wraps its
output using terminal cursor movements even when stdout is a pipe, so the raw bytes
contain sequences like `الفريض\x1b[6D\x1b[K الفريضة والنافلة` — cursor-back-6 then
erase-line. A terminal renders that correctly; anything reading the bytes sees a
duplicated fragment and reconstructs a word that is not in the text, which then
fails anchor verification for a reason that looks like a model error and is not.
`--hidethinking` drops the reasoning block, which comes back in Chinese and is of
no use to us.

`--format json` is deliberately NOT used. It constrains the model to emit JSON
immediately, which stops it reasoning before it answers: on the same chunk it
returned 2 boundaries with the flag and 7 without it, for the same wall time. A
correctly-shaped answer that misses most of the boundaries is worse than one that
needs a regex to find the JSON in it.

The prompt goes in on **stdin**, never as an argv element. A boundary prompt is
~14 KB of Arabic; passing that as an argument risks ARG_MAX and mangles quoting.

The child is started in its **own process group** and, on timeout, the whole group
is killed rather than just the `ollama` process. `ollama run` can leave a child
holding the connection, and killing only the parent leaves it running — this
project has already been bitten once by orphaned model processes that survived for
two hours after their launcher died.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time

from . import LLMError, parse_json_reply

DEFAULT_MODEL = "deepseek-v4-flash:cloud"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class OllamaCLIError(LLMError):
    pass


def ask(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = 300,
        retries: int = 2) -> str:
    """Send one prompt through `ollama run`, return the reply text.

    Each attempt is a fresh process: request, reply, exit. Nothing is kept warm
    between calls, so there is no session state to accumulate or clean up.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        proc = subprocess.Popen(
            ["ollama", "run", "--nowordwrap", "--hidethinking", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            out, err = proc.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            last = TimeoutError(f"no reply within {timeout}s")
        else:
            if proc.returncode == 0 and out.strip():
                # The progress spinner goes to stderr, but strip escapes from
                # stdout too rather than trusting that to stay true.
                return ANSI_RE.sub("", out).strip()
            # stderr carries the spinner's frames before the error itself.
            message = ANSI_RE.sub("", err or out)
            if "Error:" in message:
                message = message[message.rindex("Error:"):]
            last = OllamaCLIError(f"rc={proc.returncode}: {message.strip()[:200]}")
        if attempt < retries:
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    raise OllamaCLIError(str(last))


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group, then make sure it is gone.

    SIGTERM first so ollama can close its connection cleanly; SIGKILL after a
    grace period, because this project has seen these ignore SIGTERM entirely.
    """
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
    return parse_json_reply(ask(prompt, **kw), OllamaCLIError)

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Send stdin (or --prompt) to a cloud model.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--prompt", help="prompt text; omit to read stdin")
    args = ap.parse_args()

    text = args.prompt if args.prompt is not None else sys.stdin.read()
    started = time.time()
    reply = ask(text, model=args.model, timeout=args.timeout, retries=args.retries)
    print(reply)
    print(f"\n--- {time.time() - started:.1f}s ---", file=sys.stderr)
