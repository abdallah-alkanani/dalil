#!/bin/sh
# Copy the dalil package (code only) next to the API so the Docker build context
# is self-contained. Run before `railway up`.
set -eu
# The engine is this repository, three levels up (web/backend/scripts).
SRC="${DALIL_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor"
rm -rf "$DEST" && mkdir -p "$DEST"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$SRC/dalil" "$DEST/"
cp "$SRC/pyproject.toml" "$SRC/README.md" "$SRC/LICENSE" "$DEST/"
# The answering prompt, read by dalil_api.answer.
mkdir -p "$DEST/prompts" && cp "$SRC/prompts/answering_system_prompt.md" "$DEST/prompts/"
echo "vendored $(find "$DEST/dalil" -name '*.py' | wc -l | tr -d ' ') files from $SRC"
