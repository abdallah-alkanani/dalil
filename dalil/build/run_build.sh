#!/bin/bash
# Build the search database from the source texts and recipe/, stage by stage, on a GPU machine.
#
#   P=<project dir> bash dalil/build/run_build.sh prepare|extract|select|plan|embed|assemble|index
#
# $P holds .venv/ and data/ (the books in data/books/original/<author>/<book>/, at the paths
# the manifests in recipe/books/ give). Everything the build writes, logs and temp
# files included, goes to $BUILD (default $P/data/build); the finished stores to $BUILD/db.
#
# prepare   the repaired or derived copies in $BUILD/books (92317, 1423, 6006, the manzuma's
#           paired verses) and the Ketabonline elements of every package
# extract   the core corpus and the collection -> $BUILD/structured/{core,collection}, validated
# select    drop books, garbage views, author ids -> $BUILD/corpora; token check (embed_views
#           silently truncates past 8,192 tokens)
# plan      per corpus: the views whose text has no vector yet; one part per GPU
# embed     one embed_views process per GPU
# assemble  per corpus: a shard for every view, reused or new
# index     empty stores, every corpus appended, then author ids, reading order, BM25, checks
#
# GPUS (default "0 1 2 3") lists the GPUs; plan, embed and assemble must see the same list.
# CACHE (default $P/data/cache) holds the Hugging Face models (Qwen3-Embedding-8B, its tokenizer).
# ENGINE=vllm with EMBED_PY=<a venv with vllm and dalil> embeds 1.4x faster.
# WORKERS (default: every core, nproc) is the process count of each parallel step.
# REUSE_DB (optional) is a set of stores whose vectors are reused for identical text; without
# it every view is embedded. The manzuma's verses are read from its .doc with LibreOffice (soffice).
set -euo pipefail
SELF=$(realpath "$0")
case "${1:-}" in prepare|extract|select|plan|embed|assemble|index) ;; *) sed -n 2,23p "$SELF"; exit 1 ;; esac
CODE=$(dirname "$(dirname "$(dirname "$SELF")")")
P=${P:?set P to the project dir holding .venv/ and data/}
D=$P/data; B=${BUILD:-$D/build}; CACHE=${CACHE:-$D/cache}
export DALIL_DATA_DIR=$D HF_HOME=$CACHE/huggingface TORCH_HOME=$CACHE/torch XDG_CACHE_HOME=$CACHE \
       TMPDIR=$B/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$P/.venv/bin/python
# The embed stage's interpreter and engine: vLLM is faster, and lives in its own venv.
EMBED_PY=${EMBED_PY:-$PY}; ENGINE=${ENGINE:-sentence-transformers}
O=$D/books/original; K=$B/books; CUTS=$CODE/recipe/decisions/semantic_boundaries.jsonl
CORPORA="core collection"
read -ra GPU <<< "${GPUS:-0 1 2 3}"
WORKERS=${WORKERS:-$(nproc)}
L=$B/logs
mkdir -p $L $B/tmp
cd $B

case "${1:-}" in
prepare)
  rm -rf prepared && mkdir prepared
  # Three independent chains side by side; `wait PID` fails the stage if one failed.
  $PY -m dalil.s2_prepare.ketabonline_handoff prepared/ketabonline_handoff --workers $WORKERS & handoff=$!
  ( $PY -m dalil.s2_prepare.question_marks &&
    $PY -m dalil.s2_prepare.ketabonline --raw $K --only 92317 --author walid_alsaeedan=walid_alsaeedan \
        --no-db-overlap --output $B/prepared/ketabonline_92317 ) > $L/prepare_92317.log 2>&1 & q92317=$!
  ( $PY -m dalil.s2_prepare.albani_restore 1423 6006 &&
    $PY -m dalil.s2_prepare.ketabonline --raw $K --only 1423 --only 6006 \
        --no-db-overlap --no-collection-overlap --output $B/prepared/ketabonline_1423_6006 ) > $L/prepare_1423_6006.log 2>&1 & albani=$!
  doc=$(find $O -name 054.doc); txt=$K${doc#$O}; txt=${txt%.doc}.txt; mkdir -p "$(dirname "$txt")"
  if command -v soffice > /dev/null; then
    soffice --headless --convert-to html --outdir prepared/manzuma "$doc" > /dev/null
    $PY -m dalil.s2_prepare.manzuma prepared/manzuma/054.html > "$txt"
  else
    echo "soffice (LibreOffice) not installed: keeping the existing $txt"; test -s "$txt"
  fi
  wait $handoff; wait $q92317; wait $albani
  cat prepared/{ketabonline_handoff,ketabonline_92317,ketabonline_1423_6006}/elements.jsonl > prepared/ketabonline_elements.jsonl ;;
extract)
  rm -rf structured && mkdir structured
  # Side by side: the core corpus's longest books overlap the collection's.
  $PY -m dalil.s4_extract.core_corpus --output-root structured/core --workers $WORKERS > $L/extract_core.log 2>&1 & core=$!
  $PY -m dalil.s4_extract.collection --ketab-elements prepared/ketabonline_elements.jsonl \
      --boundaries $CUTS --output-root structured/collection --workers $WORKERS > $L/extract_collection.log 2>&1 & coll=$!
  wait $core; wait $coll
  tail -n 8 $L/extract_core.log $L/extract_collection.log
  for c in $CORPORA; do
    $PY -m dalil.s4_extract.validate --root structured/$c --sample 0 > $L/validate_$c.log 2>&1 &
  done
  wait
  tail -n 2 $L/validate_*.log ;;
select)
  rm -rf corpora
  $PY -m dalil.build.select --work $B $(for c in $CORPORA; do echo --corpus $c=structured/$c; done)
  for c in $CORPORA; do
    $PY -m dalil.s5_audit.token_stats corpora/$c/retrieval_views.jsonl --workers $((WORKERS / 2)) > $L/tokens_$c.log 2>&1 &
  done
  wait
  grep -H -E "max |> 4096|> 8192" $L/tokens_*.log ;;
plan)
  rm -rf to_embed parts empty_db; mkdir -p to_embed
  vectors=${REUSE_DB:-}
  if [ -z "$vectors" ]; then vectors=empty_db; $PY -m dalil.s7_index.schema --db empty_db; fi
  for c in $CORPORA; do
    $PY -m dalil.s6_embed.reuse plan --corpus corpora/$c --db $vectors --out to_embed/$c.jsonl > $L/plan_$c.log 2>&1 &
  done
  wait
  grep -h -A4 '"views"' $L/plan_*.log || true
  $PY -m dalil.s6_embed.split_parts to_embed parts ${#GPU[@]} ;;
embed)
  rm -rf new
  for i in "${!GPU[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU[$i]} timeout 10h $EMBED_PY -m dalil.s6_embed.embed_views --engine $ENGINE \
        --views parts/part_$i.jsonl --out new/part$i \
        > $L/embed_part$i.log 2>&1 &
  done
  wait ;;
assemble)
  rm -rf embeddings; mkdir -p embeddings
  vectors=${REUSE_DB:-empty_db}
  new=""; texts=""
  for i in "${!GPU[@]}"; do new="$new --new new/part$i"; texts="$texts --texts parts/part_$i.jsonl"; done
  for c in $CORPORA; do
    $PY -m dalil.s6_embed.reuse assemble --corpus corpora/$c --db $vectors $new $texts \
        --out embeddings/$c > $L/assemble_$c.log 2>&1 &
  done
  wait
  tail -n 5 $L/assemble_*.log ;;
index)
  rm -rf db
  $PY -m dalil.s7_index.schema --db db
  for c in $CORPORA; do
    $PY -m dalil.s7_index.append --corpus corpora/$c --embeddings embeddings/$c --db db --no-backup \
        2>&1 | tee $L/append_$c.log | tail -6
  done
  $PY -m dalil.build.finalize --db db --corpora corpora --authors author_ids.json ;;
esac
