#!/usr/bin/env bash
# Run the baseline, the full system and every ablation, then aggregate.
# Usage: experiments/run_all.sh [--max-cqs N] [--workers N] [extra `ontoagents run` args]
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-4}
for cfg in baseline-ontogenia baseline-single-pass agents-full ablation-no-critic ablation-llm-planner \
           ablation-no-regression ablation-no-translator ablation-no-repair; do
  echo "=== $cfg"
  uv run ontoagents run --config "experiments/configs/$cfg.yaml" --workers "$WORKERS" "$@" || true
done
uv run ontoagents report --results-dir experiments/results
