#!/usr/bin/env bash
# Geneformer on a 16 GB Mac: one 1,000-cell chunk per process, so the OS reclaims
# all memory between chunks. CPU, not MPS: on 200 cells MPS took 1,325 s with an
# 18.5 GB footprint, CPU 235 s with 6.0 GB.
# Chunks already in data/geneformer_chunks are skipped.
# Usage: bash hpc/geneformer_local.sh data/muscle_prepared.h5ad [extra embed options]
# e.g.   ... --geneformer-model gf-6L-10M-i2048 --geneformer-key X_gf6l_zeroshot
set -uo pipefail
h5ad=${1:?path to prepared h5ad}
shift
while true; do
  caffeinate -i .venv/bin/scfm-validity embed "$h5ad" --methods geneformer --max-chunks 1 \
    --geneformer-device cpu "$@"
  status=$?
  [ $status -eq 0 ] && { echo "Geneformer complete"; exit 0; }
  [ $status -ne 3 ] && { echo "Failed with exit code $status"; exit $status; }
done
