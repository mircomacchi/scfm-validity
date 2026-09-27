#!/usr/bin/env bash
# Zero-shot vs fine-tuned Geneformer (gf-6L-10M-i2048) on a 16 GB Mac, CPU only.
# 1. zero-shot embeddings, 2. fine-tune split by study + linear-probe baseline,
# 3. fine-tuned embeddings, 4. artefact metrics for every embedding.
# Usage: bash hpc/finetune_local.sh data/muscle_prepared.h5ad
set -euo pipefail
h5ad=${1:?path to prepared h5ad}
ft_dir=$(dirname "$h5ad")/gf6l_finetuned
bash hpc/geneformer_local.sh "$h5ad" --geneformer-model gf-6L-10M-i2048 --geneformer-key X_gf6l_zeroshot
[ -f "$ft_dir/model.pt" ] || caffeinate -i .venv/bin/scfm-validity finetune "$h5ad" "$ft_dir"
bash hpc/geneformer_local.sh "$h5ad" --geneformer-weights "$ft_dir" --geneformer-key X_gf6l_finetuned
caffeinate -i .venv/bin/scfm-validity evaluate "$h5ad" results/
echo "Pipeline complete"
