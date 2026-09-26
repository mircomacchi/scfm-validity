#!/bin/bash -l
#SBATCH --job-name=scfm-geneformer
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# Geneformer embeddings on one V100 of the ULHPC Iris cluster.
# Usage (from the project root in scratch): sbatch hpc/geneformer_iris.sbatch
set -euo pipefail
S=/scratch/users/$USER
# Home quota is small: keep every cache in scratch.
export XDG_CACHE_HOME=$S/tools/cache HF_HOME=$S/tools/cache/huggingface
export UV_CACHE_DIR=$S/tools/uv-cache MPLCONFIGDIR=$S/tools/cache/mpl
export CACHE_DIR_HELICAL_PREFIX=$S/tools  # model weights -> $S/tools/.cache/helical/models
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}  # ULHPC docs: match allocated cores

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv/bin/python -c "import torch; print('cuda available:', torch.cuda.is_available())"

.venv/bin/python - <<'PY'
import logging, anndata as ad
from scfm_validity import embed
logging.basicConfig(level=logging.INFO)
path = "data/muscle_prepared.h5ad"
adata = ad.read_h5ad(path)
embed.embed_geneformer(adata, device="cuda", batch_size=32)
adata.write_h5ad(path, compression="gzip")
print("saved", adata.obsm["X_geneformer"].shape)
PY
