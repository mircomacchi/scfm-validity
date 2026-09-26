# scfm-validity

**Do single-cell foundation model embeddings capture biology, or artefacts?**

[![CI](https://github.com/mircomacchi/scfm-validity/actions/workflows/ci.yml/badge.svg)](https://github.com/mircomacchi/scfm-validity/actions/workflows/ci.yml)

Doublets and ambient RNA are the two artefacts that most often become a "new cell type" in
scRNA-seq. This package injects both, with known labels, into a real dataset, embeds the cells
three ways and measures whether each embedding keeps the artefacts apart from real biology or
turns them into a convincing population.

| Embedding | What it is |
|---|---|
| PCA | log-normalised counts, 2,000 highly variable genes, 50 components (baseline) |
| scVI | variational autoencoder trained here on raw counts, batch-conditioned, 30 latent dimensions |
| Geneformer | `gf-12L-38M-i4096`, zero-shot through [helical](https://github.com/helicalAI/helical), 512 dimensions |

## Result on human skeletal muscle

Data: the CZ CELLxGENE skeletal muscle atlas (Eraslan et al., *Cell* 2023,
doi:10.1016/j.cell.2023.11.026). 20,184 dissociated cells from three studies (droplet assays
only), subsampled with every cell type kept, plus 1,009 simulated heterotypic doublets and 1,009
cells with 30% ambient contamination.

![Artefact metrics](results/artefact_metrics.png)

| Embedding | Artefact | Detectability (AUROC) | Artefacts in phantom clusters | Doublets near a parent type |
|---|---|---|---|---|
| PCA | doublet | 0.910 | **0.0%** | 95.5% |
| PCA | ambient | 0.982 | 28.8% | |
| scVI | doublet | 0.955 | 15.0% | 93.9% |
| scVI | ambient | 0.994 | 48.5% | |
| Geneformer | doublet | 0.979 | 29.9% | 91.5% |
| Geneformer | ambient | 0.995 | **0.0%** | |

**What it shows.**

1. **Every embedding makes artefacts detectable** (AUROC 0.91 to 0.995). A linear classifier can
   find them, so the information is there.
2. **Whether they become a fake population depends on the model and the artefact.** In the
   Geneformer embedding, Leiden clustering produces one cluster of 434 cells, 70% of them
   simulated doublets, mostly **fibroblast + satellite cell** pairs. An analyst without the labels
   would read it as a hybrid fibro-myogenic population. PCA puts no doublets in an
   artefact-dominated cluster. scVI and PCA both build an ambient-RNA cluster (92% artefact) out of
   mesenchymal stem cell and satellite cell profiles; Geneformer does not.
3. **Zero-shot Geneformer does not beat PCA on standard integration metrics** (scib total 0.524 vs
   0.538), while scVI, which models batch explicitly, scores highest (0.598, from better batch
   correction).

| Embedding | Bio conservation | Batch correction | scib total |
|---|---|---|---|
| PCA | 0.670 | 0.339 | 0.538 |
| scVI | 0.659 | 0.507 | 0.598 |
| Geneformer | 0.656 | 0.326 | 0.524 |

![UMAPs](results/umap_grid.png)

**Limits.** One dataset, one seed, one clustering resolution (Leiden 1.0, "phantom" = at least 50%
artefact). The artefacts are simulated: doublets are summed counts thinned to 1.6x singlet depth;
ambient contamination uses the pooled profile of each batch, since empty droplets are not
available. Geneformer is used zero-shot; fine-tuning could change the picture. Treat the numbers
as a case study, not a benchmark.

## Metrics

| Metric | Question | Better |
|---|---|---|
| `detectability_auroc` | Can a 5-fold cross-validated logistic regression tell artefacts from real cells in the embedding? | higher |
| `knn_enrichment` | How much more often do artefacts neighbour other artefacts than their prevalence predicts? | context |
| `phantom_fraction` | Share of artefact cells in Leiden clusters that are at least 50% artefact | lower |
| `parent_consistency` | Share of doublets whose majority real-cell neighbours are one of their two parent types | higher |

Plus the scib-metrics benchmark (bio conservation and batch correction) on the real cells.

## Use

```bash
pip install -e ".[scvi,fm,dev]"     # Python 3.11 or 3.12

scfm-validity prepare atlas.h5ad data/prepared.h5ad --n-cells 20000 --batch-key Dataset \
  --obs-filter '{"suspension_type": ["cell"]}'
scfm-validity embed data/prepared.h5ad --methods pca,scvi
bash hpc/geneformer_local.sh data/prepared.h5ad          # or hpc/geneformer_iris.sbatch on a GPU cluster
scfm-validity evaluate data/prepared.h5ad results/
```

Outputs: `results/artefact_metrics.parquet`, `results/scib_metrics.parquet` and the two figures.

## Engineering notes

- **Tests.** `pytest` runs on a synthetic 600-cell dataset, including an end-to-end CLI run, so CI
  needs no download. GitHub Actions runs ruff and pytest on Python 3.11 and 3.12, then builds the
  Docker image.
- **Memory on a 16 GB laptop.** Geneformer on Apple MPS thrashed memory: 200 cells took 1,325 s with
  an 18.5 GB footprint, against 235 s and 6.0 GB on CPU. Memory also grew across chunks within one
  process (40 GB after 11 chunks), so `hpc/geneformer_local.sh` embeds one 1,000-cell chunk per
  process, checkpoints each chunk and resumes where it stopped. Geneformer reads only the counts
  and writes its embedding straight into the h5ad. Chunks 1 to 11 of this run were computed on MPS
  and 12 to 23 on CPU with the same weights.
- **helical 3.1.3 workaround.** Its gene-collapsing step drops `var["ensembl_id"]` and then fails;
  `embed.py` deduplicates in-vocabulary genes and turns collapsing off.

## Author

Mirco Macchi, computational biologist (PhD, LCSB, University of Luxembourg).
Built with AI pair programming (Claude Code).
MIT licence.
