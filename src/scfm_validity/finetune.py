"""Fine-tune Geneformer for cell-type classification and evaluate it honestly.

The split is by study: the model trains on some batches and is scored on a
held-out batch it never saw, so the score measures transfer across labs and
protocols rather than recall of the training set. A logistic regression on the
zero-shot embedding of the same model is the baseline the fine-tuning must beat.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scfm_validity.data import stratified_indices
from scfm_validity.embed import vocab_subset

log = logging.getLogger(__name__)
META = "scfm_meta.json"


def split_by_batch(
    obs: pd.DataFrame,
    label_key: str,
    batch_key: str,
    holdout_batch: str,
    min_cells: int = 50,
    n_train: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Positions of training and held-out real cells, and the label classes.

    Classes are the labels with at least `min_cells` training cells; held-out
    cells with any other label are dropped, since no model could predict them.
    """
    real = obs["artefact"].astype(str).eq("none").to_numpy() if "artefact" in obs else True
    labels = obs[label_key].astype(str)
    batches = obs[batch_key].astype(str)
    train_pool = np.flatnonzero(real & batches.ne(holdout_batch).to_numpy())
    counts = labels.iloc[train_pool].value_counts()
    classes = sorted(counts.index[counts >= min_cells])
    train_pool = train_pool[labels.iloc[train_pool].isin(classes).to_numpy()]
    train = train_pool[stratified_indices(labels.iloc[train_pool], n_train, min_cells, seed)]
    holdout = np.flatnonzero(
        real & batches.eq(holdout_batch).to_numpy() & labels.isin(classes).to_numpy()
    )
    return train, holdout, classes


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 over the classes present in the held-out batch.

    A held-out study rarely contains every training class; averaging over absent
    classes would count each stray prediction as an F1 of 0 for a class that has no
    true cells, so the average is restricted to the classes that are there.
    """
    present = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro")),
        "n_classes_scored": int(present.size),
    }


def linear_probe(
    Z: np.ndarray, y: np.ndarray, train: np.ndarray, holdout: np.ndarray
) -> dict[str, float]:
    """Logistic regression on a frozen embedding: the baseline for fine-tuning."""
    scaler = StandardScaler().fit(Z[train])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(Z[train]), y[train])
    pred = clf.predict(scaler.transform(Z[holdout]))
    return {**_scores(y[holdout], pred), "predictions": pred.tolist()}


def finetune_geneformer(
    adata: ad.AnnData,
    out_dir: str | Path,
    label_key: str = "cell_type",
    batch_key: str = "Dataset",
    holdout_batch: str = "He et al. 2020",
    model_name: str = "gf-6L-10M-i2048",
    epochs: int = 3,
    freeze_layers: int = 4,
    lr: float = 1e-4,
    batch_size: int = 4,
    n_train: int = 5000,
    min_cells: int = 50,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Fine-tune a classification head plus the unfrozen layers; save and score it."""
    import torch
    from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

    torch.manual_seed(seed)
    train, holdout, classes = split_by_batch(
        adata.obs, label_key, batch_key, holdout_batch, min_cells, n_train, seed
    )
    code = {c: i for i, c in enumerate(classes)}
    log.info(
        "Fine-tuning on %d cells, scoring on %d held-out cells (%s), %d classes",
        len(train),
        len(holdout),
        holdout_batch,
        len(classes),
    )

    config = GeneformerConfig(model_name=model_name, batch_size=batch_size, device=device)
    model = GeneformerFineTuningModel(config, "classification", output_size=len(classes))
    model.tk.collapse_gene_ids = False
    X, var = vocab_subset(adata, model.tk.gene_token_dict)

    def dataset(pos: np.ndarray):
        sub = ad.AnnData(X=X[pos], obs=pd.DataFrame(index=adata.obs_names[pos]), var=var)
        y = [code[c] for c in adata.obs[label_key].astype(str).to_numpy()[pos]]
        return model.process_data(sub, gene_names="ensembl_id").add_column("label", y), np.array(y)

    train_ds, _ = dataset(train)
    holdout_ds, y_holdout = dataset(holdout)
    # helical steps the scheduler once per epoch, not per batch: schedule in epochs.
    # (A warm-up counted in batches leaves the learning rate at 0 for a whole epoch.)
    model.train(
        train_dataset=train_ds,
        label="label",
        epochs=epochs,
        freeze_layers=freeze_layers,
        optimizer_params={"lr": lr},
        lr_scheduler_params={"name": "linear", "num_warmup_steps": 0, "num_training_steps": epochs},
    )
    logits = model.get_outputs(holdout_ds.remove_columns("label"))
    report = {
        "model_name": model_name,
        "holdout_batch": holdout_batch,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "classes": classes,
        "epochs": epochs,
        "freeze_layers": freeze_layers,
        "lr": lr,
        "finetuned": _scores(y_holdout, logits.argmax(axis=1)),
    }
    np.save(out_dir_path(out_dir) / "holdout_pred.npy", logits.argmax(axis=1))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.pt"))
    np.save(out / "train_idx.npy", train)
    np.save(out / "holdout_idx.npy", holdout)
    (out / META).write_text(json.dumps(report, indent=2))
    log.info("Fine-tuned held-out scores: %s", report["finetuned"])
    return report


def out_dir_path(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_finetuned(weights: str | Path, device: str = "cpu", batch_size: int = 4):
    """Rebuild a model saved by `finetune_geneformer`."""
    from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

    weights = Path(weights)
    meta = json.loads((weights / META).read_text())
    config = GeneformerConfig(model_name=meta["model_name"], batch_size=batch_size, device=device)
    model = GeneformerFineTuningModel(config, "classification", output_size=len(meta["classes"]))
    import torch

    model.load_model(str(weights / "model.pt"))
    # helical's fine-tuning class overrides Module.train() with its training loop,
    # so model.eval() would start training; switch to inference mode directly.
    torch.nn.Module.train(model, False)
    return model
