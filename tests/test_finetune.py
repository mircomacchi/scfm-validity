import numpy as np
import pandas as pd

from scfm_validity import finetune


def _obs():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cell_type": rng.choice(["a", "b", "rare"], 1000, p=[0.5, 0.47, 0.03]),
            "Dataset": np.repeat(["s1", "s2"], 500),
            "artefact": ["none"] * 980 + ["doublet"] * 20,
        }
    )


def test_split_holds_out_a_whole_batch():
    obs = _obs()
    train, holdout, classes = finetune.split_by_batch(obs, "cell_type", "Dataset", "s2", 50, 200)
    assert set(obs["Dataset"].iloc[train]) == {"s1"}
    assert set(obs["Dataset"].iloc[holdout]) == {"s2"}
    assert classes == ["a", "b"]  # "rare" has < 50 training cells
    assert set(obs["cell_type"].iloc[holdout]) <= set(classes)
    assert (obs["artefact"].iloc[holdout] == "none").all()


def test_linear_probe_separable():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 1, (100, 4)), rng.normal(5, 1, (100, 4))])
    y = np.repeat(["a", "b"], 100)
    idx = rng.permutation(200)
    scores = finetune.linear_probe(Z, y, idx[:150], idx[150:])
    assert scores["accuracy"] > 0.95 and scores["macro_f1"] > 0.95
