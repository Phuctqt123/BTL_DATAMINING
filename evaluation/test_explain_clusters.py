import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from evaluation import explain_clusters as xc


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 200
    labels = pd.Series(np.repeat([0, 1, 2, 3], n // 4), name="kmeans")
    features = pd.DataFrame(rng.normal(size=(n, len(xc.FEATURE_COLUMNS))), columns=xc.FEATURE_COLUMNS)
    # Monetary quyết định hoàn toàn nhãn, các cột khác là nhiễu
    features["Monetary"] = labels * 10 + rng.normal(scale=0.5, size=n)
    return features, labels


def test_surrogate_recovers_clean_rule(toy):
    features, labels = toy
    tree = xc.fit_surrogate(features, labels)
    assert tree.get_depth() <= xc.SURROGATE_DEPTH
    assert tree.score(features, labels) == 1.0
    imp = xc.surrogate_permutation_importance(tree, features, labels)
    assert imp["surrogate_perm_mean"].idxmax() == "Monetary"


def test_decision_path_ends_with_persona(toy):
    features, labels = toy
    tree = xc.fit_surrogate(features, labels)
    steps = xc.decision_path(tree, features.iloc[0])
    assert steps[-1] == "VIP"
    assert all("Monetary" in s for s in steps[:-1])


def test_kmeans_shuffle_importance_is_share(toy):
    features, _ = toy
    km = KMeans(n_clusters=4, n_init=5, random_state=xc.RANDOM_STATE).fit(features)
    imp = xc.kmeans_shuffle_importance(km, features, n_repeats=3)
    assert list(imp.index) == xc.FEATURE_COLUMNS
    assert imp["kmeans_change_mean"].between(0, 1).all()
    assert imp["kmeans_change_mean"].idxmax() == "Monetary"


def test_number_format_is_vietnamese():
    assert xc._fmt(1995.34) == "1.995,34"
    assert xc._fmt(56.5) == "56,5"
    assert xc._fmt(100.0) == "100"
