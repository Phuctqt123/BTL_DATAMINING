import pytest
import numpy as np

from evaluation.personas import assign_personas, get_persona


def test_get_persona_for_each_kmeans_cluster():
    assert get_persona(0)["name"] == "VIP"
    assert get_persona(1)["name"] == "Tiềm năng"
    assert get_persona(2)["name"] == "Cần níu kéo"
    assert get_persona(3)["name"] == "Ngủ đông"


def test_noise_label_has_separate_persona():
    assert get_persona(-1)["name"] == "Ngoại biên DBSCAN"


def test_numpy_integer_label_is_supported():
    assert get_persona(np.int64(0))["name"] == "VIP"


def test_unknown_cluster_label_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown K-Means cluster label"):
        get_persona(4)


def test_assign_personas_preserves_label_order():
    assert assign_personas([0, 3, 2]) == ["VIP", "Ngủ đông", "Cần níu kéo"]
