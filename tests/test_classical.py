import pytest

from qfd.classical import CLASSICAL_MODELS, build_classical
from qfd.data import Dataset
from qfd.evaluation import evaluate


@pytest.mark.parametrize("name", CLASSICAL_MODELS)
def test_model_beats_random_ranking(dataset: Dataset, name: str) -> None:
    X_train, y_train = dataset.X[:2000], dataset.y[:2000]
    X_test, y_test = dataset.X[2000:], dataset.y[2000:]
    detector = build_classical(name, {}, seed=0)

    scores = detector.fit(X_train, y_train).fraud_score(X_test)

    assert scores.shape == (len(X_test),)
    assert evaluate(y_test, scores)["auprc"] > 2 * y_test.mean()


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        build_classical("does_not_exist", {})


def test_params_override_defaults() -> None:
    detector = build_classical("random_forest", {"n_estimators": 7}, seed=3)

    params = detector.estimator.get_params()
    assert params["n_estimators"] == 7
    assert params["random_state"] == 3
