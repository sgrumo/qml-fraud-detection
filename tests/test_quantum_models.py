import numpy as np

from qfd.evaluation import evaluate
from qfd.quantum import (
    QuantumKernelSVC,
    QuantumOneClassSVM,
    StatevectorKernel,
    TrainingConfig,
    VariationalClassifier,
    make_feature_map,
)


def _kernel() -> StatevectorKernel:
    return StatevectorKernel(make_feature_map("iqp", reps=1), n_qubits=3, bandwidth=0.5)


def test_qsvc_ranks_separable_data(
    labelled_angles: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = labelled_angles

    scores = QuantumKernelSVC(_kernel(), C=10.0).fit(X, y).fraud_score(X)

    assert evaluate(y, scores)["auprc"] > 0.9


def test_support_vector_shortcut_matches_full_kernel(
    labelled_angles: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = labelled_angles
    model = QuantumKernelSVC(_kernel()).fit(X, y)

    full = model.svm.decision_function(model.kernel(X, model.X_train_))

    np.testing.assert_allclose(model.fraud_score(X), full, atol=1e-10)


def test_one_class_trains_on_genuine_only(
    labelled_angles: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = labelled_angles

    model = QuantumOneClassSVM(_kernel(), nu=0.1).fit(X, y)

    assert len(model.X_train_) == (y == 0).sum()
    assert model.fraud_score(X).shape == (len(X),)


def test_vqc_trains_and_scores(labelled_angles: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = labelled_angles
    model = VariationalClassifier(
        n_qubits=3, n_layers=2, training=TrainingConfig(epochs=8, batch_size=10)
    )

    scores = model.fit(X, y).fraud_score(X)

    assert scores.shape == (len(X),)
    assert evaluate(y, scores)["roc_auc"] > 0.7
