import numpy as np
import pytest

from qfd.feature_selection import QuboFeatureSelector, build_selection_bqm


def test_bqm_ground_state_selects_k_most_relevant() -> None:
    import dimod

    relevance = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    redundancy = np.zeros((5, 5))

    bqm = build_selection_bqm(relevance, redundancy, k=3, alpha=0.5, penalty=2.0)
    best = dimod.ExactSolver().sample(bqm).first.sample

    assert [j for j in range(5) if best[j]] == [0, 2, 4]


def test_redundancy_penalty_avoids_duplicates() -> None:
    import dimod

    relevance = np.array([1.0, 1.0, 0.8])
    redundancy = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])

    bqm = build_selection_bqm(relevance, redundancy, k=2, alpha=2.0, penalty=2.0)
    best = dimod.ExactSolver().sample(bqm).first.sample

    assert best[2] == 1
    assert best[0] + best[1] == 1


@pytest.mark.parametrize("sampler", ["exact", "simulated"])
def test_selector_keeps_informative_features(
    rng: np.random.Generator, sampler: str
) -> None:
    y = rng.integers(0, 2, size=600)
    noise = rng.normal(size=(600, 4))
    signal = y[:, None] + 0.3 * rng.normal(size=(600, 2))
    X = np.hstack([noise[:, :2], signal, noise[:, 2:]])

    selector = QuboFeatureSelector(k=2, sampler=sampler, num_reads=50).fit(X, y)

    assert list(selector.support_) == [2, 3]
    assert selector.transform(X).shape == (600, 2)
