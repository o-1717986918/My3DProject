from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "training"))

from training.tools.generate_kick_transition_corpus import (
    phase_buckets,
    stratified_rollout_split,
)


def test_phase_buckets_cover_the_cardinal_cycle():
    phases = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
    )

    np.testing.assert_array_equal(phase_buckets(phases, 8), [0, 2, 4, 6])


def test_rollout_split_is_reproducible_and_stratified():
    buckets = np.repeat(np.arange(4), 5)
    first = stratified_rollout_split(
        buckets, seed=71, validation_fraction=0.2
    )
    second = stratified_rollout_split(
        buckets, seed=71, validation_fraction=0.2
    )

    np.testing.assert_array_equal(first, second)
    for bucket in range(4):
        selected = first[buckets == bucket]
        assert np.count_nonzero(selected == 0) == 4
        assert np.count_nonzero(selected == 1) == 1


def test_split_and_phase_buckets_reject_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        phase_buckets(np.zeros((3, 3)), 8)
    with pytest.raises(ValueError, match="fraction"):
        stratified_rollout_split(
            np.zeros(4, dtype=np.int32), seed=1, validation_fraction=0.5
        )
