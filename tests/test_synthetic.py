from anchordiff.synthetic import make_fixture


def test_synthetic_fixture_is_deterministic_and_contains_ground_truth():
    first = make_fixture(seed=7, size=(96, 96), clutter=True)
    second = make_fixture(seed=7, size=(96, 96), clutter=True)

    assert first.before.shape == (96, 96, 3)
    assert (first.before == second.before).all()
    assert (first.after == second.after).all()
    assert first.metadata["ground_truth"]["residual_regions"] >= 1
