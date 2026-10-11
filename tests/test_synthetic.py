from anchordiff.synthetic import make_fixture


def test_synthetic_fixture_is_deterministic_and_contains_ground_truth():
    first = make_fixture(seed=7, size=(96, 96), clutter=True)
    second = make_fixture(seed=7, size=(96, 96), clutter=True)

    assert first.before.shape == (96, 96, 3)
    assert (first.before == second.before).all()
    assert (first.after == second.after).all()
    assert first.metadata["ground_truth"]["residual_regions"] >= 1


def test_privacy_safe_benchmark_declares_synthetic_cpu_backend(tmp_path, monkeypatch):
    import json

    from examples.run_benchmark import main

    monkeypatch.chdir(tmp_path)
    main()

    report = json.loads((tmp_path / "benchmark-report.json").read_text(encoding="utf-8"))
    assert report["privacy"] == "synthetic-only"
    assert report["backend"] == "cpu"
    assert len(report["rows"]) == 10
