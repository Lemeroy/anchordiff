from __future__ import annotations

from examples.run_edge import build_parser


def test_edge_example_exposes_onnx_and_rknn_backends():
    parser = build_parser()

    onnx_args = parser.parse_args(["--backend", "onnx", "--model", "model.onnx", "--before", "b.png", "--after", "a.png", "--output", "out"])
    rknn_args = parser.parse_args(["--backend", "rknn", "--model", "model.rknn", "--before", "b.png", "--after", "a.png", "--output", "out"])

    assert onnx_args.backend == "onnx"
    assert rknn_args.backend == "rknn"
