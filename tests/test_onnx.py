import numpy as np
import onnxruntime as ort
import torch

from clpl.policies import Denoiser


def test_denoiser_onnx_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    net = Denoiser(14).eval()
    path = str(tmp_path / "d.onnx")
    x, t, o = torch.randn(1, 8, 3), torch.zeros(1, dtype=torch.int64), torch.randn(1, 14)
    torch.onnx.export(net, (x, t, o), path, input_names=["x", "t", "obs"], output_names=["eps"],
                      dynamic_axes={n: {0: "b"} for n in ("x", "t", "obs", "eps")}, opset_version=17, dynamo=False)
    xs, ts, os_ = torch.randn(64, 8, 3), torch.randint(0, 100, (64,)), torch.randn(64, 14)
    with torch.no_grad():
        ref = net(xs, ts, os_).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        None, {"x": xs.numpy(), "t": ts.numpy(), "obs": os_.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-4
    assert np.abs(ref).max() > 0.01  # the comparison is not between two all-zero outputs
