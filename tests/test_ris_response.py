import torch
from ris_env.ris_response import generate_ris_response_from_z

def test_binary_ris_response_shapes_and_finite():
    z = torch.tensor([[0,1,0,1],[1,1,0,0]], dtype=torch.long)
    out = generate_ris_response_from_z(z, device="cpu", parity=True)
    gamma = out["gamma"]
    assert gamma.shape == (2,4)
    assert torch.isfinite(gamma.real).all()
    assert torch.isfinite(gamma.imag).all()
