from types import SimpleNamespace

import torch

import ris_env.stateful_mc as sm


class FakeCfg:
    def __init__(self, kind):
        self.kind = kind


def fake_row_to_link_configs(row):
    return FakeCfg("br"), FakeCfg("ru")


def fake_generate_native_link_chunk(
    cfg,
    N,
    *,
    generator,
    device=None,
    parity=False,
):
    dev = torch.device(device)

    rd = (
        torch.float64
        if parity
        else torch.float32
    )

    cd = (
        torch.complex128
        if parity
        else torch.complex64
    )

    # HBR: [N,I,T] = [N,3,2]
    if cfg.kind == "br":
        shape = (N, 3, 2)

    # HRU: [N,R,I] = [N,1,3]
    else:
        shape = (N, 1, 3)

    re = torch.randn(
        shape,
        generator=generator,
        dtype=rd,
        device=dev,
    )

    im = torch.randn(
        shape,
        generator=generator,
        dtype=rd,
        device=dev,
    )

    return torch.complex(
        re,
        im,
    ).to(cd)


def test_stateful_split_matches_one_shot(
    monkeypatch,
):

    monkeypatch.setattr(
        sm,
        "row_to_link_configs",
        fake_row_to_link_configs,
    )

    monkeypatch.setattr(
        sm,
        "generate_native_link_chunk",
        fake_generate_native_link_chunk,
    )


    row = SimpleNamespace(
        nT1=1,
        nT2=1,
        nRIS=3,
        ch_seed=123,
    )


    W = torch.tensor(
        [
            [1+0j, 0+0j],
            [0+0j, 1+0j],
        ],
        dtype=torch.complex64,
    )


    gamma = torch.tensor(
        [
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1,  1, -1],
            [-1, -1,  1],
        ],
        dtype=torch.complex64,
    )


    # ====================================================
    # ONE-SHOT 32
    # ====================================================

    one = sm.create_stateful_mc(
        row,
        W,
        gamma,
        mc_chunk=8,
        w_chunk=1,
        z_chunk=2,
        device="cpu",
        parity=False,
    )

    sm.advance_stateful_mc(
        one,
        additional_samples=32,
    )

    one_out = sm.snapshot_stateful_mc(
        one
    )


    # ====================================================
    # STATEFUL 16 + 16
    # ====================================================

    split = sm.create_stateful_mc(
        row,
        W,
        gamma,
        mc_chunk=8,
        w_chunk=1,
        z_chunk=2,
        device="cpu",
        parity=False,
    )


    sm.advance_stateful_mc(
        split,
        additional_samples=16,
    )

    first = sm.snapshot_stateful_mc(
        split
    )

    assert first["n_samples"] == 16


    sm.advance_stateful_mc(
        split,
        additional_samples=16,
    )

    split_out = sm.snapshot_stateful_mc(
        split
    )

    assert split_out["n_samples"] == 32


    # ====================================================
    # EXACT PARITY
    # ====================================================

    assert torch.equal(
        one.sum1,
        split.sum1,
    )

    assert torch.equal(
        one.sum2,
        split.sum2,
    )

    assert torch.equal(
        one_out["meanEmp"],
        split_out["meanEmp"],
    )

    assert torch.equal(
        one_out["varEmp"],
        split_out["varEmp"],
    )


def test_paired_gamma_matches_independent_w_runs(monkeypatch):
    monkeypatch.setattr(sm, "row_to_link_configs", fake_row_to_link_configs)
    monkeypatch.setattr(
        sm, "generate_native_link_chunk", fake_generate_native_link_chunk
    )
    row = SimpleNamespace(nT1=1, nT2=1, nRIS=3, ch_seed=321)
    w = torch.tensor(
        [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]],
        dtype=torch.complex64,
    )
    shared = torch.tensor(
        [[1, 1, 1], [1, -1, 1], [-1, 1, -1]],
        dtype=torch.complex64,
    )
    paired_gamma = torch.stack([shared, -shared], dim=0)
    paired = sm.create_stateful_mc(
        row, w, paired_gamma, mc_chunk=8, w_chunk=2, z_chunk=2,
        device="cpu", parity=False,
    )
    sm.advance_stateful_mc(paired, 32)
    paired_out = sm.snapshot_stateful_mc(paired)

    for k in range(2):
        single = sm.create_stateful_mc(
            row, w[k:k + 1], paired_gamma[k], mc_chunk=8,
            w_chunk=1, z_chunk=2, device="cpu", parity=False,
        )
        sm.advance_stateful_mc(single, 32)
        single_out = sm.snapshot_stateful_mc(single)
        assert torch.allclose(
            paired_out["meanEmp"][k], single_out["meanEmp"][0]
        )
        assert torch.allclose(
            paired_out["varEmp"][k], single_out["varEmp"][0]
        )
