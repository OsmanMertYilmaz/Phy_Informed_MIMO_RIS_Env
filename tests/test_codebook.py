from ris_env.codebook import generate_codebook_rank1, flatten_codebook_matlab_loop_order

def test_four_port_codebook_has_at_least_32_entries():
    cb = generate_codebook_rank1(2,2,1,1,1,device="cpu",parity=True)
    W, idx = flatten_codebook_matlab_loop_order(cb)
    assert W.shape[0] == 4
    assert W.shape[1] >= 32
    assert idx.shape[0] == W.shape[1]
