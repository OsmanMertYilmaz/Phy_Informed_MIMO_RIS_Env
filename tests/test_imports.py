def test_public_imports():
    import ris_env
    assert hasattr(ris_env, "ArraySpec")
    assert hasattr(ris_env, "BankInput")
    assert hasattr(ris_env, "run_symmetric_gg_label_engine")
