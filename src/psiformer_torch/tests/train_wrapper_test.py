import pytest

from psiformer_torch.config import large_conf
from psiformer_torch.train import wrapper


def test_wrapper_does_not_mutate_preset_singletons():
    # Baseline values from the module-level Train_Config instance.
    assert large_conf[1].run_name == "Train"
    assert large_conf[1].checkpoint_name == ""

    _, cfg_a = wrapper("large", run_name="RunA", checkpoint_name="CkptA")
    _, cfg_b = wrapper("large", run_name="RunB", checkpoint_name="CkptB")

    assert cfg_a.run_name == "RunA_LARGE"
    assert cfg_b.run_name == "RunB_LARGE"

    # Ensure we didn't mutate the global preset objects.
    assert large_conf[1].run_name == "Train"
    assert large_conf[1].checkpoint_name == ""


def test_wrapper_rejects_unknown_preset():
    with pytest.raises(ValueError):
        wrapper("not_a_real_preset")
