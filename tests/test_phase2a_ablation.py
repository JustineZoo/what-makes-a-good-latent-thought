"""Unit tests for Phase 2a ablation generator across all 5 systems."""
import pytest
pytest.importorskip("transformers")
pytest.importorskip("peft")
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dlm.train.train_phase2_ablation import CoLaRAblationGenerator


def _make_tiny_llama():
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaForCausalLM(cfg)


@pytest.mark.parametrize("system", ["colar", "colar_c1", "colar_c2", "colar_c1_c2", "colar_c1_c2_c3a"])
def test_phase2a_ablation_systems(system):
    torch.manual_seed(42)
    backbone = _make_tiny_llama()
    gen = CoLaRAblationGenerator(backbone, system=system, d_z=16, K=4, head_inner_steps=2)

    q_ids = torch.randint(3, 128, (2, 10))
    q_mask = torch.ones_like(q_ids)
    target_z = torch.randn(2, 4, 16)

    # 1. Forward loss & backward pass
    loss = gen.fit_step(q_ids, q_mask, target_z)
    assert loss.ndim == 0 and not torch.isnan(loss) and loss.item() > 0
    loss.backward()

    # 2. Free-running sample rollout
    gen.eval()
    fr_z = gen.sample(q_ids, q_mask, n_latents=4, steps=2)
    assert fr_z.shape == (2, 4, 16)
