import torch
from transformers import LlamaConfig, LlamaForCausalLM
from dlm.models.llm_codec import LLMCodecConfig, LLMLatentCodec, LLMLatentGenerator


def _tiny_llama():
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


def test_llm_codec_and_generator_shapes():
    torch.manual_seed(0)
    codec_cfg = LLMCodecConfig(model_name="tiny", R=4, d_z=16, max_cot_len=32)
    codec = LLMLatentCodec(_tiny_llama(), _tiny_llama(), codec_cfg)

    q_ids = torch.randint(3, 128, (2, 10))
    q_mask = torch.ones_like(q_ids)
    cot_ids = torch.randint(3, 128, (2, 16))
    cot_mask = torch.ones_like(cot_ids)
    labels = cot_ids.clone()

    total_loss, rec_loss, kl_loss = codec.compute_loss(q_ids, q_mask, cot_ids, cot_mask, labels)
    assert total_loss.ndim == 0 and rec_loss > 0 and kl_loss > 0
    total_loss.backward()

    codec.eval()
    z_gold = codec.encode_standardized(cot_ids, cot_mask)
    assert z_gold.shape == (2, 4, 16)

    gen = LLMLatentGenerator(_tiny_llama(), d_z=16, head_inner_steps=2)
    flow_loss = gen.fit_step(q_ids, q_mask, z_gold)
    assert flow_loss.ndim == 0
    flow_loss.backward()

    gen.eval()
    tf_z = gen.teacher_forced_sample(q_ids, q_mask, z_gold, steps=2)
    fr_z = gen.sample(q_ids, q_mask, n_latents=4, steps=2)
    assert tf_z.shape == (2, 4, 16)
    assert fr_z.shape == (2, 4, 16)

    decoded = codec.decode_logits(codec.unstandardize(fr_z), q_ids, q_mask, out_len=16)
    assert decoded.shape == (2, 16, 128)
