import torch
from dlm.generators.ar import EntitySlotBuffer

mod = EntitySlotBuffer(d_model=256, n_slots=5, tau=0.1)

# Verify prompt-conditioned initialization
dummy_cond = torch.randn(8, 8, 256)
slots_0 = mod.init_slots(dummy_cond)
assert slots_0.shape == (8, 5, 256), f"Initial slot shape mismatch: {slots_0.shape}"

# Verify autograd flow through routing and updates
dummy_h = torch.randn(8, 1, 256, requires_grad=True)
h_out, next_slots = mod(dummy_h, slots_0)

loss = h_out.sum() + next_slots.sum()
loss.backward()

assert dummy_h.grad is not None, "Gradients failed to backpropagate!"
assert next_slots.shape == (8, 5, 256), f"Updated slot shape mismatch: {next_slots.shape}"

params = sum(p.numel() for p in mod.parameters())
print(f"Patched Slot Buffer tests PASSED! Parameters: {params:,}")
