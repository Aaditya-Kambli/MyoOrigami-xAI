import torch
from origami_controller import OrigamiXAIController

controller = OrigamiXAIController()

resting = torch.randn(1, 5, 25, 14) * 0.01

gesture = torch.randn(1, 5, 25, 14) * 0.01  # start with noise
gesture[:, :, :, :5] = torch.randn(1, 5, 25, 5) * 2.0  # activate only first 5 channels

chaotic = torch.randn(1, 5, 25, 14)
for ch in range(14):
    gesture_ch_scale = 0.1 + ch * 0.5  # channels 0-13 get scales 0.1 to 6.6
    chaotic[:, :, :, ch] = torch.randn(1, 5, 25) * gesture_ch_scale

print("=== RESTING (should be low H_data, Tier 1) ===")
rest_h = controller.calculate_h_data(resting)
for i, h in enumerate(rest_h[0]):
    print(f"  Window {i}: H={h.item():.4f} → Tier {controller.compute_recursion_tier(h.item())}")

print("\n=== GESTURE (should be HIGHER H_data — active vs inactive channels) ===")
gest_h = controller.calculate_h_data(gesture)
for i, h in enumerate(gest_h[0]):
    print(f"  Window {i}: H={h.item():.4f} → Tier {controller.compute_recursion_tier(h.item())}")

print("\n=== CHAOTIC (should be HIGHEST H_data — all channels different) ===")
chao_h = controller.calculate_h_data(chaotic)
for i, h in enumerate(chao_h[0]):
    print(f"  Window {i}: H={h.item():.4f} → Tier {controller.compute_recursion_tier(h.item())}")
