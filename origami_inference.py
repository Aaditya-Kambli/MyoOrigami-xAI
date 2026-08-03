import torch

def run_origami_inference(model, emg_input, initial_kin, controller, user_lambda=None):
    model.eval()
    with torch.no_grad():
        h_data = controller.calculate_h_data(emg_input)  # Shape: [B, W]
        h_scalar = h_data.mean().item()                   # ← ADD THIS LINE
        tier = controller.compute_recursion_tier(h_scalar, user_lambda)  # ← USE h_scalar
        depth_map = {1: 1, 2: 2, 3: 4}
        output = model(emg_input, initial_kin, recursion_depth=depth_map[tier])
    return output, h_data, tier