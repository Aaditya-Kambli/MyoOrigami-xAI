import os, sys, time
import torch
import torch.nn.utils.prune as prune
import numpy as np

# Paths 
base = r"C:\Users\KrGT7mXfZaN7531vW\Desktop\SRP_2025_2026\SRP_Progams_Executables"
origami_dir = os.path.join(base, "origami_xai")

if base not in sys.path:
    sys.path.insert(0, base)

# Imports
from rmst_model import MultimodalSparseTransformer
from origami_controller import OrigamiXAIController
from origami_inference import run_origami_inference

# 1. Load model with pruning fix 
def finalize_model(model):
    """Converts parametrized weights (from Phase 2 pruning) into standard weights."""
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            prune.remove(module, 'weight')
    return model

print(">>> Loading model...")
model = MultimodalSparseTransformer(emg_dim=14, kin_dim=35, embed_dim=128)
model.load_state_dict(
    torch.load(r"C:\Users\KrGT7mXfZaN7531vW\Desktop\SRP_2025_2026\SRP_Progams_Executables\origami_xai\checkpoints\rmst_weights.pth", map_location="cpu"),
    strict=False  
)
model = finalize_model(model)  
model.eval()
print(">>> Model loaded and finalized.")

# 2. Load global stats for un-z-scoring
data_dir = os.path.join(base, "data", "MOVMUS-UJI_DATASET", "DATASET")
g_mean = np.load(os.path.join(data_dir, "global_kinematic_mean.npy"))
g_std  = np.load(os.path.join(data_dir, "global_kinematic_std.npy"))

# 3. Load real test data
test_dir = os.path.join(data_dir, "Processed_Tasks_test1")
emg_data   = np.load(os.path.join(test_dir, "task_0001_emg.npy"))    # [W, 25, 14]
angle_data = np.load(os.path.join(test_dir, "task_0001_angle.npy"))   # [W, 25, 35]

# Take 9 windows, add batch dim
emg_input = torch.tensor(emg_data[:9][np.newaxis, ...], dtype=torch.float32)  # [1, 9, 25, 14]

seed_window = torch.tensor(angle_data[0:1][np.newaxis, ...], dtype=torch.float32)  # [1, 1, 25, 35]

print(f"EMG shape: {emg_input.shape}")       
print(f"Seed shape: {seed_window.shape}")      

# 4. Instantiate controller
controller = OrigamiXAIController()

# 5. Run inference
print(">>> Running Origami inference...")
output, h_data, tier = run_origami_inference(model, emg_input, seed_window, controller)

# 6. Channel attribution
print(">>> Computing channel attribution...")
channel_importance = controller.compute_channel_attribution(model, emg_input, seed_window, joint_index=0)

# 7. Results
print("\n" + "="*50)
print(f"Output Shape:     {output.shape}")           
print(f"Recursion Tier:   {tier}")
print(f"H_data shape:     {h_data.shape}")           
print(f"H_data mean:      {h_data.mean().item():.4f}")
print(f"Channel Importance (%): {channel_importance}")
print(f"Sum of importance:      {channel_importance.sum():.1f}%")
print("="*50)

pred_deg = (output[0, 0, 0].item() * g_std[0]) + g_mean[0]
print(f"\nJoint 0 prediction (degrees): {pred_deg:.2f}°")
print(">>> DONE — pipeline works on real data!")
