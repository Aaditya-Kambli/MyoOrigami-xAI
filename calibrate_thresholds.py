import glob
import numpy as np
import torch
from origami_controller import OrigamiXAIController

controller = OrigamiXAIController()
test_files = glob.glob(r"C:\Users\KrGT7mXfZaN7531vW\Desktop\SRP_2025_2026\SRP_Progams_Executables\origami_xai\test_data\Processed_Tasks_test1\*_emg.npy")
all_h = []

for f in test_files:
    data = np.load(f)
    batch = torch.tensor(data[np.newaxis, ...], dtype=torch.float32)
    h = controller.calculate_h_data(batch)  # [1, W]
    all_h.append(h.flatten().numpy())

all_h = np.concatenate(all_h)
T1, T2 = np.percentile(all_h, [33, 66])
print(f"H_data range: {all_h.min():.3f} to {all_h.max():.3f}")
print(f"T1={T1:.3f}, T2={T2:.3f}")

controller.base_threshold.data.fill_(T1)
print(f"Updated base_threshold to {T1:.3f}")