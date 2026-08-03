from xml.parsers.expat import model

import torch
import torch.nn as nn

class OrigamiXAIController(nn.Module):
    def __init__(self):
        super(OrigamiXAIController, self).__init__()
        self.register_buffer("lambda_constant", torch.tensor(0.5))
        self.register_buffer("base_threshold", torch.tensor(1.2))

    def calculate_h_data(self, emg_input):
        # Four Step Calculation
        # (1) Take your EMG input of shape [Batch, Windows, 25_samples, 14_channels]
        # (2)Compute `torch.var()` across the 25 time samples (dim=2) to get spatial variance per channel
        # (3) Take `torch.log()` of that variance (this is the "log-envelope" that mimics how biologists measure muscle activation)
        # (4) Compute `torch.std()` of the log-envelope across channels to get a single scalar representing spatial chaos
        log_envelope = torch.log(torch.var(emg_input, dim=2) + 1e-6)  # Adding a small constant to avoid log(0)
        h_data = torch.std(log_envelope, dim=-1)  # Compute std across channels
        return h_data

    def compute_recursion_tier(self, h_data, user_lambda_override=None):
        current_lambda = user_lambda_override if user_lambda_override is not None else self.lambda_constant.item()
        adjusted_threshold = self.base_threshold.item() * current_lambda
        # Determine recursion tier
        if h_data < adjusted_threshold:
            return 1
        elif h_data < adjusted_threshold * 1.8:
            return 2
        else:
            return 3

    # Create a randomized test function to validate the calculations
    def test_calculations(self):
        # Generate random EMG input of shape [Batch, Windows, 25_samples, 14_channels]
        emg_input = torch.randn(10, 5, 25, 14)  # Example: Batch=10, Windows=5
        h_data = self.calculate_h_data(emg_input)
        recursion_tiers = [self.compute_recursion_tier(h) for h in h_data.flatten()]
        return h_data, recursion_tiers

    def compute_channel_attribution(self, model, emg_input, initial_kin, joint_index=0):
        emg_input = emg_input.clone().detach().requires_grad_(True)
        output = model(emg_input, initial_kin)  
        target = output[0, 0, joint_index]  
        target.backward(retain_graph=True)
        # Gradient shape matches input: [B, N, 25, 14]
        grad = emg_input.grad
        # Average absolute gradient over batch, windows, time → one value per channel
        channel_importance = grad.abs().mean(dim=(0, 1, 2))  # Shape: [14]
        channel_importance = channel_importance / channel_importance.sum() * 100
        return channel_importance.detach().numpy()
    
if __name__ == "__main__":
    controller = OrigamiXAIController()
    h_data, recursion_tiers = controller.test_calculations()
    print("H Data:", h_data)
    print("Recursion Tiers:", recursion_tiers)
