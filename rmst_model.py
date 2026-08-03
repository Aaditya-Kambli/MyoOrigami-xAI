# Model Architecture Type 2
import math
from os import name
import torch
# torch.nn contains all the neural network building blocks like layers, loss functions, etc.)
import torch.nn as nn
# torch.nn.functional gives the raw operations I might need like activation functions
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=25):
        """
        d_model: 128 (The feature dimension)
        max_len: 25 (The window size/time-steps)
        """
        super(PositionalEncoding, self).__init__()
        
        # 1. Creates the base matrix with the shape [25, 128]
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create position vector [0, 1, 2...24] (0 through 24)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Frequency scaling (This is Standard Transformer math)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Applies Sine to even indices and Cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. Store it as [1, 25, 128] - The standard transformer input 3D shape
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x can be:
        3D: (Batch, 25, 128) -> common for initial limb state
        4D: (Batch, Windows, 25, 128) -> common for EMG sequences
        """
        # Slices pe to match the current window size of x (the second to last dimension)
        # This handles cases where a final window might be shorter than 25ms.
        seq_len = x.size(-2)
        pe_sliced = self.pe[:, :seq_len, :] # Shape: [1, seq_len, 128]
        
        if x.dim() == 4:
            # For (b, n, 25, 128), we unsqueeze pe to (1, 1, 25, 128)
            # This applies the exact same 0-24 time-stamp to EVERY single window.
            return x + pe_sliced.unsqueeze(1)
        
        # For (b, 25, 128), we just add [1, 25, 128] (This adds the positional encoder values to the main data)
        return x + pe_sliced
    
class AxialAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.05):
        """
        embed_dim: 128 (The original feature count)
        num_heads: 8
        """
        super(AxialAttention, self).__init__()
        
        # SCAN 1: Temporal Attention (Looks at the 25 time-steps)
        # Here, the 'features' are the 128 channels.
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # SCAN 2: Feature Attention (Looks at the 128 channels)
        # Here, the data is transposed, so the 'features' are the 25 time-steps.
        # We keep embed_dim=25 and num_heads=5 because 25 is divisible by 5
        # (otherwise we would get a shape mismatch error)
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=25, 
            num_heads=5, 
            dropout=dropout, 
            batch_first=True
        )
        
        # PRE-NORMALIZATION DESIGN
        self.norm_time = nn.LayerNorm(embed_dim)
        self.norm_feature = nn.LayerNorm(25) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (Batch, Windows, 25, 128)
        """
        b, n, w, f = x.shape
        
        # 1. Prepares for Scan: (b*n, 25, 128)
        # This treats each window as an independent sequence
        x_flat = x.reshape(b * n, w, f) 

        # --- STEP 1: TIME-DOMAIN ATTENTION (Pre-Norm) ---
        # Data is in the form of (b*n, 25, 128). Here we attend over the 25-length dimension.
        norm_x = self.norm_time(x_flat)
        t_out, _ = self.temporal_attn(query=norm_x, key=norm_x, value=norm_x)
        x_flat = x_flat + self.dropout(t_out)

        # --- STEP 2: CHANNEL-DOMAIN ATTENTION (Pre-Norm) ---
        # Data is in the shape of (b*n, 25, 128). We transpose to (b*n, 128, 25).
        # Now, the model sees 128 'steps' and 25 'features' which is perfect!
        x_trans = x_flat.transpose(1, 2).contiguous()
        
        # This normalizes the new 'feature' dimension (25)
        norm_x_t = self.norm_feature(x_trans)
        
        # Attention now operates on the 25 features across the 128 channels
        f_out, _ = self.feature_attn(query=norm_x_t, key=norm_x_t, value=norm_x_t)
        
        # Adds residual to the transposed state
        x_trans = x_trans + self.dropout(f_out)
        
        # --- RESTORE AND RETURN ---
        # Flips back to (b*n, 25, 128) effectively untransposing it
        res = x_trans.transpose(1, 2).contiguous()
        
        # Returns the data back to 4D: (batch_size, num_windows, 25, 128)
        return res.reshape(b, n, w, f)
    
class RMCABlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.05):
        super(RMCABlock, self).__init__()
        
        # 1. INTRA-MODAL (Axial Scanning)
        self.emg_axial = AxialAttention(embed_dim, num_heads, dropout)
        self.kin_axial = AxialAttention(embed_dim, num_heads, dropout)
        
        # 2. CROSS-TALK (Fusion) Here the EMG attends to the Kinematic data and Vice Versa
        self.emg_from_kin = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.kin_from_emg = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 3. FEED-FORWARD (Reasoning) These define the feedforward layers where the attention scores are processed
        self.ffn_emg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_kin = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # PRE-NORMALIZATION STABILIZATION 
        self.norm1_emg = nn.LayerNorm(embed_dim)
        self.norm1_kin = nn.LayerNorm(embed_dim)
        self.norm2_emg = nn.LayerNorm(embed_dim)
        self.norm2_kin = nn.LayerNorm(embed_dim)
        self.norm3_emg = nn.LayerNorm(embed_dim)
        self.norm3_kin = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, emg_x, kin_x):
        """
        emg_x: (Batch, Windows, 25, 128)
        kin_x: (Batch, Windows, 25, 128)
        """
        # --- STAGE 1: INTRA-MODAL REFINEMENT (Pre-Norm) ---
        # AxialAttention handles the 4D input and returns 4D output
        # We wrap the whole operation in the residual connection
        emg_x = emg_x + self.dropout(self.emg_axial(self.norm1_emg(emg_x)))
        kin_x = kin_x + self.dropout(self.kin_axial(self.norm1_kin(kin_x)))

        # --- STAGE 2: CROSS-TALK FUSION (Pre-Norm) ---
        b, n, w, f = emg_x.shape
        # Flattens to (b, n*w, F) so now MultiheadAttention can see the whole sequence
        emg_flat = emg_x.view(b, n * w, f) 
        kin_flat = kin_x.view(b, n * w, f)

        # Applies normalization to the flattened features
        norm_e_flat = self.norm2_emg(emg_flat)
        norm_k_flat = self.norm2_kin(kin_flat)

        # Cross-modal communication
        # EMG queries Kinematics (What muscle signals explain this movement?)
        attn_emg, _ = self.emg_from_kin(query=norm_e_flat, key=norm_k_flat, value=norm_k_flat)
        # Kinematics queries EMG (What movement follows this muscle intent?)
        attn_kin, _ = self.kin_from_emg(query=norm_k_flat, key=norm_e_flat, value=norm_e_flat)
        
        # Residual connections on flattened tensors
        emg_fused = emg_flat + self.dropout(attn_emg)
        kin_fused = kin_flat + self.dropout(attn_kin)

        # --- STAGE 3: FEED-FORWARD REASONING (Pre-Norm) ---
        # We stay in flattened 3D space for the Linear layers (it's more efficient)
        emg_out = emg_fused + self.dropout(self.ffn_emg(self.norm3_emg(emg_fused)))
        kin_out = kin_fused + self.dropout(self.ffn_kin(self.norm3_kin(kin_fused)))

        # --- STAGE 4: RESTORE TO 4D ---
        # Crucial for the next RMCABlock or the loop in the main Transformer
        emg_final = emg_out.view(b, n, w, f)
        kin_final = kin_out.view(b, n, w, f)

        return emg_final, kin_final

class MultimodalSparseTransformer(nn.Module):
    def __init__(self, emg_dim=14, kin_dim=35, embed_dim=128, num_blocks=4):
        super(MultimodalSparseTransformer, self).__init__()
        
        # 1. GATED RECYCLING COMPONENTS
        # This decides how much "Old Intent" to keep vs "New EMG" to listen to
        self.emg_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid() 
        )
        
        # This stabilizes the recycled state before it meets the new input
        self.emg_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Drift resilience for real-world sensor jitter
        self.noise_std = 0.01

        # 2. GATEWAYS & SCALING
        self.emg_embed = nn.Linear(emg_dim, embed_dim)
        self.kin_embed = nn.Linear(kin_dim, embed_dim)
        self.embed_scale = math.sqrt(embed_dim) # Aligned scaling factor

         # --- THE TDC ADDITION ---
        # groups=embed_dim makes this a "Depthwise" convolution (efficient & focused)
        # kernel_size=3 allows it to look at (t-1, t, t+1) to find the slope/velocity.
        self.tdc = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            kernel_size=3, 
            padding=1, 
            groups=embed_dim
        )
        
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 3. THE REFINEMENT TRUNK
        self.rmca_stack = nn.ModuleList([
            RMCABlock(embed_dim) for _ in range(num_blocks)
        ])
        
        self.out_head = nn.Linear(embed_dim, 35)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Now we can check if the layer's name contains "emg_gate"
                    if "emg_gate" in name:
                        nn.init.constant_(m.bias, 2.0) # Start with "Memory Faucet" Open
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, emg_seq, initial_kin, recursion_depth=4, teacher_forcing_kin=None):
        # 1. DATA GUARD: Ensure the input hasn't been corrupted or mis-shaped
        assert emg_seq.shape[-1] == 14, f"EMG Channels missing! Expected 14, got {emg_seq.shape[-1]}"

        # --- 2. GLOBAL SCALING & TDC REFINEMENT ---
        # We embed the raw signal, then apply TDC to "glue" the time steps
        emg_features = self.emg_embed(emg_seq) * self.embed_scale
        
        b, n, w, f = emg_features.shape

        # --- THE CRASH PREVENTER ---
        if n == 0:
            # Return an empty but correctly shaped tensor [Batch, 0, 35]
            return torch.zeros((b, 0, 35)).to(emg_seq.device)
        
        # --- 1. THE COLD START GUARD RAIL ---
        # We no longer repeat. We assume initial_kin is a full [B, 25, 35] window.
        if initial_kin.dim() == 4:
            initial_kin = initial_kin.squeeze(1) # Ensure (B, 25, 35)

        emg_reshaped = emg_features.view(b * n, w, f).transpose(1, 2) # (B*N, 128, 25)
        emg_tdc = self.tdc(emg_reshaped).transpose(1, 2)              # Back to (B*N, 25, 128)
        emg_h = self.pos_encoder(emg_tdc.view(b, n, w, f))
        # --- 2. GLOBAL SCALING ---
        # Scaling is applied ONCE here to ensure all downstream math is balanced
        # emg_h = self.pos_encoder(self.emg_embed(emg_seq) * self.embed_scale)

        # current_kin_h = self.pos_encoder(self.kin_embed(initial_kin) * self.embed_scale).unsqueeze(1)
        current_emg_h = emg_h[:, 0:1, :, :] 

        predictions = []

        # --- 3. THE GATED RECYCLING LOOP (PROSTHETIC-LOGIC VERSION) ---
        # We start with the actual physical state (initial_kin)
        # Ensure current_kin_phys is 4D (Batch, 1, 25, 35) for the embedder
        current_kin_phys = initial_kin.unsqueeze(1) if initial_kin.dim() == 3 else initial_kin

        for i in range(n):
            x_emg_raw = emg_h[:, i:i+1, :, :] 
            
            # EMG Gating Logic (Keep your existing gating code here)
            if i > 0:
                if self.training:
                    current_emg_h = current_emg_h + torch.randn_like(current_emg_h) * self.noise_std
                combined = torch.cat([x_emg_raw, current_emg_h], dim=-1)
                gate_weight = self.emg_gate(combined)
                x_emg = gate_weight * x_emg_raw + (1 - gate_weight) * self.emg_transform(current_emg_h)
            else:
                x_emg = x_emg_raw
            
            # --- THE CRITICAL FIX: PHYSICAL ANCHORING ---
            if teacher_forcing_kin is not None:
                # Training: Use truth to "guide" the loop
                actual_window = teacher_forcing_kin[:, i:i+1, :, :]
                x_kin_input = self.pos_encoder(self.kin_embed(actual_window) * self.embed_scale)
            else:
                # Deployment/Autonomy: The "anchor" is the PREVIOUS physical state
                x_kin_input = self.pos_encoder(self.kin_embed(current_kin_phys) * self.embed_scale)
            
            # RMCA Refinement
            x_kin = x_kin_input # Seed the refinement with our anchored state

            for block in self.rmca_stack[:recursion_depth]:
                x_emg, x_kin = block(x_emg, x_kin)
            
            # --- PROJECT TO PHYSICS ---
            # Project the refined 128-dim state back to 35 joint angles
            # DO NOT USE TANH HERE (as discussed previously for RMSE stability)
            # 1. The RAW output (This is what was hitting 10^10)
            predicted_phys = self.out_head(x_kin)
            

            # 2. Apply the BRAKES (LayerNorm)
            # We take the raw output and force it to behave.
            # Now 'stable_phys' is in the range of -3 to +3 (Z-scores).
            stable_phys = torch.nn.functional.layer_norm(predicted_phys, (35,))
            # stable_phys = self.out_head(x_kin) # Raw Linear Projection as we are now using l_phys to keep the movement biologically plausible.
            predictions.append(stable_phys)
            
            # RECYCLE: Update the physical state for the NEXT window (i+1)
            if self.training:
                current_kin_phys = stable_phys # Use its own prediction to learn
                current_emg_h = x_emg 
            else:
                current_kin_phys = stable_phys.detach() # Safety for inference
                current_emg_h = x_emg.detach()

        # --- 4. OUTPUT (Memory-Safe Reshape) ---
        full_recon = torch.cat(predictions, dim=1) 
        # out = self.out_head(full_recon)
        # res = out.reshape(b, n * w, 35)
        
        # reshape is safer for real-time deployment than contiguous().view()
        # return out.reshape(b, n * w, 35)
        return full_recon.reshape(b, n * w, 35)

def run_smoke_test():
    # 1. SET THE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 2. INITIALIZE MODEL
    # emg_dim=14, kin_dim=35, embed_dim=128
    model = MultimodalSparseTransformer().to(device)
    model.eval() # Set to evaluation mode

    # 3. CREATE "FAKE" DATA
    # Scenario: 1 Trial, 10 Windows, 25ms per window, 14 EMG sensors
    batch_size = 1
    num_windows = 10
    window_size = 25
    emg_sensors = 14
    kin_sensors = 35

    # Fake EMG: (B, N, W, F)
    fake_emg = torch.randn(batch_size, num_windows, window_size, emg_sensors).to(device)
    
    # Fake "Real-Time" Seed: Only 1 frame of kinematics (B, 1, F)
    # This tests your "Cold Start" handler logic!
    fake_kin_seed = torch.randn(batch_size, 25, kin_sensors).to(device)

    print("\n--- Starting Forward Pass ---")
    try:
        with torch.no_grad():
            # We want to predict 17 joints (1 arm)
            output = model(fake_emg, fake_kin_seed)
        
        # 4. VALIDATE OUTPUT
        # Expected shape: (Batch, Total_Time_Steps, Joints)
        # Total_Time_Steps = num_windows (10) * window_size (25) = 250
        expected_shape = (batch_size, num_windows * window_size, 35)
        
        print(f"Model Output Shape: {output.shape}")
        
        if output.shape == expected_shape:
            print("\nSMOKE TEST PASSED!")
        else:
            print(f"\nSHAPE MISMATCH: Expected {expected_shape}, got {output.shape}")

    except Exception as e:
        print("\nSMOKE TEST FAILED!")
        print(f"Error Message: {e}")
        # This will tell us EXACTLY which line in your architecture is broken.

if __name__ == "__main__":
    run_smoke_test()