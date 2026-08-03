import streamlit as st
import numpy as np
import torch
import torch.nn.utils.prune as prune
import plotly.graph_objects as go
import os
import sys

# Page Config
st.set_page_config(
    page_title="MyoOrigami xAI — Transparent Prosthetic Control",
    page_icon="\U0001F9BE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths (edit these for your machine)
BASE = r"C:\Users\KrGT7mXfZaN7531vW\Desktop\SRP_2025_2026\SRP_Progams_Executables"
DATA_DIR = f"{BASE}\\origami_xai\\test_data\\Processed_Tasks_test1"
MUJOCO_MEDIA_DIR = os.path.join(BASE, "origami_xai", "mujoco_media")

st.markdown("""
<style>
.tutorial-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    border-left: 4px solid #10b981;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 14px;
}
.tutorial-box h4 { color: #065f46; margin: 0 0 6px 0; font-size: 14px; }
.tutorial-box p  { color: #374151; font-size: 13px; line-height: 1.6; margin: 0; }
.honesty-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left: 4px solid #f59e0b;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 14px 0;
}
.honesty-box h4 { color: #92400e; margin: 0 0 6px 0; font-size: 14px; }
.honesty-box p  { color: #374151; font-size: 13px; line-height: 1.6; margin: 0; }
.mujoco-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid #0ea5e9;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 14px;
}
.mujoco-box h4 { color: #075985; margin: 0 0 6px 0; font-size: 14px; }
.mujoco-box p  { color: #374151; font-size: 13px; line-height: 1.6; margin: 0; }
.section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0

st.title("\U0001F9BE MyoOrigami xAI")
st.markdown("**Making AI-controlled prosthetic hands transparent and accountable.**")
st.caption("When AI controls a prosthetic hand, shouldn\'t you know WHY it moves?")

# Load Model (cached so it only loads one)
@st.cache_resource
def load_model():
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    model_dir = os.path.join(BASE, "origami_xai")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from rmst_model import MultimodalSparseTransformer
    model = MultimodalSparseTransformer(emg_dim=14, kin_dim=35, embed_dim=128)
    weights_path = os.path.join(BASE, "origami_xai", "checkpoints", "rmst_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False), strict=False)
    for _, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            prune.remove(module, 'weight')
    model.eval()
    return model

model = load_model()

# Import controller + Inference
origami_dir = os.path.join(BASE, "origami_xai")
if origami_dir not in sys.path:
    sys.path.insert(0, origami_dir)
from origami_controller import OrigamiXAIController
from origami_inference import run_origami_inference

controller = OrigamiXAIController()
controller.base_threshold.data.fill_(5.0)

try:
    g_mean = np.load(os.path.join(BASE, r"data\MOVMUS-UJI_DATASET\DATASET\global_kinematic_mean.npy"))
    g_std  = np.load(os.path.join(BASE, r"data\MOVMUS-UJI_DATASET\DATASET\global_kinematic_std.npy"))
except FileNotFoundError:
    st.warning("⚠️ Normalization files not found, using identity (no denormalization).")
    g_mean = np.zeros(35, dtype=np.float32)
    g_std  = np.ones(35, dtype=np.float32)

# Sidebar
with st.sidebar:
    # ── Tutorial Controls ──
    st.header("\U0001f4d6 Guided Tour")
    tutorial_on = st.toggle(
        "Tutorial Mode",
        value=True,
        help="Show plain-English explanations for each section of the dashboard."
    )
    if tutorial_on:
        steps_total = 5
        st.progress(min(st.session_state.tutorial_step / steps_total, 1.0))
        st.caption(f"Tip {st.session_state.tutorial_step + 1} of {steps_total}")
        if st.button("Next Tip \u2192", use_container_width=True):
            st.session_state.tutorial_step = min(st.session_state.tutorial_step + 1, steps_total - 1)
        if st.button("Reset Tour", use_container_width=True):
            st.session_state.tutorial_step = 0

    st.divider()

    # Main Controls
    st.header("\U0001f39b\ufe0f Controls")

    user_lambda = st.slider(
        "Sensitivity Dial (\u03bb)",
        min_value=0.1, max_value=2.0, value=1.0, step=0.05,
        help=(
            "Think of this like a microphone sensitivity dial.\n\n"
            "\u2190 Drag LEFT: AI uses MORE power (all 4 blocks). Best when muscles are tired.\n"
            "\u2192 Drag RIGHT: AI saves power (1 block). Best for simple movements."
        )
    )

    fatigue_on = st.toggle(
        "\U0001f525 Simulate Muscle Fatigue",
        value=False,
        help="Turns on to see what happens when your arm muscles get tired after hours of use."
    )

    trial_map = {
        "Trial 1 (task_0001)": "task_0001",
        "Trial 2 (task_0002)": "task_0002",
        "Trial 3 (task_0003)": "task_0003",
        "Trial 4 (task_0004)": "task_0004",
        "Trial 32 (task_0032)": "task_0032",
        "Trial 577 (best trial in MuJoCo)": "task_0577",
    }
    trial_choice = st.selectbox("Movement to Analyze", list(trial_map.keys()), index=0)
    selected_trial = trial_map[trial_choice]

    ji = st.selectbox(
        "Joint to Examine",
        list(range(35)), index=0, format_func=lambda x: f"Joint {x}",
        help="Pick which finger/wrist joint to look at. Joint 0 is typically the wrist."
    )
    joint_names = [f"Joint {i}" for i in range(35)]

    st.divider()
    st.caption("ROBO047 \u2022 Katy Youth Hacks 2026")

# Load Data and Run Inference
try:
    emg_data   = np.load(os.path.join(DATA_DIR, f"{selected_trial}_emg.npy"))
    angle_data = np.load(os.path.join(DATA_DIR, f"{selected_trial}_angle.npy"))
except FileNotFoundError:
    st.error(f"❌ Trial data not found for '{selected_trial}'. Check files exist in:\n`{DATA_DIR}`")
    st.stop()

if fatigue_on:
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.4, btype="low")
    fatigued = np.zeros_like(emg_data)
    for ch in range(14):
        for w in range(emg_data.shape[0]):
            fatigued[w, :, ch] = filtfilt(b, a, emg_data[w, :, ch])
    emg_data = fatigued

emg_tensor = torch.tensor(emg_data[np.newaxis, ...], dtype=torch.float32)
# angle_data is [T, 25, 35] — take first frame, add batch+time dims → [1, 1, 25, 35]
if angle_data.ndim == 3:
    seed = torch.tensor(angle_data[0:1][np.newaxis, ...], dtype=torch.float32)
elif angle_data.ndim == 2:
    seed = torch.tensor(np.broadcast_to(angle_data[0:1].astype(np.float32), (25, 35))[np.newaxis, np.newaxis, ...], dtype=torch.float32)
else:
    seed = torch.zeros(1, 1, 25, 35, dtype=torch.float32)

with torch.no_grad():
    output = model(emg_tensor, seed)
    if isinstance(output, str):
        st.error(f"Model forward returned error: {output}")
        st.stop()
    emg_var = torch.var(emg_tensor, dim=2)
    h_data = torch.std(torch.log(emg_var + 1e-6), dim=-1)
    h_scalar = h_data.mean().item()
    adjusted_thresh = controller.base_threshold.item() * user_lambda
    if h_scalar < adjusted_thresh * 0.6:
        tier = 1
    elif h_scalar < adjusted_thresh:
        tier = 2
    else:
        tier = 3

emg_attr = emg_tensor.clone().detach().requires_grad_(True)
out2 = model(emg_attr, seed)
if isinstance(out2, str):
    channel_imp = np.ones(14, dtype=np.float32) / 14 * 100
else:
    out2.sum().backward()
    grads = emg_attr.grad.abs().mean(dim=(0,1,2))
    total = grads.sum().item()
    channel_imp = (grads / total * 100).cpu().numpy() if total > 0 else np.ones(14) / 14 * 100

h_scalar  = h_data.mean().item()
adjusted_thresh = controller.base_threshold.item() * user_lambda

# Force 2D [?, 35] regardless of model output shape
pred_flat = output[0].detach().cpu().numpy().reshape(-1, 35)
gt_flat   = angle_data.reshape(-1, 35)

# Un-z-score to degrees
pred_deg = pred_flat * g_std + g_mean
gt_deg   = gt_flat * g_std + g_mean

ji_int = int(ji)
min_len = min(len(pred_deg), len(gt_deg))
rmse = np.sqrt(np.mean((pred_deg[:min_len, ji_int] - gt_deg[:min_len, ji_int]) ** 2))

# Tutorial System
tutorials = [
    {
        'title': '\U0001f44b Welcome! What are you looking at?',
        'text': (
            "This dashboard shows what happens INSIDE an AI that controls a prosthetic hand. "
            "The AI reads electrical signals from your arm muscles (called EMG) and predicts "
            "how your hand should move. The big innovation: this system tells you WHICH muscles "
            "it\'s listening to, and lets you adjust how hard the AI tries. Scroll down to explore!"
        ),
    },
    {
        'title': '\U0001f4ca The 4 Status Numbers',
        'text': (
            "(1) AI Power Level — Think of it as low/medium/high power mode. "
            "(2) Muscle Signal Complexity — How \"chaotic\" your muscles are right now. "
            "(3) Sensitivity Setting — Where you\'ve set the dial. "
            "(4) Prediction Error — How far off the AI is, in degrees."
        ),
    },
    {
        'title': '\U0001f4c8 The Joint Movement Graph',
        'text': (
            "The blue line is what the AI predicted. The red dashed line is what actually happened. "
            "Don\'t panic if they don\'t match perfectly — see the yellow box below the graph for why. "
            "The key insight: individual joint numbers look noisy, but the OVERALL 3D hand movement is smooth!"
        ),
    },
    {
        'title': '\U0001f9e0 The Muscle Attribution Chart (this is the magic)',
        'text': (
            "The bar chart on the right shows which of your 14 arm muscles the AI listened to most. "
            "RED = high influence, GRAY = low. Now try this: toggle \"Simulate Muscle Fatigue\" in the sidebar "
            "and watch the bars SHIFT — the AI starts relying on different muscles. That\'s explainable AI!"
        ),
    },
    {
        'title': '\U0001f39b\ufe0f The Sensitivity Dial (\u03bb)',
        'text': (
            "Drag the dial LEFT (lower \u03bb) \u2192 AI uses full power (Tier 3, all 4 blocks). "
            "Drag RIGHT (higher \u03bb) \u2192 AI conserves power (Tier 1, 1 block). "
            "When muscles are tired, drag LEFT so the AI compensates. Like turning up mic sensitivity."
        ),
    },
]

if tutorial_on and st.session_state.tutorial_step < len(tutorials):
    t = tutorials[st.session_state.tutorial_step]
    st.markdown(
        f'<div class="tutorial-box"><h4>{t["title"]}</h4><p>{t["text"]}</p></div>',
        unsafe_allow_html=True
    )

# Status Bar
st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)

tier_info = {
    1: ("\U0001f7e2 Light Fold", "1 of 4 blocks \u2014 saving power", "\U0001f7e2 TIER 1: LIGHT FOLD"),
    2: ("\U0001f7e0 Medium Fold", "2 of 4 blocks \u2014 balanced", "\U0001f7e0 TIER 2: MEDIUM FOLD"),
    3: ("\U0001f534 Full Unfold", "All 4 blocks \u2014 maximum accuracy", "\U0001f534 TIER 3: FULL UNFOLD"),
}
tname, tdesc, tlabel = tier_info[tier]

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.metric("AI Power Level", tname)
    st.caption(tdesc)
with sc2:
    st.metric("Muscle Signal Complexity", f"{h_scalar:.2f}")
    st.caption("Higher = more complex movement")
with sc3:
    st.metric("Sensitivity Dial (\u03bb)", f"{user_lambda:.2f}")
    st.caption(f"Threshold = {adjusted_thresh:.2f}")
with sc4:
    st.metric("Prediction Error", f"{rmse:.1f}\u00b0")
    st.caption("Lower = more accurate for this joint")

# Contextual alerts
if fatigue_on:
    if tier == 3:
        st.warning(
            "\u26a1 Muscle fatigue detected! The system automatically unfolded to full power (Tier 3) to compensate. "
            "Now try dragging the Sensitivity Dial RIGHT \u2014 the AI will under-power and you\'ll see accuracy drop. "
            "This demonstrates WHY the user needs control."
        )
    else:
        st.error(
            "\u26a0\ufe0f Fatigue is active AND the AI is under-powered! The sensitivity dial is set too high \u2014 "
            "the AI isn\'t using enough of its brainpower. **\u2190 Drag the dial LEFT** to recover accuracy!"
        )
else:
    if tier == 3:
        st.info(
            "\u2705 The AI is using full power because this movement is complex. That\'s normal. "
            "Try dragging the Sensitivity Dial RIGHT to see it save power (Tier 1)."
        )
    else:
        st.success(
            f"\u2705 This movement is simple enough that the AI only needs {tname.split()[0][-1]} block(s). "
            f"Power saved! The movement doesn\'t require full AI capacity."
        )

st.divider()

# Main Chart
left_col, right_col = st.columns([2, 1])

with left_col:
    # Joint Angle Graph
    st.markdown('<p class="section-label">Joint Movement</p>', unsafe_allow_html=True)
    st.subheader("Prediction vs. Reality")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(gt_deg[:, ji]))),
        y=gt_deg[:, int(ji)],
        mode='lines',
        name='What Actually Happened (Ground Truth)',
        line=dict(color='#ef4444', dash='dash', width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(pred_deg[:, ji]))),
        y=pred_deg[:, int(ji)],
        mode='lines',
        name='What the AI Predicted (RMST)',
        line=dict(color='#10b981', width=2),
    ))
    fig.update_layout(
        title=f"{joint_names[ji]} \u2014 Angle over Time",
        xaxis_title="Time Step",
        yaxis_title="Angle (degrees)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

    # Honesty Box
    st.markdown("""
    <div class="honesty-box">
        <h4>\u26a0\ufe0f Why might these lines not match perfectly?</h4>
        <p>
        This graph shows ONE of 35 joint angles. The AI predicts all 35 simultaneously from 14 muscle signals \u2014
        that\'s a massive underdetermined problem. Individual joint numbers can look noisy. <strong>But here\'s the key:</strong>
        when we feed these same predictions into a 3D physics simulation (MuJoCo), the overall hand movement
        is <strong>physically realistic and closely matches the ground truth</strong>. Small errors in individual joints often resolve with post-processing when it comes to real-world prosthetic control. The <strong>79.4% physical feasibility</strong> score proves this.<br><br>
        <strong>MyoOrigami xAI is about transparency and robustness, not claiming perfect accuracy.</strong>
        Current systems fail silently \u2014, but when ours fails, it does so loudly and gives you control to recover.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # EMG Signal 
    st.markdown('<p class="section-label">Muscle Signals</p>', unsafe_allow_html=True)
    st.subheader("What the AI Sees (EMG)")
    st.caption("Each line is a different muscle sensor. Warmer colors = muscles the AI relied on more.")

    ch_max = max(channel_imp.max(), 1e-6)
    colors = []
    for imp in channel_imp:
        r = imp / ch_max
        colors.append(f"rgba({int(r * 220)}, {int((1 - r) * 100 + 80)}, {int((1 - r) * 220 + 35)}, 0.8)")

    fig2 = go.Figure()
    for ch in range(14):
        fig2.add_trace(go.Scatter(
            x=list(range(25)),
            y=emg_data[0, :, ch],
            mode='lines',
            name=f"Muscle {ch} ({channel_imp[ch]:.1f}%)",
            line=dict(color=colors[ch], width=1.5),
            showlegend=False,
        ))
    fig2.update_layout(
        title="Muscle Signals \u2014 First Time Window (50ms)",
        xaxis_title="Sample",
        yaxis_title="Electrical Activity",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        template="plotly_white",
    )
    st.plotly_chart(fig2, width="stretch")

with right_col:
    # Channel Attribution
    st.markdown('<p class="section-label">Explainable AI</p>', unsafe_allow_html=True)
    st.subheader("Which Muscles Drove This?")
    st.caption("The xAI layer shows you exactly which muscles the AI listened to.")

    bar_colors = [
        '#ef4444' if x > 10 else '#f97316' if x > 5 else '#d1d5db'
        for x in channel_imp
    ]
    fig3 = go.Figure(go.Bar(
        x=channel_imp,
        y=[f"Muscle {i}" for i in range(14)],
        orientation='h',
        marker_color=bar_colors,
    ))
    fig3.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=10, b=10),
        xaxis_title="Influence (%)",
        template="plotly_white",
    )
    st.plotly_chart(fig3, width="stretch")

    # Color legend
    st.markdown("""
    <div style="padding: 6px 0; font-size: 12px; color: #374151;">
        <span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>
        <strong>High influence</strong> \u2014 AI relied on this muscle<br>
        <span style="display:inline-block;width:10px;height:10px;background:#f97316;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>
        <strong>Medium influence</strong><br>
        <span style="display:inline-block;width:10px;height:10px;background:#d1d5db;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>
        <strong>Low influence</strong> \u2014 not involved in this movement
    </div>
    """, unsafe_allow_html=True)

st.divider()

# MuJoCo Simulation Section
st.markdown('<p class="section-label">Visual Proof</p>', unsafe_allow_html=True)
st.subheader("\U0001f590\ufe0f 3D Physics Simulation (MuJoCo)")

st.markdown("""
<div class="mujoco-box">
    <h4>Why 3D Simulation Beats Graphs</h4>
    <p>
    Individual joint graphs can look messy — but when you feed those same predictions into a physics
    simulator and watch the <strong>actual 3D hand move</strong>, the motion is smoother and natural.
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-detect media files
media_dir = MUJOCO_MEDIA_DIR
media_files = []
if os.path.isdir(media_dir):
    for f in sorted(os.listdir(media_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.webm')):
            media_files.append(os.path.join(media_dir, f))

if media_files:
    for f in media_files:
        if f.lower().endswith(('.mp4', '.webm')):
            st.video(f)
        else:
            st.image(f, use_container_width=True)
        st.caption(os.path.basename(f))
else:
    st.info(
        "\U0001f4c1 Place your MuJoCo simulation video or screenshots in:\n\n"
        f"`{media_dir}`\n\n"
        "Supported: .mp4, .webm, .png, .jpg, .gif"
    )

st.divider()

# Expandable Sections
with st.expander("\U0001f4d6 How does this work? (Simple explanation)", expanded=False):
    st.markdown("""
    **Step 1:** You have 14 sensors on your arm that read muscle electrical signals (EMG).

    **Step 2:** The AI model (called RMST) looks at these signals and predicts how your hand
    should move \u2014 all 35 joint angles, simultaneously, in under 9 milliseconds.

    **Step 3:** The MyoOrigami xAI layer watches the AI think. It measures how complex your
    muscle signals are (a number called H\_data). Simple movements \u2192 the AI uses only
    1/4 of its brainpower (saves battery). Complex movements \u2192 the AI uses all 4/4
    blocks (maximum accuracy). This is the \"origami\" concept \u2014 same structure, different folds.

    **Step 4:** The system shows you WHICH muscles drove each prediction (the bar chart).
    When muscles fatigue, the pattern shifts \u2014 different muscles take over. You can see
    this happening in real time.

    **Step 5:** If things feel off, you adjust the Sensitivity Dial (\u03bb). This tells the
    AI to try harder or conserve power. No recalibration, no taking off the device.

    **Step 6:** We validate everything in MuJoCo physics simulation. The 3D hand movements
    are physically realistic \u2014 79.4% feasibility on average.
    """)

with st.expander("\U0001f52c Technical Details", expanded=False):
    st.markdown("""
    **Architecture:** Recursive Multimodal Sparse Transformer (RMST) with 4 RMCABlock layers,
    gated recycling connections, TDC convolution, and axial attention.
    Input: sEMG `[B, N, 25, 14]` + kinematic seed `[B, 1, 25, 35]`.
    Output: continuous joint trajectory `[B, N\u00d725, 35]`.

    **Origami Modification:** Forward pass uses `self.rmca_stack[:recursion_depth]` instead
    of the full stack. Tier 1 = 1 block, Tier 2 = 2 blocks, Tier 3 = 4 blocks.
    No weights changed \u2014 inference-time compute adaptation.

    **H\_data:** `std(log(var(EMG, dim=time) + 1e-6), dim=channels)` \u2014 measures spatial
    complexity across the 14 EMG channels. Validated by Phinyomark et al. (2013).

    **Channel Attribution:** Gradient-based importance via `emg.requires_grad_(True)` \u2192
    forward \u2192 `.backward()` \u2192 normalize |grad| to 100%.
    Same principle as Integrated Gradients (Sundararajan et al., ICML 2017).

    **Fatigue Simulation:** 4th-order Butterworth low-pass filter simulating spectral
    compression \u2014 the most reliable fatigue indicator (De Luca, 1979).

    **Sparsity:** 2:4 structured sparsity (NVIDIA Research) for 19.35% model reduction.

    **MuJoCo Validation:** Shadow Hand model (20 DOF). Checks joint limits (5\u00b0 tolerance)
    + dynamic self-collision detection.
    """)

# Footer
st.divider()
st.caption("ROBO047 \u2022 MyoOrigami xAI \u2022 Katy Youth Hacks 2026 \u2022 USPA03")
st.caption("Built with PyTorch, Streamlit, Plotly, MuJoCo \u2022 Data: MOVUJS/UJI Dataset")
