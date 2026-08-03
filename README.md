**MyoOrigami-xAI
**
Explainable AI for continuous regression myoelectric prosthetic hand control using a Recursive Multimodal Sparse Transformer with dynamic model unfolding.
The Problem

AI-powered prosthetic hands are black boxes. They read your muscle signals and move, but if something goes wrong, there's no explanation and no way to fix it without a full recalibration. I wanted to change that.
What This Does

**Origami xAI adds three things on top of our RMST model:
**
    A sensitivity dial (λ) — A single slider the user can adjust. When muscles get tired, drag left and the AI uses more processing power. For simple movements, drag right to save battery. No recalibration needed.

    Channel attribution — After each prediction, a bar chart shows which of the 14 forearm muscles the AI relied on most. Toggle on the fatigue simulation and watch the bars shift in real-time as the AI starts relying on different muscle groups.

    Dynamic model unfolding — The model has 4 processing blocks but doesn't always use all of them. Simple movements get 1 block (fast, efficient). Complex movements unfold all 4. Same model, different folds — like origami. No weights change.

**How It Works
**
14 EMG sensors → RMST (4 RMCABlock layers with gated recycling, TDC convolution, axial attention) → 35 joint angle predictions → Robotic hand

The xAI layer computes signal complexity using H_data = std(log(var(EMG) + ε)) across channels, selects processing tier based on the λ threshold, and generates gradient-based channel attribution by backpropagating through the input.
MuJoCo Physics Validation

I feed predictions into MuJoCo physics simulation (Shadow Hand, 20 DOF) and check joint limits and self-collisions. The model achieves 79.4% physical feasibility. Competing approaches (LSTM, Ridge regression) appear to score well because they predict nearly identical static poses regardless of input — they aren't actually decoding different movements.

**Tech Stack
**
PyTorch, Streamlit, Plotly, MuJoCo, NumPy, SciPy
Dataset: MOVUJS/UJI sEMG Dataset
Sparsity: 2:4 structured pruning (19.35% model reduction)
Run the Dashboard

pip install streamlit torch numpy plotly scipy
streamlit run dashboard.py
Team

**ROBO047 — Katy Youth Hacks 2026
License: MIT**
