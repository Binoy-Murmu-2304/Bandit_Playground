"""
Live Simulation — pull arms step by step and watch the agent learn.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from bandit.environment import KArmedBandit
from bandit.agents import (
    GreedyAgent,
    EpsilonGreedyAgent,
    OptimisticAgent,
    UCBAgent,
    ThompsonSamplingAgent,
)

st.set_page_config(page_title="Live Simulation", page_icon="🔬", layout="wide")

st.title("🔬 Live Simulation")
st.markdown(
    "Pull arms one at a time and watch Q-values, pull counts, and rewards update "
    "in real time. Or hit **Run All** to see the full learning curve."
)

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    algorithm = st.selectbox(
        "Algorithm",
        ["Greedy", "ε-greedy", "Optimistic", "UCB", "Thompson Sampling"],
        index=1,
    )

    # Algorithm-specific parameters
    epsilon = 0.1
    c = 2.0
    initial_value = 5.0

    if algorithm == "ε-greedy":
        epsilon = st.slider("ε (exploration rate)", 0.0, 1.0, 0.1, 0.01)
    elif algorithm == "UCB":
        c = st.slider("c (exploration constant)", 0.1, 5.0, 2.0, 0.1)
    elif algorithm == "Optimistic":
        initial_value = st.slider("Initial Q value", 0.0, 10.0, 5.0, 0.5)

    st.divider()

    k = st.slider("Arms (k)", 2, 20, 10)
    noise = st.slider("Reward noise (σ)", 0.1, 5.0, 1.0, 0.1)
    n_pulls = st.slider("Max pulls", 100, 5000, 1000, 100)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    use_const_alpha = st.checkbox("Use constant α (instead of 1/N)")
    alpha = (
        st.slider("α (step size)", 0.01, 1.0, 0.1, 0.01)
        if use_const_alpha
        else None
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        btn_start = st.button("🆕 New Sim", type="primary", use_container_width=True)
    with col_b:
        btn_reset = st.button("🔄 Reset", use_container_width=True)
    btn_pull = st.button("🎲 Pull Once", use_container_width=True)
    btn_run_all = st.button("▶️ Run Remaining", use_container_width=True)


# ── Helper: create agent ────────────────────────────────────────

def create_agent():
    common = dict(k=k, alpha=alpha, seed=seed + 1)
    if algorithm == "Greedy":
        return GreedyAgent(**common)
    if algorithm == "ε-greedy":
        return EpsilonGreedyAgent(epsilon=epsilon, **common)
    if algorithm == "Optimistic":
        return OptimisticAgent(initial_value=initial_value, **common)
    if algorithm == "UCB":
        return UCBAgent(c=c, **common)
    return ThompsonSamplingAgent(noise_std=noise, **common)


# ── Session state management ────────────────────────────────────

DEFAULTS = {
    "sim_bandit": None,
    "sim_agent": None,
    "sim_rewards": [],
    "sim_actions": [],
    "sim_optimal_picks": [],
    "sim_running": False,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if btn_reset:
    for key, default in DEFAULTS.items():
        st.session_state[key] = default
    st.rerun()

if btn_start:
    st.session_state.sim_bandit = KArmedBandit(k=k, noise=noise, seed=seed)
    st.session_state.sim_agent = create_agent()
    st.session_state.sim_rewards = []
    st.session_state.sim_actions = []
    st.session_state.sim_optimal_picks = []
    st.session_state.sim_running = True
    st.rerun()

# ── Pull logic ──────────────────────────────────────────────────

def do_one_pull():
    bandit = st.session_state.sim_bandit
    agent = st.session_state.sim_agent
    if bandit is None or agent is None:
        return
    if len(st.session_state.sim_rewards) >= n_pulls:
        return

    action = agent.select_action()
    reward = bandit.pull(action)
    agent.update(action, reward)

    st.session_state.sim_actions.append(action)
    st.session_state.sim_rewards.append(reward)
    st.session_state.sim_optimal_picks.append(
        1 if action == bandit.optimal_action() else 0
    )

if btn_pull:
    if not st.session_state.sim_running:
        st.warning("Start a new simulation first.")
    else:
        do_one_pull()

if btn_run_all:
    if not st.session_state.sim_running:
        st.warning("Start a new simulation first.")
    else:
        remaining = n_pulls - len(st.session_state.sim_rewards)
        progress = st.progress(0.0, text="Running...")
        for i in range(remaining):
            do_one_pull()
            if (i + 1) % 50 == 0:
                progress.progress((i + 1) / remaining)
        progress.empty()
        st.rerun()

# ── Display ─────────────────────────────────────────────────────

bandit = st.session_state.sim_bandit
agent = st.session_state.sim_agent
rewards = st.session_state.sim_rewards
actions = st.session_state.sim_actions
optimal_picks = st.session_state.sim_optimal_picks

if not st.session_state.sim_running:
    st.info("👈 Configure and click **New Sim** to begin.")
    st.stop()

n_done = len(rewards)

# ── Metrics row ─────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Pulls", f"{n_done} / {n_pulls}")
with m2:
    avg_r = np.mean(rewards) if rewards else 0.0
    st.metric("Avg Reward", f"{avg_r:.3f}")
with m3:
    pct = np.mean(optimal_picks) * 100 if optimal_picks else 0.0
    st.metric("% Optimal", f"{pct:.1f}%")
with m4:
    regret = bandit.optimal_value() * n_done - sum(rewards) if rewards else 0
    st.metric("Regret", f"{regret:.1f}")
with m5:
    found = (
        "✅ Yes"
        if int(np.argmax(agent.Q)) == bandit.optimal_action() and n_done > k
        else "❌ Not yet"
    )
    st.metric("Best Arm?", found)

# ── Charts ──────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Reward over time
    fig_r = go.Figure()
    if rewards:
        fig_r.add_trace(go.Scatter(
            y=rewards, mode="lines", name="Reward",
            line=dict(color="#6366f1", width=1), opacity=0.4,
        ))
        if n_done >= 10:
            w = max(3, n_done // 30)
            ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
            fig_r.add_trace(go.Scatter(
                y=ma, mode="lines", name="Moving Avg",
                line=dict(color="#f59e0b", width=3),
            ))
    fig_r.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Pull #", yaxis_title="Reward",
    )
    st.plotly_chart(fig_r, use_container_width=True)

with chart_col2:
    # Q estimates vs true values
    fig_q = go.Figure()
    x_labels = [f"Arm {i}" for i in range(k)]
    fig_q.add_trace(go.Bar(
        x=x_labels, y=agent.Q.tolist(),
        name="Q(a) estimate", marker_color="#6366f1",
    ))
    fig_q.add_trace(go.Bar(
        x=x_labels, y=bandit.true_values.tolist(),
        name="q*(a) true", marker_color="#f59e0b",
    ))
    fig_q.update_layout(
        barmode="group", template="plotly_dark", height=350,
        yaxis_title="Value",
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ── Pull distribution ───────────────────────────────────────────

fig_pulls = go.Figure()
fig_pulls.add_trace(go.Bar(
    x=[f"Arm {i}" for i in range(k)],
    y=agent.N.tolist(),
    marker_color=["#10b981" if i == bandit.optimal_action() else "#475569"
                 for i in range(k)],
))
fig_pulls.update_layout(
    template="plotly_dark", height=280,
    yaxis_title="Pulls per Arm",
    title="Pull Distribution (green = optimal arm)",
)
st.plotly_chart(fig_pulls, use_container_width=True)

# ── Decision log ────────────────────────────────────────────────

if n_done > 0:
    with st.expander(f"📋 Decision Log (last 20 pulls)", expanded=False):
        log_data = []
        for i in range(max(0, n_done - 20), n_done):
            a = actions[i]
            r = rewards[i]
            opt = bandit.optimal_action()
            log_data.append({
                "Step": i + 1,
                "Arm Chosen": f"Arm {a}",
                "Q(a) (now)": f"{agent.Q[a]:.3f}",
                "Reward": f"{r:+.3f}",
                "Optimal?": "✅" if a == opt else "❌",
            })
        st.dataframe(log_data, use_container_width=True, hide_index=True)