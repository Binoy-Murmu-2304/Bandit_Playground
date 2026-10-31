"""
Non-Stationary Bandit — the world shifts mid-game.
Compares sample-average (1/N) vs constant-α adaptation.
This is the experiment that justifies LEARNING_RATE_ALPHA in the FYP.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from bandit.environment import NonStationaryBandit
from bandit.agents import EpsilonGreedyAgent

st.set_page_config(page_title="Non-Stationary", page_icon="🌍", layout="wide")

st.title("🌍 Non-Stationary Bandit")
st.markdown(
    """
    The true arm values **change mid-game** — simulating data drift or
    concept drift in a production system. Which learning rule adapts faster?

    - **Sample average (1/N):** every past reward counts equally. Slow to adapt.
    - **Constant α:** recent rewards matter more. Quick recovery after a shift.
    """
)

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Arms (k)", 2, 20, 10)
    noise = st.slider("Reward noise (σ)", 0.1, 3.0, 1.0, 0.1)
    n_pulls = st.slider("Total pulls", 200, 5000, 2000, 100)
    change_step = st.slider("World shifts at step", 50, 4000, 1000, 50)
    drift_size = st.slider("Drift magnitude", 0.5, 5.0, 2.0, 0.1)
    epsilon = st.slider("ε (exploration)", 0.01, 0.5, 0.1, 0.01)
    alpha_val = st.slider("α (constant step)", 0.01, 1.0, 0.1, 0.01)
    n_runs = st.slider("Averaging runs", 10, 500, 200, 10)

    st.divider()
    run_btn = st.button("▶️ Run Experiment", type="primary",
                        use_container_width=True)


# ── Cached runner ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_nonstationary(k, noise, n_pulls, change_step, drift_size,
                      epsilon, alpha_val, n_runs):
    sa_rewards = np.zeros(n_pulls)  # sample average
    ca_rewards = np.zeros(n_pulls)  # constant alpha
    sa_optimal = np.zeros(n_pulls)
    ca_optimal = np.zeros(n_pulls)

    for run in range(n_runs):
        for mode, rew_acc, opt_acc in [
            ("sample_avg", sa_rewards, sa_optimal),
            ("const_alpha", ca_rewards, ca_optimal),
        ]:
            bandit = NonStationaryBandit(
                k=k, noise=noise,
                change_step=change_step,
                drift_size=drift_size,
                seed=run * 13,
            )
            alpha = alpha_val if mode == "const_alpha" else None
            agent = EpsilonGreedyAgent(
                k=k, epsilon=epsilon, alpha=alpha, seed=run * 13 + 1,
            )

            for t in range(n_pulls):
                action = agent.select_action()
                reward = bandit.pull(action)
                agent.update(action, reward)
                rew_acc[t] += reward
                opt_acc[t] += 1 if action == bandit.optimal_action() else 0

    sa_rewards /= n_runs
    ca_rewards /= n_runs
    sa_optimal /= n_runs
    ca_optimal /= n_runs

    # Recovery time: pulls after change_step until avg reward
    # is within 10% of the post-shift optimal value
    return sa_rewards, ca_rewards, sa_optimal, ca_optimal


# ── Run & Display ───────────────────────────────────────────────

if run_btn:
    with st.spinner("Running non-stationary experiment…"):
        sa_r, ca_r, sa_o, ca_o = run_nonstationary(
            k, noise, n_pulls, change_step, drift_size,
            epsilon, alpha_val, n_runs,
        )

    # ── Chart 1: Reward Over Time ───────────────────────────────
    st.subheader("📈 Average Reward — Sample Average vs Constant α")

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        y=sa_r, mode="lines", name="Sample Average (1/N)",
        line=dict(color="#ef4444", width=2),
    ))
    fig_r.add_trace(go.Scatter(
        y=ca_r, mode="lines", name=f"Constant α={alpha_val}",
        line=dict(color="#10b981", width=2),
    ))

    # Vertical line at the shift
    fig_r.add_vline(
        x=change_step, line_dash="dash", line_color="#f59e0b",
        annotation_text="🌍 World Shift",
        annotation_position="top",
        annotation_font_size=14,
    )

    fig_r.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Pull #", yaxis_title="Average Reward",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Chart 2: % Optimal ──────────────────────────────────────
    st.subheader("🎯 % Optimal Action")

    fig_o = go.Figure()
    fig_o.add_trace(go.Scatter(
        y=sa_o * 100, mode="lines", name="Sample Average (1/N)",
        line=dict(color="#ef4444", width=2),
    ))
    fig_o.add_trace(go.Scatter(
        y=ca_o * 100, mode="lines", name=f"Constant α={alpha_val}",
        line=dict(color="#10b981", width=2),
    ))
    fig_o.add_vline(
        x=change_step, line_dash="dash", line_color="#f59e0b",
    )
    fig_o.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Pull #", yaxis_title="% Optimal",
    )
    st.plotly_chart(fig_o, use_container_width=True)

    # ── Analysis ────────────────────────────────────────────────
    st.subheader("📊 Post-Shift Recovery Analysis")

    # Measure performance in windows before and after the shift
    pre_start = max(0, change_step - 200)
    pre_end = change_step
    post_start = change_step
    post_end = min(n_pulls, change_step + 200)
    late_start = min(n_pulls, change_step + 200)
    late_end = n_pulls

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Before shift** (last 200 pulls)")
        st.metric("Sample Avg", f"{sa_r[pre_start:pre_end].mean():.3f}")
        st.metric("Constant α", f"{ca_r[pre_start:pre_end].mean():.3f}")
    with a2:
        st.markdown("**Right after shift** (first 200 pulls)")
        sa_drop = sa_r[post_start:post_end].mean() - sa_r[pre_start:pre_end].mean()
        ca_drop = ca_r[post_start:post_end].mean() - ca_r[pre_start:pre_end].mean()
        st.metric("Sample Avg", f"{sa_r[post_start:post_end].mean():.3f}",
                  delta=f"{sa_drop:+.3f}")
        st.metric("Constant α", f"{ca_r[post_start:post_end].mean():.3f}",
                  delta=f"{ca_drop:+.3f}")
    with a3:
        st.markdown("**Late** (last 200 pulls)")
        st.metric("Sample Avg", f"{sa_r[late_start:late_end].mean():.3f}")
        st.metric("Constant α", f"{ca_r[late_start:late_end].mean():.3f}")

    st.divider()

    st.markdown(
        """
        ### 🔑 What this tells us

        **Sample average (1/N)** treats every past reward as equally relevant.
        After 1000 pulls of history, one new reward moves the estimate by
        only 1/1001 — barely at all. When the world shifts, the agent
        is **anchored to stale data** and recovers slowly.

        **Constant α** weights recent experience by a fixed factor
        (α = 0.1 → each new reward moves the estimate 10% of the gap).
        Old data fades exponentially. After the shift, it **re-adapts quickly**.

        > **This is exactly why the Self-Healing MLOps Engine uses
        > `LEARNING_RATE_ALPHA = 0.1`** — production environments are
        > non-stationary (retraining changes everything), so the Q-table
        > must keep adapting rather than settle on stale averages.
        """
    )
else:
    st.info("👈 Configure parameters and click **Run Experiment** to begin.")