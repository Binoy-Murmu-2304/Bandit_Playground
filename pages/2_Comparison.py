"""
Comparison — run all five algorithms on the same bandit and overlay curves.
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

st.set_page_config(page_title="Comparison", page_icon="📊", layout="wide")

st.title("📊 Algorithm Comparison")
st.markdown(
    "All five algorithms on the **same bandit** (same true values, same noise). "
    "Curves are averaged over multiple independent runs for smooth results."
)

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Arms (k)", 2, 20, 10)
    noise = st.slider("Reward noise (σ)", 0.1, 5.0, 1.0, 0.1)
    n_pulls = st.slider("Pulls per run", 100, 5000, 1000, 100)
    n_runs = st.slider("Independent runs", 50, 2000, 500, 50)

    st.divider()

    show_greedy = st.checkbox("Greedy", value=True)
    show_eps01 = st.checkbox("ε-greedy (ε=0.01)", value=True)
    show_eps10 = st.checkbox("ε-greedy (ε=0.1)", value=True)
    show_opt = st.checkbox("Optimistic", value=True)
    show_ucb = st.checkbox("UCB (c=2.0)", value=True)
    show_thompson = st.checkbox("Thompson Sampling", value=True)

    st.divider()
    run_btn = st.button("▶️ Run Comparison", type="primary",
                        use_container_width=True)


# ── Cached experiment runner ────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_comparison_cached(k, noise, n_pulls, n_runs,
                          show_greedy, show_eps01, show_eps10,
                          show_opt, show_ucb, show_thompson):
    """
    Run every selected algorithm on the same set of bandits.
    Returns dict: name -> (avg_rewards_array, pct_optimal_array)
    """
    algorithms = []

    if show_greedy:
        algorithms.append((
            "Greedy (ε=0)",
            lambda seed: GreedyAgent(k=k, seed=seed),
        ))
    if show_eps01:
        algorithms.append((
            "ε-greedy (ε=0.01)",
            lambda seed: EpsilonGreedyAgent(k=k, epsilon=0.01, seed=seed),
        ))
    if show_eps10:
        algorithms.append((
            "ε-greedy (ε=0.1)",
            lambda seed: EpsilonGreedyAgent(k=k, epsilon=0.1, seed=seed),
        ))
    if show_opt:
        algorithms.append((
            "Optimistic (init=5.0)",
            lambda seed: OptimisticAgent(k=k, initial_value=5.0, seed=seed),
        ))
    if show_ucb:
        algorithms.append((
            "UCB (c=2.0)",
            lambda seed: UCBAgent(k=k, c=2.0, seed=seed),
        ))
    if show_thompson:
        algorithms.append((
            "Thompson Sampling",
            lambda seed: ThompsonSamplingAgent(k=k, noise_std=noise, seed=seed),
        ))

    results = {}
    for name, factory in algorithms:
        avg_rewards = np.zeros(n_pulls)
        pct_optimal = np.zeros(n_pulls)

        for run in range(n_runs):
            bandit = KArmedBandit(k=k, noise=noise, seed=run * 1000)
            agent = factory(seed=run * 1000 + 1)       # keyword arg: seed
            optimal = bandit.optimal_action()

            for t in range(n_pulls):
                action = agent.select_action()
                reward = bandit.pull(action)
                agent.update(action, reward)
                avg_rewards[t] += reward
                pct_optimal[t] += 1 if action == optimal else 0

        avg_rewards /= n_runs
        pct_optimal /= n_runs
        results[name] = (avg_rewards, pct_optimal)

    return results


# ── Colour map ──────────────────────────────────────────────────

COLORS = {
    "Greedy (ε=0)": "#ef4444",
    "ε-greedy (ε=0.01)": "#f59e0b",
    "ε-greedy (ε=0.1)": "#10b981",
    "Optimistic (init=5.0)": "#3b82f6",
    "UCB (c=2.0)": "#8b5cf6",
    "Thompson Sampling": "#ec4899",
}

# ── Run ─────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Running {n_runs} independent experiments per algorithm…"):
        results = run_comparison_cached(
            k, noise, n_pulls, n_runs,
            show_greedy, show_eps01, show_eps10,
            show_opt, show_ucb, show_thompson,
        )

    if not results:
        st.warning("Select at least one algorithm.")
        st.stop()

    # ── Chart 1: Average Reward ─────────────────────────────────
    st.subheader("📈 Average Reward Over Time")
    fig_reward = go.Figure()
    for name, (rew, _) in results.items():
        fig_reward.add_trace(go.Scatter(
            y=rew.tolist(), mode="lines", name=name,
            line=dict(color=COLORS.get(name, "#ffffff"), width=2),
        ))
    fig_reward.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Pull #", yaxis_title="Average Reward",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_reward, use_container_width=True)

    # ── Chart 2: % Optimal Action ───────────────────────────────
    st.subheader("🎯 % of Time Picking the Best Arm")
    fig_opt = go.Figure()
    for name, (_, opt) in results.items():
        fig_opt.add_trace(go.Scatter(
            y=(opt * 100).tolist(), mode="lines", name=name,
            line=dict(color=COLORS.get(name, "#ffffff"), width=2),
        ))
    fig_opt.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Pull #", yaxis_title="% Optimal",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_opt, use_container_width=True)

    # ── Summary Table ───────────────────────────────────────────
    st.subheader("📋 Final Performance (last 100 pulls)")
    table_rows = []
    for name, (rew, opt) in results.items():
        table_rows.append({
            "Algorithm": name,
            "Avg Reward": f"{rew[-100:].mean():.3f}",
            "% Optimal": f"{opt[-100:].mean() * 100:.1f}%",
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Tip:** Increase 'Independent runs' for smoother curves. "
        "50 is fast; 2000 is publication-quality."
    )
else:
    st.info("👈 Configure parameters and click **Run Comparison** to begin.")