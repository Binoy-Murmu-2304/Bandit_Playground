"""
Findings — key insights with evidence, ready for interviews and viva prep.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from bandit.environment import KArmedBandit, NonStationaryBandit
from bandit.agents import (
    GreedyAgent,
    EpsilonGreedyAgent,
    OptimisticAgent,
    UCBAgent,
)

st.set_page_config(page_title="Findings", page_icon="📝", layout="wide")

st.title("📝 Key Findings")
st.markdown(
    "The insights that matter — each backed by a reproducible experiment. "
    "Use these in interviews, LinkedIn posts, and viva preparation."
)


# ── Cached experiment helpers ───────────────────────────────────

@st.cache_data(show_spinner=False)
def run_epsilon_sweep(k, noise, n_pulls, n_runs, epsilons):
    """Run ε-greedy with different ε values. Returns dict ε -> (rew, opt)."""
    results = {}
    for eps in epsilons:
        rew = np.zeros(n_pulls)
        opt = np.zeros(n_pulls)
        for run in range(n_runs):
            bandit = KArmedBandit(k=k, noise=noise, seed=run * 17)
            agent = EpsilonGreedyAgent(k=k, epsilon=eps, seed=run * 17 + 1)
            optimal = bandit.optimal_action()
            for t in range(n_pulls):
                a = agent.select_action()
                r = bandit.pull(a)
                agent.update(a, r)
                rew[t] += r
                opt[t] += 1 if a == optimal else 0
        rew /= n_runs
        opt /= n_runs
        results[eps] = (rew, opt)
    return results


@st.cache_data(show_spinner=False)
def run_greedy_vs_ucb(k, noise, n_pulls, n_runs):
    """Compare greedy vs UCB on the same bandits."""
    g_rew = np.zeros(n_pulls)
    u_rew = np.zeros(n_pulls)
    g_opt = np.zeros(n_pulls)
    u_opt = np.zeros(n_pulls)

    for run in range(n_runs):
        for agent_cls, r_acc, o_acc in [
            (GreedyAgent, g_rew, g_opt),
            (UCBAgent, u_rew, u_opt),
        ]:
            bandit = KArmedBandit(k=k, noise=noise, seed=run * 23)
            agent = agent_cls(k=k, seed=run * 23 + 1)
            optimal = bandit.optimal_action()
            for t in range(n_pulls):
                a = agent.select_action()
                r = bandit.pull(a)
                agent.update(a, r)
                r_acc[t] += r
                o_acc[t] += 1 if a == optimal else 0

    return (g_rew / n_runs, u_rew / n_runs,
            g_opt / n_runs, u_opt / n_runs)


@st.cache_data(show_spinner=False)
def run_nonstationary_comparison(k, noise, n_pulls, change_step,
                                 drift_size, epsilon, alpha_val, n_runs):
    sa_r = np.zeros(n_pulls)
    ca_r = np.zeros(n_pulls)
    for run in range(n_runs):
        for mode, acc in [("sa", sa_r), ("ca", ca_r)]:
            bandit = NonStationaryBandit(
                k=k, noise=noise, change_step=change_step,
                drift_size=drift_size, seed=run * 29,
            )
            a = alpha_val if mode == "ca" else None
            agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, alpha=a,
                                        seed=run * 29 + 1)
            for t in range(n_pulls):
                act = agent.select_action()
                r = bandit.pull(act)
                agent.update(act, r)
                acc[t] += r
    return sa_r / n_runs, ca_r / n_runs


# ── Finding 1: The Greedy Trap ──────────────────────────────────

st.header("1. The Greedy Trap")
st.markdown(
    """
    **Claim:** Pure exploitation (greedy) gets stuck on a suboptimal arm
    and never recovers — even with thousands of pulls.

    **Evidence:** Greedy vs UCB on the same bandits.
    """
)

if st.button("Run Experiment 1", key="exp1"):
    with st.spinner("Running…"):
        g_r, u_r, g_o, u_o = run_greedy_vs_ucb(
            k=10, noise=1.0, n_pulls=1000, n_runs=500,
        )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=g_r, mode="lines", name="Greedy",
                                 line=dict(color="#ef4444", width=2)))
        fig.add_trace(go.Scatter(y=u_r, mode="lines", name="UCB",
                                 line=dict(color="#8b5cf6", width=2)))
        fig.update_layout(template="plotly_dark", height=350,
                          title="Average Reward", xaxis_title="Pull #")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=g_o * 100, mode="lines", name="Greedy",
                                  line=dict(color="#ef4444", width=2)))
        fig2.add_trace(go.Scatter(y=u_o * 100, mode="lines", name="UCB",
                                  line=dict(color="#8b5cf6", width=2)))
        fig2.update_layout(template="plotly_dark", height=350,
                           title="% Optimal Action", xaxis_title="Pull #")
        st.plotly_chart(fig2, use_container_width=True)

    st.success(
        f"**Result:** Greedy ends at {g_o[-100:].mean()*100:.1f}% optimal; "
        f"UCB at {u_o[-100:].mean()*100:.1f}%. "
        f"The gap never closes — greedy is permanently stuck."
    )

st.divider()

# ── Finding 2: Exploration Budget Matters ───────────────────────

st.header("2. The Right ε Depends on Your Budget")
st.markdown(
    """
    **Claim:** More exploration is not always better. The optimal ε
    depends on how many pulls you have.
    """
)

if st.button("Run Experiment 2", key="exp2"):
    with st.spinner("Running ε sweep…"):
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        results = run_epsilon_sweep(10, 1.0, 1000, 500, epsilons)

    fig = go.Figure()
    colors = ["#6b7280", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ef4444"]
    for (eps, (_, opt)), color in zip(results.items(), colors):
        fig.add_trace(go.Scatter(
            y=opt * 100, mode="lines", name=f"ε={eps}",
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        template="plotly_dark", height=420,
        title="% Optimal Action — ε Sweep",
        xaxis_title="Pull #", yaxis_title="% Optimal",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Read the curves:** With 1000 pulls, ε=0.1 wins on % optimal. "
        "But ε=0.01 is still climbing — give it 5000 pulls and it likely "
        "surpasses. The 'best' ε is relative to your budget."
    )

st.divider()

# ── Finding 3: Non-Stationarity ─────────────────────────────────

st.header("3. Constant α Wins When the World Changes")
st.markdown(
    """
    **Claim:** Sample average (1/N) fails to adapt after a distribution
    shift. Constant step-size α recovers quickly.

    **Connection:** This directly justifies `LEARNING_RATE_ALPHA = 0.1`
    in the Self-Healing MLOps Engine.
    """
)

if st.button("Run Experiment 3", key="exp3"):
    with st.spinner("Running non-stationary experiment…"):
        sa_r, ca_r = run_nonstationary_comparison(
            k=10, noise=1.0, n_pulls=2000,
            change_step=1000, drift_size=2.0,
            epsilon=0.1, alpha_val=0.1, n_runs=200,
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=sa_r, mode="lines", name="Sample Average (1/N)",
        line=dict(color="#ef4444", width=2),
    ))
    fig.add_trace(go.Scatter(
        y=ca_r, mode="lines", name="Constant α=0.1",
        line=dict(color="#10b981", width=2),
    ))
    fig.add_vline(x=1000, line_dash="dash", line_color="#f59e0b",
                  annotation_text="World Shift")
    fig.update_layout(
        template="plotly_dark", height=420,
        title="Reward: Sample Average vs Constant α",
        xaxis_title="Pull #", yaxis_title="Average Reward",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "After the shift at step 1000, constant α dips less and recovers "
        "faster. Sample average stays anchored to stale history."
    )

st.divider()

# ── Summary ─────────────────────────────────────────────────────

st.header("🏆 Summary: The Three Takeaways")
st.markdown(
    """
    | # | Finding | Practical Implication |
    |---|---------|----------------------|
    | 1 | Pure exploitation gets stuck at ~35% optimal | **Always explore** — at least during training |
    | 2 | The best ε depends on your time budget | **Decay ε over time** — explore early, exploit later |
    | 3 | Constant α adapts to drift; 1/N doesn't | **Use constant α** in non-stationary environments (production!) |

    ---

    ### 🌉 Connection to the Self-Healing MLOps Engine

    These three findings map directly to the FYP's RL design:

    - **Finding 1** → `EPSILON_LIVE = 0.0` is safe *because* exploration
      already happened offline (`EPSILON_START = 1.0` during pretraining).

    - **Finding 2** → `EPSILON_DECAY = 0.995` and `EPSILON_FLOOR = 0.1`
      implement the "explore early, exploit later" schedule.

    - **Finding 3** → `LEARNING_RATE_ALPHA = 0.1` ensures the Q-table
      keeps adapting as the production environment drifts.

    **The playground demonstrates the theory. The MLOps engine applies it.**
    """
)