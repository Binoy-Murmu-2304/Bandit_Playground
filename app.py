"""
Bandit Playground — Home Page
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Bandit Playground",
    page_icon="🎰",
    layout="wide",
)

# ── Hero ────────────────────────────────────────────────────────

st.title("🎰 Bandit Playground")
st.markdown(
    """
    **An interactive exploration of the exploration-exploitation dilemma**
    — the core challenge in multi-armed bandits and reinforcement learning.

    Navigate through the pages in the sidebar to run experiments,
    compare algorithms, and see the key findings.
    """
)

st.divider()

# ── Canonical results (static numbers, no computation) ──────────

st.subheader("📊 Canonical Results")
st.caption("10-armed testbed · 1000 pulls · 2000 independent runs")

cols = st.columns(5)
metrics = [
    ("Greedy",        "35.8%", "Gets stuck",        "#ef4444"),
    ("ε-greedy 0.01", "60.1%", "Too slow",          "#f59e0b"),
    ("ε-greedy 0.1",  "80.4%", "Solid balance",     "#10b981"),
    ("Optimistic",    "70.0%", "Fast but flat",     "#3b82f6"),
    ("UCB c=2.0",     "85.7%", "Smart exploration", "#8b5cf6"),
]

for col, (name, pct, verdict, color) in zip(cols, metrics):
    with col:
        st.markdown(
            f"<div style='text-align:center; padding:1rem 0'>"
            f"<div style='font-size:2.4rem; font-weight:700; color:{color}'>{pct}</div>"
            f"<div style='font-size:0.95rem; font-weight:600; margin-top:0.3rem'>{name}</div>"
            f"<div style='font-size:0.82rem; color:#9ca3af'>{verdict}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.caption("*% Optimal Action — how often each algorithm picks the truly best arm.*")

st.divider()

# ── The Problem ─────────────────────────────────────────────────

st.subheader("❓ The Problem in One Sentence")

st.markdown(
    """
    > You have **10 slot machines**, each with a hidden average payout.
    > You have **1,000 pulls**. Earn as much as possible.
    >
    > Do you **exploit** (keep pulling your current best) or
    > **explore** (try others to find something better)?
    """
)

st.divider()

# ── Algorithm Guide ─────────────────────────────────────────────

st.subheader("🤖 The 5 Algorithms")

algo_cols = st.columns(5)

algos = [
    ("🔴", "Greedy",
     "Always pick the current best.\n\nFast but blind — gets stuck permanently.",
     "#ef4444"),
    ("🟡", "ε-greedy",
     "Exploit with prob (1−ε), explore randomly with prob ε.\n\nSimple and effective.",
     "#f59e0b"),
    ("🔵", "Optimistic",
     "Start believing everything is amazing. Reality forces curiosity.\n\nDeterministic exploration.",
     "#3b82f6"),
    ("🟣", "UCB",
     "Favor arms with high uncertainty.\n\nQ(a) + c·√(ln t / N(a))\nSmartest explorer.",
     "#8b5cf6"),
    ("🩷", "Thompson",
     "Bayesian posterior sampling.\n\nProbabilistically balances exploration and exploitation.",
     "#ec4899"),
]

for col, (icon, name, desc, color) in zip(algo_cols, algos):
    with col:
        st.markdown(
            f"<div style='background:#1e293b; padding:1.2rem; "
            f"border-radius:0.8rem; text-align:center; min-height:200px;'>"
            f"<div style='font-size:2rem'>{icon}</div>"
            f"<div style='font-size:1.15rem; font-weight:700; "
            f"color:{color}; margin:0.5rem 0'>{name}</div>"
            f"<div style='font-size:0.83rem; color:#94a3b8; "
            f"line-height:1.5'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Navigation Guide ────────────────────────────────────────────

st.subheader("🗺️ Explore the Pages")

nav_cols = st.columns(4)
pages_info = [
    ("🔬", "Live Simulation",
     "Pull arms step-by-step. Watch Q-values update in real time. "
     "Feel the explore/exploit tradeoff with your own hands."),
    ("📊", "Comparison",
     "Run all 5 algorithms on the same bandit. "
     "Overlay curves and see which strategy wins."),
    ("🌍", "Non-Stationary",
     "The world shifts mid-game. Watch sample-average fail and "
     "constant-α recover — the core justification for RL design."),
    ("📝", "Findings",
     "Key insights with evidence. The takeaways for interviews, "
     "viva prep, and real-world system design."),
]

for col, (icon, title, desc) in zip(nav_cols, pages_info):
    with col:
        st.markdown(
            f"<div style='background:#0f172a; border:1px solid #334155; "
            f"padding:1.2rem; border-radius:0.8rem; text-align:center; "
            f"min-height:180px;'>"
            f"<div style='font-size:2.2rem'>{icon}</div>"
            f"<div style='font-size:1.1rem; font-weight:600; "
            f"margin:0.5rem 0'>{title}</div>"
            f"<div style='font-size:0.85rem; color:#94a3b8; "
            f"line-height:1.5'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Key Insight ─────────────────────────────────────────────────

st.subheader("🔑 The Core Lesson")

st.markdown(
    """
    **Pure exploitation is a trap.** Greedy gets stuck at ~35% optimal —
    it locks onto a "pretty good" arm and never discovers the truly best one.

    **Strategic exploration pays compounding returns.** UCB reaches ~86%
    optimal by exploring *only where uncertain* — not randomly, not always,
    but intelligently.

    **The right balance depends on context.** The best ε depends on your
    budget. Constant α adapts when the world changes. There is no
    one-size-fitsall — only principled tradeoffs.

    > *"The exploration-exploitation dilemma is not a problem to be solved,
    > but a tension to be managed."*
    """
)

st.divider()
st.caption(
    "Built to internalise [Sutton & Barto, Ch. 2]"
    "(http://incompleteideas.net/book/the-book.html) — then apply it. "
    "Run `streamlit run app.py` to explore."
)