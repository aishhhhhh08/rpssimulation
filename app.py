"""
State / Militia / Intelligentsia -- Convergence Simulator (web app)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for a shareable link: see README.md.

This file is deliberately thin -- it imports sim_engine.py and
sim_visualize.py directly (the same files used in the notebook) rather
than re-implementing the model, so the web app and the notebook can never
drift out of sync with each other.
"""

import streamlit as st
import streamlit.components.v1 as components

from sim_engine import CARDS, SimConfig, run_simulation, edge_label
from sim_visualize import build_animation, final_message

DECK_LABEL = {"S": "State", "M": "Militia", "I": "Intelligentsia"}

st.set_page_config(page_title="Convergence Simulator", layout="wide")
st.title("State \u00b7 Militia \u00b7 Intelligentsia \u2014 Convergence Simulator")
st.caption(
    "Three actors locked in a rock-paper-scissors rivalry, coupled to a public "
    "that normalizes around whoever is winning. Toggle pre-conditions to see "
    "whether one actor wins outright, two actors form a coalition that squeezes "
    "the third out, or the rivalry never resolves."
)

with st.sidebar:
    st.header("Pre-conditions")
    st.caption(
        "Each card has a checkbox (on/off) and ONE theta value (0-1, default 1.0). "
        "That single value is multiplied against this card's FIXED directional "
        "coefficients (shown under each card) -- so the same theta produces a "
        "different, automatically-differentiated effect on each actor, since the "
        "base coefficients themselves already differ in sign and size."
    )

    active_cards = []
    card_weights = {}
    for deck in ("S", "M", "I"):
        st.subheader(f"{DECK_LABEL[deck]} deck")
        for cid, card in CARDS.items():
            if card["deck"] != deck:
                continue
            col1, col2 = st.columns([3, 1])
            with col1:
                on = st.checkbox(card["label"], value=False, key=f"chk_{cid}")
            with col2:
                wt = st.number_input(
                    "theta", min_value=0.0, max_value=1.0, value=1.0, step=0.01,
                    key=f"wt_{cid}", label_visibility="collapsed",
                )
            edge_desc = "  |  ".join(f"{edge_label(e)}: base {b:+.1f}" for e, b in card["weights"].items())
            st.caption(edge_desc)
            if on:
                active_cards.append(cid)
            card_weights[cid] = wt

    st.divider()
    st.header("Settings")
    n_agents = st.slider("N agents", 50, 1000, 300, step=50)
    years = st.slider("Years", 1, 300, 100, step=1)
    eta = st.slider("eta", 0.01, 0.30, 0.05, step=0.01)
    drift_rate = st.slider("drift (mu)", 0.01, 0.50, 0.05, step=0.01)
    indiv_thresh = st.slider("Individual win %", 0.50, 0.95, 0.70, step=0.01)
    combo_thresh = st.slider("Combined win %", 0.70, 0.99, 0.90, step=0.01)
    speed_ms = st.slider("ms / year (playback speed)", 100, 2000, 600, step=100)
    seed = st.number_input("seed", value=7, step=1)

    st.divider()
    st.subheader("Starting strengths")
    s_s = st.slider("State s0", 0.0, 1.0, 1 / 3, step=0.05)
    s_m = st.slider("Militia s0", 0.0, 1.0, 1 / 3, step=0.05)
    s_i = st.slider("Intel s0", 0.0, 1.0, 1 / 3, step=0.05)
    st.caption("Rescaled to sum to 100% -- these three represent a share of power.")

    st.divider()
    precond_w = st.slider("Precondition wt", 0.0, 1.5, 0.5, step=0.05,
                            help="How much active preconditions push the rivalry itself.")
    legit_w = st.slider("Legitimacy wt", 0.0, 1.0, 0.3, step=0.05,
                          help="How much active preconditions shift public opinion, independent of raw power.")

    run_button = st.button("\u25b6 Run simulation", type="primary", use_container_width=True)

if run_button:
    total = s_s + s_m + s_i
    init = {"S": s_s / total, "M": s_m / total, "I": s_i / total}

    if min(init.values()) < 0.12:
        st.warning(
            "Your third actor is starting quite low -- the combined-dominance "
            "check may trigger almost immediately rather than showing real dynamics."
        )

    cfg = SimConfig(
        active_cards=active_cards, n_agents=n_agents, eta=eta, drift_rate=drift_rate,
        years=years, absorption_threshold=indiv_thresh, combo_threshold=combo_thresh,
        playback_interval_ms=speed_ms, precondition_weight=precond_w,
        legitimacy_weight=legit_w, card_weights=card_weights, seed=int(seed),
    )

    st.write(f"**Active cards:** {', '.join(CARDS[c]['label'] for c in active_cards) if active_cards else '(none)'}")

    with st.spinner("Running simulation..."):
        result = run_simulation(cfg, init)
        anim, _ = build_animation(cfg, init, sim_result=result)
        anim_html = anim.to_jshtml()

    components.html(anim_html, height=980, scrolling=True)
    st.markdown(final_message(result, cfg, active_cards))
else:
    st.info("Set your pre-conditions and settings in the sidebar, then press **Run simulation**.")
