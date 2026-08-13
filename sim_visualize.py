"""
Visualization v3 --

Layout (top to bottom):
  1. Stacked area chart (replaces the donut) -- S/M/I share of power over
     time, filled area growing year by year. Easier for a layman to read
     as "who owns how much of the pie, and how has that changed" than a
     single-frame donut.
  2. Public network graph -- unchanged, still colors nodes by this year's
     tolerate/sanction decision.
  3. Line trace -- S/M/I strength + the dotted public-acceptance line,
     unchanged from before.
  4. Narrator strip -- one auto-generated plain-language sentence per
     frame, describing who's leading, whether they're gaining or losing
     ground, and where public opinion stands. This is template-based
     (not a language model) -- it reads off the same numbers the charts
     show, just in a sentence instead of a line.

After the animation ends, final_message() produces a longer written
explanation of WHY the run ended the way it did (or didn't converge).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from sim_engine import ACTOR_LABEL, CARDS, run_simulation, precondition_scores, precondition_breakdown

ACTOR_COLOR = {"S": "#3B8AD4", "M": "#D85A30", "I": "#5DCAA5"}


def _trend_word(delta, up="gaining ground", down="losing ground", flat="holding steady", eps=0.01):
    if delta > eps:
        return up
    if delta < -eps:
        return down
    return flat


def narrate(history, cfg, active_cards=None):
    """One sentence describing the current tick, using only numbers already
    on screen -- a template, not free-form generation."""
    s = history[-1]
    year = len(history) - 1
    dominant = max(("S", "M", "I"), key=lambda k: s[k])
    share = s[dominant]
    pub = s["public_T"]

    lookback = min(5, len(history) - 1)
    prev = history[-1 - lookback] if lookback > 0 else s
    power_trend = _trend_word(share - prev[dominant])
    pub_trend = _trend_word(pub - prev["public_T"], up="rising", down="falling", flat="holding steady")

    dist = max(cfg.absorption_threshold - share, 0) * 100
    score_note = ""
    if active_cards is not None:
        score = precondition_scores(active_cards, cfg.card_weights)[dominant]
        score_note = f" {ACTOR_LABEL[dominant]} currently has a precondition score of {score:.2f}."
    return (f"Year {year}: {ACTOR_LABEL[dominant]} leads with {share*100:.0f}% of power and is {power_trend}. "
            f"Public acceptance of {ACTOR_LABEL[dominant]} is {pub*100:.0f}% and {pub_trend}. "
            f"{ACTOR_LABEL[dominant]} needs {dist:.0f} more points to rule outright "
            f"({cfg.absorption_threshold*100:.0f}%).{score_note}")


def final_message(result, cfg, active_cards):
    """Longer, end-of-run explanation of why the simulation ended the way
    it did. Rule-based: it cites the actual thresholds crossed, the cards
    that were active for the relevant actor(s), and -- for a non-converged
    run -- how often the lead changed hands as a rough stand-in for
    'the cycle kept balancing itself out'."""
    outcome = result["outcome"]
    history = result["history"]
    s = history[-1]
    year = result["years_run"]
    scores = precondition_scores(active_cards, cfg.card_weights)
    score_line = ", ".join(f"{ACTOR_LABEL[a]} {scores[a]:.2f}" for a in ("S", "M", "I"))

    def cards_for(deck):
        breakdown = precondition_breakdown(deck, active_cards, cfg.card_weights)
        if not breakdown:
            return "none directly -- the base rivalry math and public normalization alone were enough"
        return ", ".join(f"{label} ({pct:.0f}%)" for label, val, pct in breakdown)

    precond_note = f"**Precondition score (active cards per actor):** {score_line}."

    if outcome["type"] == "individual":
        w = outcome["winner"]
        return (f"### Result: {ACTOR_LABEL[w]} achieved outright dominance\n\n"
                f"By year {year}, {ACTOR_LABEL[w]} controlled {s[w]*100:.0f}% of total power, crossing the "
                f"{cfg.absorption_threshold*100:.0f}% outright-dominance threshold on its own. Public acceptance "
                f"of {ACTOR_LABEL[w]} stood at {s['public_T']*100:.0f}% at that point.\n\n"
                f"{precond_note}\n\n"
                f"**Precondition breakdown for {ACTOR_LABEL[w]}** (share of {ACTOR_LABEL[w]}'s own score): "
                f"{cards_for(w)}.")

    elif outcome["type"] == "duopoly":
        a, b = outcome["pair"]
        excl = outcome["excluded"]
        return (f"### Result: {ACTOR_LABEL[a]} + {ACTOR_LABEL[b]} broke out of the rivalry together\n\n"
                f"By year {year}, {ACTOR_LABEL[a]} and {ACTOR_LABEL[b]} together held "
                f"{outcome['combined']*100:.0f}% of total power, squeezing {ACTOR_LABEL[excl]} down to "
                f"{s[excl]*100:.0f}% -- crossing the {cfg.combo_threshold*100:.0f}% combined-dominance threshold, "
                f"even though **neither individually reached the {cfg.absorption_threshold*100:.0f}% outright-win "
                f"line**. This is the model's answer to whether two actors can co-dominate and marginalize the "
                f"third without either one 'winning' alone: here, yes.\n\n"
                f"{precond_note}\n\n"
                f"**Precondition breakdown** (share of each actor's own score): "
                f"{ACTOR_LABEL[a]} -- {cards_for(a)}. "
                f"{ACTOR_LABEL[b]} -- {cards_for(b)}.")

    else:
        w = max(("S", "M", "I"), key=lambda k: s[k])
        leaders = [max(("S", "M", "I"), key=lambda k: h[k]) for h in history]
        switches = sum(1 for i in range(1, len(leaders)) if leaders[i] != leaders[i - 1])
        return (f"### Result: no convergence within {cfg.tick_horizon} years\n\n"
                f"{ACTOR_LABEL[w]} ended with the largest share ({s[w]*100:.0f}%), but never crossed the "
                f"{cfg.absorption_threshold*100:.0f}% individual threshold, and no pair ever reached the "
                f"{cfg.combo_threshold*100:.0f}% combined threshold. Leadership changed hands {switches} times "
                f"over the run -- consistent with the rock-paper-scissors cycle balancing itself out rather than "
                f"collapsing into a winner or a two-way alliance. Public acceptance of the current leader was "
                f"{s['public_T']*100:.0f}% at the end.\n\n"
                f"{precond_note}\n\n"
                f"**Precondition breakdown for {ACTOR_LABEL[w]}** (current leader, share of its own score): "
                f"{cards_for(w)}.\n\n"
                f"**Why it likely didn't converge:** either no actor holds more active preconditions than the "
                f"others (so their structural scores cancel out, as here), or the active cards weren't stacked "
                f"heavily enough on one side. Try stacking multiple cards onto a single actor (for an individual "
                f"win) or onto two actors against the third (for a duopoly) to push past a threshold.")


def build_animation(cfg, initial_strengths, sim_result=None, figsize=(11, 9), frame_stride=1):
    if sim_result is None:
        sim_result = run_simulation(cfg, initial_strengths)

    history = sim_result["history"]
    decisions_history = sim_result["decisions_history"]
    public = sim_result["public"]
    n_frames = len(decisions_history)
    pos = nx.spring_layout(public.graph, seed=7)

    fig = plt.figure(figsize=figsize, facecolor="#0d0f10")
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.1, 0.8], hspace=0.55, wspace=0.25)
    ax_area = fig.add_subplot(gs[0, 0])
    ax_net = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])
    ax_narr = fig.add_subplot(gs[2, :])
    ax_narr.axis("off")

    def draw_area(frame_idx, ax):
        ax.clear()
        ax.set_facecolor("#0d0f10")
        upto = frame_idx + 2
        xs = list(range(upto))
        ys_S = [history[i]["S"] for i in range(upto)]
        ys_M = [history[i]["M"] for i in range(upto)]
        ys_I = [history[i]["I"] for i in range(upto)]
        ax.stackplot(xs, ys_S, ys_M, ys_I,
                     colors=[ACTOR_COLOR["S"], ACTOR_COLOR["M"], ACTOR_COLOR["I"]],
                     labels=["State", "Militia", "Intelligentsia"], alpha=0.9)
        ax.set_xlim(0, n_frames + 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_title(f"Year {frame_idx+1}/{n_frames} -- share of power", color="white", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, facecolor="#1a1c1d", labelcolor="white", framealpha=0.6)

    def draw_network(frame_idx, ax):
        ax.clear(); ax.axis("off")
        s = history[frame_idx + 1] if frame_idx + 1 < len(history) else history[-1]
        dominant = max(("S", "M", "I"), key=lambda k: s[k])
        decisions = decisions_history[frame_idx]
        # Tolerating nodes take on the DOMINANT ACTOR's own color (matches the
        # stacked area / trace panels), so "public accepts State" actually
        # shows blue, "accepts Militia" shows orange, "accepts Intelligentsia"
        # shows teal -- instead of a flat green regardless of who's in charge.
        # Sanctioning nodes stay a neutral muted red.
        tolerate_color = ACTOR_COLOR[dominant]
        sanction_color = "#8A3A3A"
        colors = [tolerate_color if d else sanction_color for d in decisions]
        nx.draw_networkx_edges(public.graph, pos, ax=ax, edge_color="#333", width=0.4, alpha=0.5)
        nx.draw_networkx_nodes(public.graph, pos, ax=ax, node_color=colors, node_size=25, linewidths=0)
        legend_handles = [
            mpatches.Patch(color=tolerate_color, label=f"tolerates {ACTOR_LABEL[dominant]}"),
            mpatches.Patch(color=sanction_color, label="sanctions"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
                  facecolor="#1a1c1d", labelcolor="white", framealpha=0.6)
        t_frac = decisions.mean()
        ax.set_title(f"public toward {ACTOR_LABEL[dominant]}  "
                      f"tolerate {t_frac*100:.0f}% / sanction {(1-t_frac)*100:.0f}%",
                      color="white", fontsize=10)

    def draw_trace(frame_idx, ax):
        ax.clear()
        ax.set_facecolor("#0d0f10")
        xs = list(range(frame_idx + 2))
        for actor in ["S", "M", "I"]:
            ys = [history[i][actor] for i in range(frame_idx + 2)]
            ax.plot(xs, ys, color=ACTOR_COLOR[actor], linewidth=2, label=ACTOR_LABEL[actor])
        pub_ys = [history[i]["public_T"] for i in range(frame_idx + 2)]
        ax.plot(xs, pub_ys, color="white", linewidth=2, linestyle=":", label="Public accepts")
        ax.axhline(cfg.absorption_threshold, color="#888", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#555", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, n_frames + 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.legend(loc="upper left", fontsize=8, facecolor="#1a1c1d", labelcolor="white", framealpha=0.6)

    def draw_narration(frame_idx, ax):
        ax.clear(); ax.axis("off")
        text = narrate(history[:frame_idx + 2], cfg, cfg.active_cards)
        ax.text(0.02, 0.5, text, ha="left", va="center", fontsize=10.5, color="#e8e8e0",
                wrap=True, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#1a1c1d", edgecolor="#333"))

    def update(frame_idx):
        draw_area(frame_idx, ax_area)
        draw_network(frame_idx, ax_net)
        draw_trace(frame_idx, ax_trace)
        draw_narration(frame_idx, ax_narr)
        return []

    anim = FuncAnimation(fig, update, frames=range(0, n_frames, frame_stride),
                          interval=cfg.playback_interval_ms, blit=False)
    plt.close(fig)
    return anim, sim_result
