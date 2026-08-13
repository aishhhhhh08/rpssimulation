"""
Core simulation engine, v3 --

New in this version:
  - classify_outcome(): checks for TWO different kinds of convergence --
      (a) INDIVIDUAL absorption: any one actor >= absorption_threshold (0.70)
      (b) DUOPOLY breakout: any two actors' COMBINED share >= combo_threshold
          (0.90), meaning the third actor has been squeezed out even though
          neither of the two winners individually crossed 70%. This directly
          answers "can a combination break out of the RPS cycle and rule
          together" -- it's a distinct outcome type, checked every tick
          alongside the individual check.
  - 1 tick = 1 year (tick_horizon is now presented to the user as "years").
  - history entries now carry everything the narrator needs.
"""

import numpy as np
import networkx as nx

BASE_MATRIX = {
    "a_SM": -0.3, "a_MS": +0.8,
    "a_MI": -0.2, "a_IM": +0.5,
    "a_IS": -0.4, "a_SI": +0.6,
}

CARDS = {
    "polarization": {
        "deck": "S", "label": "Political polarization",
        "weights": {"a_SI": +0.8, "a_SM": +0.4},
        "lambda_mult": {"S": 0.6},
    },
    "media_capture": {
        "deck": "S", "label": "Media capture",
        "weights": {"a_SI": +0.9, "a_IS": -0.7},
        "drift_rate_mult": {"S": 0.4},
        "lambda_mult": {"S": 0.6},
    },
    "judicial_capture": {
        "deck": "S", "label": "Judicial & electoral capture",
        "weights": {"a_SI": +0.6, "a_SM": +0.3},
    },
    "repressive_dependency": {
        "deck": "M", "label": "State repressive dependency",
        "weights": {"a_MS": +1.2, "a_MI": -0.3},
    },
    "military_autonomy": {
        "deck": "M", "label": "Military autonomy",
        "weights": {"a_MS": +1.0, "a_IS": -0.5},
        "lambda_mult": {"M": 0.5},
    },
    "macro_collapse": {
        "deck": "M", "label": "Macroeconomic collapse",
        "weights": {"a_MS": +0.7, "a_IS": -0.4},
        "risk_tolerance_shift": {"M": -0.15},
    },
    "legitimacy_collapse": {
        "deck": "I", "label": "State legitimacy collapse",
        "weights": {"a_IS": +1.1, "a_IM": +0.8},
    },
    "civil_society_density": {
        "deck": "I", "label": "Civil society network density",
        "weights": {"a_IS": +0.7, "a_IM": +0.6},
        "drift_rate_mult": {"I": 2.2},
    },
    "ideological_fanaticism": {
        "deck": "I", "label": "Ideological fanaticism",
        "weights": {"a_IM": +1.0, "a_MI": -0.8},
        "k_mult": {"M": 0.15},
    },
}

ACTOR_LABEL = {"S": "State", "M": "Militia", "I": "Intelligentsia"}


class SimConfig:
    def __init__(self, active_cards=None, n_agents=500, k=5.0, eta=0.15,
                 drift_rate=0.08, lambda_base=None, network_type="watts_strogatz",
                 years=150, absorption_threshold=0.70, combo_threshold=0.90,
                 playback_interval_ms=500, precondition_weight=0.5,
                 legitimacy_weight=0.3, card_weights=None,
                 sustained_years=20, plurality_margin=0.10, plurality_floor=0.20,
                 seed=None):
        self.active_cards = active_cards or []
        self.n_agents = n_agents
        self.k = k
        self.eta = eta
        self.drift_rate = drift_rate
        self.lambda_base = lambda_base or {"S": 0.5, "M": 0.5, "I": 0.5}
        self.network_type = network_type
        self.tick_horizon = years          # 1 tick = 1 simulated year
        self.absorption_threshold = absorption_threshold
        self.combo_threshold = combo_threshold
        self.playback_interval_ms = playback_interval_ms
        # How strongly ACTIVE PRECONDITIONS (not current strength) push the
        # outcome. Each card an actor has active counts as one unit of
        # "structural score"; this is mean-centered across the three actors,
        # so an actor with zero active cards actively LOSES ground relative
        # to one that has cards -- it no longer free-rides on the base
        # matrix's built-in cyclic asymmetries. precondition_weight scales
        # this into the rivalry (fitness); legitimacy_weight scales it into
        # how legitimate/tolerable the public perceives the dominant actor,
        # independent of that actor's raw power share.
        self.precondition_weight = precondition_weight
        self.legitimacy_weight = legitimacy_weight
        # Per-card weight, e.g. {"polarization": 0.3, "media_capture": 0.1, ...}.
        # A card not listed here defaults to weight 1.0. This is what lets
        # you make one card matter more than another, instead of every
        # active card counting as an equal flat unit.
        self.card_weights = card_weights or {}
        # A THIRD, separate convergence tier, distinct from individual/duopoly
        # absorption. Some actors never cross 70% (or 90% combined) but still
        # hold a real, persistent structural lead -- e.g. a plurality that
        # never gives way for decades. That pattern is a legitimate form of
        # "de facto control" in the political-science sense (dominance over
        # levers of power) even without crossing an abstract power-share
        # threshold. This is checked ONLY if neither individual nor duopoly
        # ever triggers, and requires the SAME actor to lead by at least
        # plurality_margin (a fraction, e.g. 0.10 = 10 percentage points)
        # over the second-place actor, continuously, for sustained_years in
        # a row. These two defaults are neutral, chosen independently of any
        # specific country case -- they should be justified (or recalibrated)
        # against several historical cases, not tuned to make one case pass.
        self.sustained_years = sustained_years
        self.plurality_margin = plurality_margin
        # ALSO require both rivals to individually fall below this floor
        # (e.g. 0.20 = neither rival holds even a fifth of total power),
        # sustained for the same window -- not just trailing the leader by
        # plurality_margin. Without this, a case where the "leader" pulls
        # ahead while both rivals still hold a meaningful ~20-30% each (a
        # normal three-way spread, not a squeeze-out) gets misclassified as
        # de facto capture. NOTE: this catches early false positives, but
        # does not fully resolve long-horizon cases -- see run_simulation's
        # docstring note on this.
        self.plurality_floor = plurality_floor
        self.rng = np.random.default_rng(seed)


def precondition_score(actor, active_cards, card_weights=None):
    """Sum of this actor's active cards' intensity (default 1.0 each if no
    override was given). Accepts the same two card_weights formats as
    compute_payoff_matrix: a flat scalar per card (used directly), or a
    per-edge dict (averaged across that card's edges to get one scalar
    'how strongly is this card present' summary)."""
    card_weights = card_weights or {}
    total = 0.0
    for c in active_cards:
        if CARDS[c]["deck"] != actor:
            continue
        override = card_weights.get(c, 1.0)
        if isinstance(override, dict):
            edges = CARDS[c]["weights"].keys()
            mults = [override.get(e, 1.0) for e in edges]
            total += sum(mults) / len(mults)
        else:
            total += override
    return total


def precondition_scores(active_cards, card_weights=None):
    return {a: precondition_score(a, active_cards, card_weights) for a in ("S", "M", "I")}


def edge_label(edge):
    """'a_SI' -> 'State \u2192 Intelligentsia' -- human-readable label for a
    directional matrix edge, used in the per-edge weight UI."""
    a, b = edge[2], edge[3]
    return f"{ACTOR_LABEL[a]} \u2192 {ACTOR_LABEL[b]}"


def compute_payoff_matrix(cfg):
    """Each active card's edges are scaled by cfg.card_weights[cid], which
    can be EITHER:
      - a flat scalar (old behavior, backward compatible): the same
        multiplier applies to every edge that card touches, e.g. 0.5 halves
        both of media_capture's edges equally.
      - a dict of {edge: multiplier} (new): each edge gets its OWN
        independent 0-1 intensity multiplier on top of that edge's fixed
        base coefficient -- e.g. {'a_SI': 0.9, 'a_IS': 0.2} applies 90% of
        the State-on-Intelligentsia effect but only 20% of the
        Intelligentsia-on-State effect for that same card. Missing edges in
        the dict default to multiplier 1.0 (full base effect).
    A card not present in cfg.card_weights defaults to multiplier 1.0 on
    every edge (full base effect, unchanged from the pre-weighting design).
    """
    a = dict(BASE_MATRIX)
    for cid in cfg.active_cards:
        override = cfg.card_weights.get(cid, 1.0)
        for edge, default_w in CARDS[cid]["weights"].items():
            mult = override.get(edge, 1.0) if isinstance(override, dict) else override
            a[edge] += default_w * mult
    return a


def get_lambda(cfg):
    lam = dict(cfg.lambda_base)
    for cid in cfg.active_cards:
        for actor, mult in CARDS[cid].get("lambda_mult", {}).items():
            lam[actor] *= mult
    return lam


def get_k_for_actor(cfg, actor):
    k = cfg.k
    for cid in cfg.active_cards:
        for a_, mult in CARDS[cid].get("k_mult", {}).items():
            if a_ == actor:
                k *= mult
    return k


def get_drift_rate(cfg, actor):
    mu = cfg.drift_rate
    for cid in cfg.active_cards:
        for a_, mult in CARDS[cid].get("drift_rate_mult", {}).items():
            if a_ == actor:
                mu *= mult
    return float(np.clip(mu, 0.0, 0.9))


def compute_fitness(strengths, a, public_frac, lam, dominant):
    f = {
        "S": strengths["M"] * a["a_SM"] + strengths["I"] * a["a_SI"],
        "M": strengths["S"] * a["a_MS"] + strengths["I"] * a["a_MI"],
        "I": strengths["S"] * a["a_IS"] + strengths["M"] * a["a_IM"],
    }
    net = public_frac["T"] - public_frac["P"]
    f[dominant] += lam[dominant] * net * strengths[dominant]
    return f


def update_strengths(strengths, f, eta):
    shift = -min(f.values()) + (1.0 / eta) if min(f.values()) * eta <= -1.0 else 0.0
    weighted = {k: strengths[k] * (1 + eta * (f[k] + shift)) for k in strengths}
    total = sum(weighted.values())
    return {k: weighted[k] / total for k in weighted}


def get_dominant_actor(strengths):
    return max(strengths, key=strengths.get)


def classify_outcome(strengths, absorption_threshold, combo_threshold):
    """Returns a dict describing which (if any) convergence condition has
    been met this tick. Individual check runs first; the pairwise/duopoly
    check only matters if no single actor has already won outright."""
    for actor, s in strengths.items():
        if s >= absorption_threshold:
            return {"type": "individual", "winner": actor}
    for a, b in [("S", "M"), ("M", "I"), ("S", "I")]:
        excluded = ({"S", "M", "I"} - {a, b}).pop()
        combined = strengths[a] + strengths[b]
        if combined >= combo_threshold:
            return {"type": "duopoly", "pair": (a, b), "excluded": excluded, "combined": combined}
    return {"type": None}


def check_sustained_plurality(history, sustained_years, plurality_margin, plurality_floor):
    """Third convergence tier: has the SAME actor led by at least
    plurality_margin over the second-place actor, AND kept BOTH rivals
    below plurality_floor individually, continuously, for the last
    sustained_years ticks? Returns None if not (yet) satisfied -- called
    only when neither individual nor duopoly has triggered."""
    if len(history) <= sustained_years:
        return None
    window = history[-sustained_years:]
    leaders = [max(("S", "M", "I"), key=lambda k: h[k]) for h in window]
    if len(set(leaders)) != 1:
        return None
    leader = leaders[0]
    for h in window:
        rivals = [h[a] for a in ("S", "M", "I") if a != leader]
        second = max(rivals)
        if h[leader] - second < plurality_margin:
            return None
        if max(rivals) >= plurality_floor:
            return None
    last = window[-1]
    second = max(last[a] for a in ("S", "M", "I") if a != leader)
    return {
        "type": "sustained_plurality",
        "leader": leader,
        "share": last[leader],
        "margin": last[leader] - second,
        "public_T": last["public_T"],
    }


class PublicLayer:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.n_agents
        self.r = cfg.rng.beta(2, 2, size=n)
        self.last_decision = cfg.rng.random(n) < 0.5

        if cfg.network_type == "watts_strogatz":
            k_neighbors = max(4, int(np.log(n)) * 2)
            self.graph = nx.watts_strogatz_graph(n, k_neighbors, 0.1, seed=int(cfg.rng.integers(1e9)))
        else:
            self.graph = nx.erdos_renyi_graph(n, 6.0 / n, seed=int(cfg.rng.integers(1e9)))

        self._neighbor_list = [list(self.graph.neighbors(i)) for i in range(n)]

    def step(self, dominant_actor, dominant_strength):
        cfg = self.cfg
        n = cfg.n_agents
        k = get_k_for_actor(cfg, dominant_actor)
        mu = get_drift_rate(cfg, dominant_actor)

        risk_shift = 0.0
        for cid in cfg.active_cards:
            risk_shift += CARDS[cid].get("risk_tolerance_shift", {}).get(dominant_actor, 0.0)
        effective_r = np.clip(self.r + risk_shift, 0.01, 0.99)

        p = 1.0 / (1.0 + np.exp(-k * (dominant_strength - effective_r)))
        draws = cfg.rng.random(n)
        decisions = draws < p
        t_frac = decisions.mean()

        local_tolerate_frac = np.empty(n)
        for i in range(n):
            neigh = self._neighbor_list[i]
            local_tolerate_frac[i] = np.mean(self.last_decision[neigh]) if neigh else t_frac
        target = 1.0 - local_tolerate_frac
        self.r = (1 - mu) * self.r + mu * target

        self.last_decision = decisions
        return {"T": t_frac, "P": 1 - t_frac}, decisions


def run_simulation(cfg, initial_strengths):
    strengths = dict(initial_strengths)
    public = PublicLayer(cfg)
    history = [dict(strengths, public_T=0.5)]
    decisions_history = []
    outcome = {"type": None}
    t = 0

    scores = precondition_scores(cfg.active_cards, cfg.card_weights)
    mean_score = sum(scores.values()) / 3.0

    for t in range(1, cfg.tick_horizon + 1):
        a = compute_payoff_matrix(cfg)
        lam = get_lambda(cfg)
        dominant = get_dominant_actor(strengths)

        # Preconditions shift how legitimate/tolerable the public perceives
        # the dominant actor, on top of (not instead of) their raw power --
        # this is the direct "do preconditions move public judgment" channel.
        legitimacy_bonus = cfg.legitimacy_weight * (scores[dominant] - mean_score)
        perceived_strength = float(np.clip(strengths[dominant] + legitimacy_bonus, 0.01, 0.99))
        public_frac, decisions = public.step(dominant, perceived_strength)
        decisions_history.append(decisions.copy())

        f = compute_fitness(strengths, a, public_frac, lam, dominant)
        # Preconditions also push the rivalry itself directly, mean-centered
        # so an actor with no active cards actively loses ground relative to
        # one that has them, rather than merely growing slower.
        for actor in ("S", "M", "I"):
            f[actor] += cfg.precondition_weight * (scores[actor] - mean_score)

        strengths = update_strengths(strengths, f, cfg.eta)
        history.append(dict(strengths, public_T=public_frac["T"]))

        outcome = classify_outcome(strengths, cfg.absorption_threshold, cfg.combo_threshold)
        if outcome["type"] is None:
            sp = check_sustained_plurality(history, cfg.sustained_years, cfg.plurality_margin, cfg.plurality_floor)
            if sp is not None:
                outcome = sp
        if outcome["type"] is not None:
            break

    return {
        "history": history,
        "decisions_history": decisions_history,
        "outcome": outcome,
        "years_run": t,
        "public": public,
        "precondition_scores": scores,
    }
