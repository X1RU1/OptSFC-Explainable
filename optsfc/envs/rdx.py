import numpy as np
import torch

rewards_coeff = [0.4, 0.3, 0.3]
OBJ_NAMES     = ["Resource", "Network", "Security"]

# ── Observation array layout ──────────────────────────────────────────────────
# Produced by dict_observation_to_array(), which calls np.hstack([v.ravel()
# for v in observation.values()]) over observation_dictionary key order.
#
# observation_dictionary keys and shapes (vnfs_size=4, cnfs_size=3 → N_RES=7,
#                                          vims_size=2 → N_VIM=2):
#
#   Key                   shape        ravel len   start idx
#   ─────────────────────────────────────────────────────────
#   nb_resources          (1,)              1           0
#   vim_resources         (2, 3)            6           1
#   id                    (7, 1)            7           7
#   apt_scores            (7, 2)           14          14
#   data_leak_scores      (7, 2)           14          28
#   dos_scores            (7, 2)           14          42
#   undefined_scores      (7, 2)           14          56
#   mtd_resource_overhead (7, 1)            7          70
#   network_penalty       (7, 1)            7          77
#   security_penalty      (7, 1)            7          84
#   nb_UEs_cnx            (7, 1)            7          91
#   vim_host              (7, 1)            7          98
#   mtd_action            (7, 2)           14         105
#   mtd_constraint        (7, 2)           14         119
#   ─────────────────────────────────────────────────────────
#   Total                                133

_N_RES  = 7
_N_VIM  = 2

_SL_NB_RES    = slice(0,   1)
_SL_VIM_RES   = slice(1,   7)
_SL_ID        = slice(7,  14)
_SL_APT       = slice(14, 28)
_SL_DATALEAK  = slice(28, 42)
_SL_DOS       = slice(42, 56)
_SL_UNDEF     = slice(56, 70)
_SL_MTD_OHD   = slice(70, 77)
_SL_NET_PEN   = slice(77, 84)
_SL_SEC_PEN   = slice(84, 91)
_SL_NB_UES    = slice(91, 98)
_SL_VIM_HOST  = slice(98, 105)
_SL_MTD_ACT   = slice(105, 119)
_SL_MTD_CON   = slice(119, 133)


def extract_state_features(obs_array: np.ndarray, env) -> dict:
    """
    Derive interpretable scalar state features from a flat observation array
    and the live environment instance.

    Feature classification
    ──────────────────────
    Resource  : VIM0/VIM1 CPU+RAM remaining, Mean MTD overhead ($)
                VIM remaining captures available capacity per host.
                MTD overhead is the active economic cost of running MTD
                actions (non-zero only when mtd_action != 0); distinct from
                VIM remaining because it measures instantaneous spend, not
                capacity headroom.

    Network   : Mean/Max network penalty, Min/Mean remaining migrations,
                Mean remaining reinstantiations, Total UEs connected.
                Remaining budget features are Network indicators because
                their limit is the service-downtime SLA (QoS constraint),
                not a resource limit (confirmed by supervisor Q5).
                Total UEs drives the latency/throughput simulation and
                provides causal context that the penalty alone cannot give.

    Security  : Mean/Max security penalty.
                All upstream inputs (CVSS scores, impact_ssla) are static
                in this simulation (std=0 across episodes); their effect
                is fully captured by the computed penalty. No additional
                security features are needed.

    Parameters
    ----------
    obs_array : np.ndarray
        Flat observation produced by dict_observation_to_array() before the
        current step's action was applied (obs_before_step).
    env : MOfiveG_net
        Live environment instance. Used only to read:
          - step_counter
          - last_mtd_step
          - security_penalty_cumul

    Returns
    -------
    dict of float scalars keyed by "feat_*" prefix.

    Notes on retained-but-excluded features
    ────────────────────────────────────────
    The following features are computed and returned for completeness /
    future use but are NOT listed in FEAT_META (morl_single_step.py) and
    therefore do not appear in state-context plots or SHAP analyses:

      feat_max_apt_score, feat_mean_apt_score,
      feat_max_dataleak_score, feat_mean_dataleak_score,
      feat_max_dos_score, feat_mean_dos_score
          → std=0 across episodes (static vulnerability catalogue).
            Dropped from analysis; retained here for traceability.

      feat_min_remaining_reinst
          → std=0 in current simulation data (budget never reaches 0
            for reinstantiations within one episode). Retained for
            future scenarios where reinstantiation budget is tighter.

      feat_nb_resources
          → constant (7 VNFs+CNFs in all episodes). No information value.

      feat_security_penalty_cumul
          → episode-level accumulator; semantics differ from single-step
            features. Kept commented out in FEAT_META.
    """
    nb_res = int(obs_array[_SL_NB_RES][0])
    active = slice(0, max(nb_res, 1))

    # ── VIM resource layout ───────────────────────────────────────────────────
    # vim_resources shape: (N_VIM, 3) → columns are [cpu, ram_gb, disk_gb].
    # We expose cpu and ram per VIM; disk is omitted (not in reward formula).
    vim_res = obs_array[_SL_VIM_RES].reshape(_N_VIM, 3)
    vim0_cpu, vim0_ram, _vim0_disk = vim_res[0]
    vim1_cpu, vim1_ram, _vim1_disk = vim_res[1]

    # ── CVSS score averages (retained but excluded from FEAT_META) ────────────
    # apt_scores / data_leak_scores / dos_scores shape: (N_RES, 2).
    # Column 0 = cvss_avg, column 1 = asp_avg (as set in update_agent_obs).
    apt_avg      = obs_array[_SL_APT     ].reshape(_N_RES, 2)[active, 0]
    dataleak_avg = obs_array[_SL_DATALEAK].reshape(_N_RES, 2)[active, 0]
    dos_avg      = obs_array[_SL_DOS     ].reshape(_N_RES, 2)[active, 0]

    # ── Per-VNF penalty vectors ───────────────────────────────────────────────
    # network_penalty shape: (N_RES, 1) → ravelled to (N_RES,).
    # Formula (simulation.py): (1 + p_loss_rate) * latency, with an optional
    # latency_sla_penalty_factor multiplier when latency < latency_sla.
    # security_penalty shape: (N_RES, 1) → ravelled to (N_RES,).
    # Formula (simulation.py): max_vuln * impact_ssla_factor, where
    # max_vuln = max(asp*cvss) over all attack types; grows with recon counter.
    net_pen = obs_array[_SL_NET_PEN][active]
    sec_pen = obs_array[_SL_SEC_PEN][active]

    # ── MTD resource overhead ─────────────────────────────────────────────────
    # mtd_resource_overhead shape: (N_RES, 1) → ravelled to (N_RES,).
    # Value = coeff_cpu*cpu + coeff_ram*ram/1000 + coeff_disk*disk  (in $).
    # Non-zero only when mtd_action[i][0] != 0 (MTD in progress).
    # This is resource_cost from the OptSFC paper (≈ mtd_cost without the
    # deployment_time factor). Using mean captures average per-VNF spend
    # across the active MTD actions; max would conflate VNF heterogeneity
    # with action intensity.
    mtd_ohd = obs_array[_SL_MTD_OHD][active]

    # ── MTD budget constraints ────────────────────────────────────────────────
    # mtd_constraint shape: (N_RES, 2) → col 0 = remaining migrations,
    #                                      col 1 = remaining reinstantiations.
    # Classified as Network features: the budget limit is the SLA downtime
    # constraint, not a computational resource limit (supervisor Q5).
    # Min remaining mig: the tightest per-VNF migration budget — tells the
    #   agent whether *any* VNF is near its migration limit.
    # Mean remaining mig: overall budget health across all VNFs.
    # Mean remaining reinst: overall reinstantiation health.
    #   (Min remaining reinst retained but excluded: std=0 in current data.)
    mtd_con    = obs_array[_SL_MTD_CON].reshape(_N_RES, 2)[active]
    rem_mig    = mtd_con[:, 0]   # remaining migrations per VNF
    rem_reinst = mtd_con[:, 1]   # remaining reinstantiations per VNF

    # ── Connected UEs ─────────────────────────────────────────────────────────
    # nb_UEs_cnx shape: (N_RES, 1) → ravelled to (N_RES,).
    # Classified as Network: UE count drives latency/throughput simulation
    # (get_random_network_metrics) and provides causal context for why
    # network_penalty is high — the penalty alone cannot distinguish
    # "high load" from "MTD overhead".
    nb_ues = obs_array[_SL_NB_UES][active]

    sec_pen_cumul = float(getattr(env, "security_penalty_cumul", 0.0))

    def _safe_max(arr):  return float(arr.max())  if arr.size > 0 else 0.0
    def _safe_mean(arr): return float(arr.mean()) if arr.size > 0 else 0.0
    def _safe_min(arr):  return float(arr.min())  if arr.size > 0 else 0.0
    def _safe_sum(arr):  return float(arr.sum())  if arr.size > 0 else 0.0

    return {
        # ── Resource features ─────────────────────────────────────────────
        # VIM remaining capacity: direct input to is_action_possible() and
        # set_double_resource_allocation(); determines whether MTD can run.
        "feat_vim0_cpu":               float(vim0_cpu),
        "feat_vim0_ram":               float(vim0_ram),
        "feat_vim1_cpu":               float(vim1_cpu),
        "feat_vim1_ram":               float(vim1_ram),
        # MTD economic overhead: non-zero only during active MTD; captures
        # instantaneous resource spend distinct from VIM capacity headroom.
        "feat_mean_mtd_overhead":      _safe_mean(mtd_ohd),

        # ── Network features ──────────────────────────────────────────────
        # Penalty = (1 + p_loss) * latency; mean gives overall QoS health,
        # max surfaces the worst-performing VNF (different optimal response).
        "feat_mean_network_penalty":   _safe_mean(net_pen),
        "feat_max_network_penalty":    _safe_max(net_pen),
        # Remaining migration budget: SLA-constrained (downtime per migration
        # ~330ms). Min captures the most-constrained VNF; mean gives overall
        # budget health. Both needed: min=0 blocks action even if mean > 0.
        "feat_min_remaining_mig":      _safe_min(rem_mig),
        "feat_mean_remaining_mig":     _safe_mean(rem_mig),
        # Remaining reinstantiation budget: mean only (min excluded: std=0
        # in current simulation data — reinstantiation budget never reaches 0
        # within one episode under current settings).
        "feat_mean_remaining_reinst":  _safe_mean(rem_reinst),
        # Total UEs: upstream driver of network_penalty; needed to distinguish
        # "penalty high due to load" vs "penalty high due to MTD overhead".
        "feat_total_ues":              _safe_sum(nb_ues),

        # ── Security features ─────────────────────────────────────────────
        # Penalty = max(asp*cvss) * impact_ssla_factor; grows with recon
        # counter, resets on MTD. Mean = overall threat level; max = worst VNF.
        "feat_mean_security_penalty":  _safe_mean(sec_pen),
        "feat_max_security_penalty":   _safe_max(sec_pen),

        # ── Retained but excluded from FEAT_META (see docstring) ──────────
        # CVSS raw scores: std=0 (static catalogue), no information value.
        "feat_max_apt_score":          _safe_max(apt_avg),
        "feat_mean_apt_score":         _safe_mean(apt_avg),
        "feat_max_dataleak_score":     _safe_max(dataleak_avg),
        "feat_mean_dataleak_score":    _safe_mean(dataleak_avg),
        "feat_max_dos_score":          _safe_max(dos_avg),
        "feat_mean_dos_score":         _safe_mean(dos_avg),
        # Min remaining reinst: std=0 in current data.
        "feat_min_remaining_reinst":   _safe_min(rem_reinst),
        # Cumulative security penalty: episode-level, not single-step.
        "feat_security_penalty_cumul": sec_pen_cumul,
        # Nb resources: constant across episodes.
        "feat_nb_resources":           float(nb_res),
    }


# ── Algorithm detection ───────────────────────────────────────────────────────

def _detect_algo(model) -> str:
    name = type(model).__name__
    if "Envelope" in name:
        return "Envelope"
    if "EUPG" in name:
        return "EUPG"
    if hasattr(model, "policy") and hasattr(model.policy, "q_net"):
        return "DQN"
    if "A2C" in name:
        return "A2C"
    if "PPO" in name:
        return type(model).__name__
    return "Unknown"


# ── Q-value extraction ────────────────────────────────────────────────────────

def _get_q_values(model, obs_np, weights_arr, algo, env=None):
    """
    Return Q-values and algorithm-specific auxiliary info.

    Returns
    -------
    q_values : np.ndarray
        Shape depends on q_type.
    q_type : str
        "MORL_vec"        (n_actions, 3)   Envelope / EUPG DecomposedQNet
        "EUPG_decomposed" (n_actions, 3)   EUPG proxy fallback
        "Scalar"          (n_actions,)     DQN scalar Q
        "PPO_Q"           (n_actions,)     PPO TD-trained proxy Q
        "A2C_Q"           (n_actions,)     A2C TD-trained proxy Q
    aux : dict
        Algorithm-specific extras.

        DQN additions:
            "all_q_values" : np.ndarray (n_actions,)

        PPO / A2C additions:
            "logit_probs"  : np.ndarray (n_actions,)
            "logits"       : np.ndarray (n_actions,)
    """
    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)

    if algo == "Envelope":
        w_t = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model.q_net(obs_t, w_t).cpu().numpy()
        if q.ndim == 3:
            q = q.squeeze(0)
        return q, "MORL_vec", {}

    elif algo == "EUPG":
        probs = model.get_action_probabilities(
            obs_np,
            getattr(env, "accrued_reward", None)
        )
        q_net = getattr(model, "decomposed_q_net", None)
        if q_net is not None:
            with torch.no_grad():
                q_vec = q_net(obs_t).squeeze(0).cpu().numpy()
            return q_vec.astype(np.float32), "MORL_vec", {"probs": probs}
        else:
            decomposed_critic = getattr(model, "decomposed_critic", None)
            with torch.no_grad():
                v = decomposed_critic(obs_t).squeeze(0).cpu().numpy()
            prob_weight = probs / (probs.mean() + 1e-8)
            q_mat       = np.outer(prob_weight, v)
            return q_mat.astype(np.float32), "EUPG_decomposed", {"probs": probs}

    elif algo == "DQN":
        with torch.no_grad():
            q = model.policy.q_net(obs_t).cpu().numpy().squeeze()
        q = q.astype(np.float32)
        return q, "Scalar", {"all_q_values": q}

    elif algo == "A2C":
        a2c_q_net = getattr(model, "a2c_q_net", None)
        with torch.no_grad():
            q_vals = a2c_q_net(obs_t).squeeze(0).cpu().numpy()
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        logit_probs = _softmax(logits)
        return q_vals.astype(np.float32), "A2C_Q", {
            "q_vals":      q_vals.astype(np.float32),
            "logits":      logits.astype(np.float32),
            "logit_probs": logit_probs.astype(np.float32),
        }

    else:   # PPO
        ppo_q_net = getattr(model, "ppo_q_net", None)
        with torch.no_grad():
            q_vals = ppo_q_net(obs_t).squeeze(0).cpu().numpy()
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        logit_probs = _softmax(logits)
        return q_vals.astype(np.float32), "PPO_Q", {
            "q_vals":      q_vals.astype(np.float32),
            "logits":      logits.astype(np.float32),
            "logit_probs": logit_probs.astype(np.float32),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Determinism classification ────────────────────────────────────────────────

_DETERMINISTIC_ALGOS = {"Envelope", "DQN"}
_STOCHASTIC_ALGOS    = {"EUPG", "PPO", "A2C"}


# ── Action selection ──────────────────────────────────────────────────────────

def _select_actions(q_values, weights_arr, q_type, algo, env_action=None, aux=None):
    if aux is None:
        aux = {}

    if q_type in ("MORL_vec", "EUPG_decomposed"):
        scalar_q = q_values @ weights_arr
    else:
        scalar_q = q_values

    if q_type in ("MORL_vec", "EUPG_decomposed") and algo == "EUPG" and "probs" in aux:
        reference_action = int(np.argmax(aux["probs"]))
    elif algo in _STOCHASTIC_ALGOS and "logits" in aux:
        reference_action = int(np.argmax(aux["logits"]))
    else:
        reference_action = int(np.argmax(scalar_q))

    if env_action is not None:
        best_action = int(env_action)
    else:
        best_action = reference_action

    match = int(best_action == reference_action) if env_action is not None else None

    if reference_action != best_action:
        alt_action = reference_action
    else:
        if q_type in ("MORL_vec", "EUPG_decomposed") and algo == "EUPG" and "probs" in aux:
            sorted_idx = np.argsort(aux["probs"])[::-1]
        elif algo in _STOCHASTIC_ALGOS and "logits" in aux:
            sorted_idx = np.argsort(aux["logits"])[::-1]
        else:
            sorted_idx = np.argsort(scalar_q)[::-1]
        alt_action = int(next(idx for idx in sorted_idx if idx != best_action))

    return best_action, alt_action, scalar_q, reference_action, match


# ── Summary builders ──────────────────────────────────────────────────────────

def _build_morl_summary(best_action, alt_action, reference_action,
                        delta, weights_arr, q_type, algo, match):
    improved = []
    degraded = []
    for i, name in enumerate(OBJ_NAMES):
        if delta[i] > 0:
            improved.append(name)
        else:
            degraded.append(name)

    component_lines = []
    for i, name in enumerate(OBJ_NAMES):
        direction = "gains" if delta[i] > 0 else "loses"
        component_lines.append(
            f"{name}: {direction} {abs(delta[i] * weights_arr[i]):.3f} weighted"
        )
    component_str = "; ".join(component_lines)
    proxy_note    = " [proxy Q]" if q_type == "EUPG_decomposed" else ""
    is_stochastic = algo in _STOCHASTIC_ALGOS
    is_eupg       = algo == "EUPG"
    ref_label     = (
        "highest-probability action" if is_eupg else "highest-Q action"
    )

    if match is None:
        summary = (
            f"Action {best_action} vs action {alt_action}{proxy_note}: "
            f"{component_str}."
        )
    elif match:
        chosen_desc = (
            f"Action {best_action} was executed and matches the "
            f"{ref_label} (action {reference_action})."
        )
        if all(d >= 0 for d in delta):
            verdict = "better on all objectives"
        elif all(d <= 0 for d in delta):
            verdict = "lower Q on all objectives vs next-best"
        else:
            verdict = (
                f"better on {', '.join(improved)}, "
                f"worse on {', '.join(degraded)}"
            )
        summary = (
            f"{chosen_desc} "
            f"Compared to next-best action {alt_action}{proxy_note}: "
            f"{verdict}. Per-component: {component_str}."
        )
    else:
        if all(d >= 0 for d in delta):
            verdict = "better on all objectives"
        elif all(d <= 0 for d in delta):
            verdict = "worse on all objectives"
        else:
            verdict = (
                f"better on {', '.join(improved)}, "
                f"worse on {', '.join(degraded)}"
            )
        reason = (
            f"Action {best_action} was executed via stochastic sampling; "
            f"the {ref_label} (action {reference_action}) was not chosen."
            if is_stochastic else
            f"Action {best_action} was executed; "
            f"the {ref_label} (action {reference_action}) was not chosen "
            f"(possible cause: exploration or policy not converged)."
        )
        summary = (
            f"{reason} "
            f"Relative to the {ref_label}{proxy_note}: "
            f"action {best_action} is {verdict}. "
            f"Per-component: {component_str}."
        )

    return summary, improved, degraded


def _build_scalar_summary(best_action, alt_action, reference_action,
                          delta, algo, q_type, match):
    is_stochastic = algo in _STOCHASTIC_ALGOS

    if q_type == "PPO_Q":
        metric    = "TD-trained Q-value"
        ref_label = f"highest-logit action (action {reference_action})"
        algo_note = (
            "[PPO: Q(s,a) trained via TD on PPO's own transitions. "
            "reference_action = argmax(logits). "
            "prob_action_i columns contain softmax(logits) for SHAP analysis. "
            "Reward decomposition not applicable.]"
        )
    elif q_type == "A2C_Q":
        metric    = "TD-trained Q-value"
        ref_label = f"highest-logit action (action {reference_action})"
        algo_note = (
            "[A2C: Q(s,a) trained via TD on A2C's own transitions. "
            "reference_action = argmax(logits). "
            "prob_action_i columns contain softmax(logits) for SHAP analysis. "
            "Reward decomposition not applicable.]"
        )
    else:   # Scalar DQN
        metric    = "Q-value"
        ref_label = f"highest-Q action (action {reference_action})"
        algo_note = (
            "[DQN: single-objective algorithm. "
            "q_a{i}_scalar columns contain per-action Q values for SHAP. "
            "Reward decomposition not applicable.]"
        )

    delta_str = f"{delta:+.4f}"

    if match is None:
        summary = (
            f"Action {best_action} vs action {alt_action}: "
            f"{metric} difference {delta_str}. {algo_note}"
        )
    elif match:
        chosen_desc = (
            f"Action {best_action} was executed and matches the {ref_label}."
            if is_stochastic else
            f"Action {best_action} was executed and is the {ref_label}."
        )
        summary = (
            f"{chosen_desc} "
            f"{metric} vs next-best action {alt_action}: {delta_str}. "
            f"{algo_note}"
        )
    else:
        reason = (
            f"Action {best_action} was executed via stochastic sampling; "
            f"the {ref_label} was not chosen."
            if is_stochastic else
            f"Action {best_action} was executed; "
            f"the {ref_label} was not chosen "
            f"(possible cause: exploration or policy not converged)."
        )
        summary = (
            f"{reason} "
            f"{metric} of action {best_action} vs {ref_label}: {delta_str}. "
            f"{algo_note}"
        )

    return summary


# ── Main explanation entry point ──────────────────────────────────────────────

def reward_difference_explanation(model, obs, weights=None, env_action=None, env=None):
    weights_arr = np.array(weights if weights is not None else rewards_coeff,
                           dtype=np.float32)
    obs_np  = np.array(obs, dtype=np.float32)
    algo    = _detect_algo(model)

    q_values, q_type, aux = _get_q_values(
        model, obs_np, weights_arr, algo, env=env
    )

    best_action, alt_action, scalar_q, reference_action, match = _select_actions(
        q_values, weights_arr, q_type, algo,
        env_action=env_action, aux=aux
    )

    if q_type in ("MORL_vec", "EUPG_decomposed"):
        best_q = q_values[best_action]
        alt_q  = q_values[alt_action]
        delta  = best_q - alt_q

        summary, improved, degraded = _build_morl_summary(
            best_action, alt_action, reference_action,
            delta, weights_arr, q_type, algo, match
        )

        result = {
            "type":               "MORL_RDX",
            "algo":               algo,
            "q_type":             q_type,
            "env_action":         int(env_action) if env_action is not None
                                  else None,
            "reference_action":   int(reference_action),
            "alternative_action": int(alt_action),
            "match":              match,
            "summary":            summary,
            "best_weighted_q_resource": float(best_q[0] * weights_arr[0]),
            "best_weighted_q_network":  float(best_q[1] * weights_arr[1]),
            "best_weighted_q_security": float(best_q[2] * weights_arr[2]),
            "alt_weighted_q_resource":  float(alt_q[0]  * weights_arr[0]),
            "alt_weighted_q_network":   float(alt_q[1]  * weights_arr[1]),
            "alt_weighted_q_security":  float(alt_q[2]  * weights_arr[2]),
            "weighted_delta_resource":  float(delta[0]  * weights_arr[0]),
            "weighted_delta_network":   float(delta[1]  * weights_arr[1]),
            "weighted_delta_security":  float(delta[2]  * weights_arr[2]),
            "overall_advantage":        float(np.dot(delta, weights_arr)),
            "improved":                 improved,
            "degraded":                 degraded,
        }
        if "probs" in aux:
            result["action_probs"] = [float(p) for p in aux["probs"]]
        return result

    elif q_type == "Scalar":
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action, reference_action,
            delta, algo, q_type, match
        )
        result = {
            "type":               "SingleObjective_RDX",
            "algo":               "DQN",
            "q_type":             q_type,
            "env_action":         env_action,
            "reference_action":   reference_action,
            "alternative_action": alt_action,
            "match":              match,
            "summary":            summary,
            "best_q":             float(scalar_q[best_action]),
            "alt_q":              float(scalar_q[alt_action]),
            "delta":              delta,
        }
        if "all_q_values" in aux:
            result["all_q_values"] = [float(v) for v in aux["all_q_values"]]
        return result

    else:   # PPO / A2C
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action, reference_action,
            delta, algo, q_type, match
        )
        result = {
            "type":               "SingleObjective_PolicyBased",
            "algo":               algo,
            "q_type":             q_type,
            "env_action":         env_action,
            "reference_action":   reference_action,
            "alternative_action": alt_action,
            "match":              match,
            "summary":            summary,
            "best_q":             float(scalar_q[best_action]),
            "alt_q":              float(scalar_q[alt_action]),
            "delta":              delta,
        }
        if "logit_probs" in aux:
            result["action_probs"] = [float(p) for p in aux["logit_probs"]]
        return result


# ── Log entry builder ─────────────────────────────────────────────────────────

def _build_log_entry(step_counter, action, explanation,
                     obs_array=None, env=None, reward_noScalar=None):
    entry = {
        "step":             step_counter,
        "algo":             explanation["algo"],
        "q_type":           explanation.get("q_type", ""),
        "exp_type":         explanation["type"],
        "env_action":       action,
        "reference_action": explanation["reference_action"],
        "alt_action":       explanation["alternative_action"],
        "match":            explanation["match"],
        "summary":          explanation["summary"],
    }

    if reward_noScalar is not None:
        entry["reward_resource"] = float(reward_noScalar[0])
        entry["reward_network"]  = float(reward_noScalar[1])
        entry["reward_security"] = float(reward_noScalar[2])

    exp_type = explanation["type"]

    if exp_type == "MORL_RDX":
        entry.update({
            "weighted_resource_diff":   explanation["weighted_delta_resource"],
            "weighted_network_diff":    explanation["weighted_delta_network"],
            "weighted_security_diff":   explanation["weighted_delta_security"],
            "best_weighted_q_resource": explanation["best_weighted_q_resource"],
            "best_weighted_q_network":  explanation["best_weighted_q_network"],
            "best_weighted_q_security": explanation["best_weighted_q_security"],
            "alt_weighted_q_resource":  explanation["alt_weighted_q_resource"],
            "alt_weighted_q_network":   explanation["alt_weighted_q_network"],
            "alt_weighted_q_security":  explanation["alt_weighted_q_security"],
        })
        if "action_probs" in explanation:
            _write_prob_columns(entry, explanation["action_probs"])

    elif exp_type == "SingleObjective_RDX":
        entry.update({
            "best_q": explanation["best_q"],
            "alt_q":  explanation["alt_q"],
            "delta":  explanation["delta"],
        })
        if "all_q_values" in explanation:
            for i, q_val in enumerate(explanation["all_q_values"]):
                entry[f"q_a{i}_scalar"] = q_val

    elif exp_type == "SingleObjective_PolicyBased":
        entry.update({
            "best_q": explanation["best_q"],
            "alt_q":  explanation["alt_q"],
            "delta":  explanation["delta"],
        })
        if "action_probs" in explanation:
            _write_prob_columns(entry, explanation["action_probs"])

    if obs_array is not None and env is not None:
        state_feats = extract_state_features(obs_array, env)
        entry.update(state_feats)

    return entry


def _write_prob_columns(entry: dict, probs: list) -> None:
    for i, p in enumerate(probs):
        entry[f"prob_action_{i}"] = round(p, 6)
    max_p = max(probs)
    entry["max_prob"]        = round(max_p, 6)
    entry["max_prob_action"] = int(probs.index(max_p))
    entry["policy_entropy"]  = round(
        -sum(p * np.log(p + 1e-8) for p in probs), 6
    )


# ── Full per-action Q table extractor (MORL only) ────────────────────────────

def _get_all_actions_q_columns(model, obs_np: np.ndarray,
                                weights: list) -> dict:
    algo = _detect_algo(model)
    if algo not in ("Envelope", "EUPG"):
        return {}

    weights_arr = np.array(weights, dtype=np.float32)
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

    try:
        if algo == "Envelope":
            w_t = torch.tensor(weights_arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = model.q_net(obs_t, w_t).cpu().numpy()
            if q.ndim == 3:
                q = q.squeeze(0)
        else:
            q_net = getattr(model, "decomposed_q_net", None)
            if q_net is None:
                return {}
            with torch.no_grad():
                q = q_net(obs_t).squeeze(0).cpu().numpy()
    except Exception:
        return {}

    scalar_q = q @ weights_arr
    cols = {}
    for a_idx, (q_vec, sq) in enumerate(zip(q, scalar_q)):
        cols[f"q_a{a_idx}_resource"] = float(q_vec[0])
        cols[f"q_a{a_idx}_network"]  = float(q_vec[1])
        cols[f"q_a{a_idx}_security"] = float(q_vec[2])
        cols[f"scalar_q_a{a_idx}"]   = float(sq)
    return cols