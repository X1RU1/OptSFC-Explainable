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
#
# apt_scores / data_leak_scores / dos_scores / undefined_scores:
#   col 0 = cvss_score_avg   (from update_agent_obs: env[*][i][2])
#   col 1 = cvss_asp_avg     (from update_agent_obs: env[*][i][6])
#
# mtd_constraint:
#   col 0 = remaining migrations
#   col 1 = remaining reinstantiations
# ─────────────────────────────────────────────────────────────────────────────

_N_RES  = 7   # vnfs_size + cnfs_size
_N_VIM  = 2   # vims_size

# Slice constants derived from the layout table above.
_SL_NB_RES    = slice(0,   1)
_SL_VIM_RES   = slice(1,   7)    # reshape → (_N_VIM, 3)
_SL_ID        = slice(7,  14)
_SL_APT       = slice(14, 28)    # reshape → (_N_RES, 2)
_SL_DATALEAK  = slice(28, 42)    # reshape → (_N_RES, 2)
_SL_DOS       = slice(42, 56)    # reshape → (_N_RES, 2)
_SL_UNDEF     = slice(56, 70)    # reshape → (_N_RES, 2)
_SL_MTD_OHD   = slice(70, 77)    # per-resource MTD resource overhead
_SL_NET_PEN   = slice(77, 84)    # per-resource network penalty
_SL_SEC_PEN   = slice(84, 91)    # per-resource security penalty
_SL_NB_UES    = slice(91, 98)    # per-resource connected UEs
_SL_VIM_HOST  = slice(98, 105)
_SL_MTD_ACT   = slice(105, 119)  # reshape → (_N_RES, 2)
_SL_MTD_CON   = slice(119, 133)  # reshape → (_N_RES, 2): [mig, reinst]


def extract_state_features(obs_array: np.ndarray, env) -> dict:
    """
    Derive interpretable scalar state features from a flat observation array
    and the live environment instance.

    Parameters
    ----------
    obs_array : np.ndarray
        Flat observation produced by dict_observation_to_array() *before* the
        current step's action was applied (i.e. obs_before_step).  This is the
        same array fed into the Q-network, so the features are semantically
        aligned with the Q-value differences computed by RDX.
    env : MOfiveG_net
        Live environment instance.  Used only to read lightweight scalar
        attributes that are not captured in the observation array:
          - step_counter        : current global step index
          - last_mtd_step       : step at which the last valid MTD action fired
          - security_penalty_cumul : running sum of per-step security penalties

    Returns
    -------
    dict
        Flat dict of float scalars keyed by "feat_*" prefix.  Every key is
        always present; missing values are filled with 0.0 so the dict can be
        directly merged into a CSV log row without conditional logic.

    Feature groups
    ──────────────
    VIM resource load   : remaining CPU / RAM on core (VIM 0) and edge (VIM 1)
    Security pressure   : per-threat-type max/mean CVSS avg score across active
                          resources; instantaneous and cumulative security penalty
    Network pressure    : mean / max network penalty across active resources
    Resource overhead   : mean MTD-induced resource overhead across active resources
    MTD budget          : minimum and mean remaining migration / reinstantiation
                          budget across active resources
    Temporal pressure   : steps elapsed since the last MTD action fired;
                          total connected UEs (proxy for traffic load)
    """
    nb_res = int(obs_array[_SL_NB_RES][0])
    active = slice(0, max(nb_res, 1))   # guard against nb_res == 0

    # ── VIM resource load ─────────────────────────────────────────────────────
    vim_res = obs_array[_SL_VIM_RES].reshape(_N_VIM, 3)
    # VIM index 0 = core (VIM 1 in the network config)
    # VIM index 1 = edge (VIM 2 in the network config)
    vim0_cpu, vim0_ram, _vim0_disk = vim_res[0]
    vim1_cpu, vim1_ram, _vim1_disk = vim_res[1]

    # ── Per-resource threat scores (col 0 = avg CVSS score) ──────────────────
    apt_avg       = obs_array[_SL_APT     ].reshape(_N_RES, 2)[active, 0]
    dataleak_avg  = obs_array[_SL_DATALEAK].reshape(_N_RES, 2)[active, 0]
    dos_avg       = obs_array[_SL_DOS     ].reshape(_N_RES, 2)[active, 0]

    # ── Penalty signals ───────────────────────────────────────────────────────
    net_pen = obs_array[_SL_NET_PEN][active]
    sec_pen = obs_array[_SL_SEC_PEN][active]

    # ── MTD overhead and budget ───────────────────────────────────────────────
    mtd_ohd    = obs_array[_SL_MTD_OHD][active]
    mtd_con    = obs_array[_SL_MTD_CON].reshape(_N_RES, 2)[active]
    rem_mig    = mtd_con[:, 0]   # remaining migrations per resource
    rem_reinst = mtd_con[:, 1]   # remaining reinstantiations per resource

    # ── UE connectivity ───────────────────────────────────────────────────────
    nb_ues = obs_array[_SL_NB_UES][active]

    # ── Temporal / episodic trackers (read from env, not obs) ─────────────────
    # last_mtd_step and security_penalty_cumul are injected into the env by
    # the step() hook added in mo_fiveg_mdp.py; fall back to safe defaults so
    # this function also works when called outside that context.
    steps_since_mtd   = float(env.step_counter - getattr(env, "last_mtd_step", 0))
    sec_pen_cumul     = float(getattr(env, "security_penalty_cumul", 0.0))

    def _safe_max(arr):  return float(arr.max())  if arr.size > 0 else 0.0
    def _safe_mean(arr): return float(arr.mean()) if arr.size > 0 else 0.0
    def _safe_min(arr):  return float(arr.min())  if arr.size > 0 else 0.0
    def _safe_sum(arr):  return float(arr.sum())  if arr.size > 0 else 0.0

    return {
        # VIM resource load
        "feat_vim0_cpu":               float(vim0_cpu),
        "feat_vim0_ram":               float(vim0_ram),
        "feat_vim1_cpu":               float(vim1_cpu),
        "feat_vim1_ram":               float(vim1_ram),
        # Security pressure — threat scores
        "feat_max_apt_score":          _safe_max(apt_avg),
        "feat_mean_apt_score":         _safe_mean(apt_avg),
        "feat_max_dataleak_score":     _safe_max(dataleak_avg),
        "feat_mean_dataleak_score":    _safe_mean(dataleak_avg),
        "feat_max_dos_score":          _safe_max(dos_avg),
        "feat_mean_dos_score":         _safe_mean(dos_avg),
        # Security pressure — penalty signals
        "feat_mean_security_penalty":  _safe_mean(sec_pen),
        "feat_max_security_penalty":   _safe_max(sec_pen),
        "feat_security_penalty_cumul": sec_pen_cumul,
        # Network pressure
        "feat_mean_network_penalty":   _safe_mean(net_pen),
        "feat_max_network_penalty":    _safe_max(net_pen),
        # Resource overhead from MTD operations
        "feat_mean_mtd_overhead":      _safe_mean(mtd_ohd),
        # MTD budget remaining
        "feat_min_remaining_mig":      _safe_min(rem_mig),
        "feat_mean_remaining_mig":     _safe_mean(rem_mig),
        "feat_min_remaining_reinst":   _safe_min(rem_reinst),
        "feat_mean_remaining_reinst":  _safe_mean(rem_reinst),
        # Temporal pressure
        "feat_steps_since_last_mtd":   steps_since_mtd,
        "feat_total_ues":              _safe_sum(nb_ues),
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
        Shape and semantics depend on q_type (see below).
    q_type : str
        "MORL_vec"        (n_actions, 3)   Envelope true vector Q, or EUPG
                                           DecomposedQNet vector Q
        "EUPG_decomposed" (n_actions, 3)   EUPG proxy vector Q (fallback when
                                           DecomposedQNet is unavailable)
        "Scalar"          (n_actions,)     DQN scalar Q
        "PPO_Q"           (n_actions,)     PPO TD-trained scalar Q
        "A2C_Q"           (n_actions,)     A2C TD-trained scalar Q
    aux : dict
        Algorithm-specific extras:
          Envelope        : {}
          EUPG (QNet)     : {"probs": (n_actions,)}  policy probabilities;
                            reference_action = argmax(probs) to reflect
                            EUPG's stochastic policy intent
          EUPG (fallback) : {"probs": (n_actions,)}  reference_action uses
                            argmax(probs) to avoid V(s) sign ambiguity
          DQN             : {}
          PPO (QNet)      : {"q_vals": ..., "logits": ...}
                            reference_action = argmax(logits) to reflect PPO's
                            stochastic policy intent
          A2C (A2CQNet)   : {"q_vals": ..., "logits": ...}
                            reference_action = argmax(logits) to reflect A2C's
                            stochastic policy intent
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
            # Per-action per-objective Q trained on EUPG's own transitions.
            # reference_action = argmax(probs), consistent with EUPG's stochastic
            # policy: the greedy Q target estimates a policy different from
            # EUPG's own, so probs better reflects EUPG's actual decision
            # criterion.
            with torch.no_grad():
                q_vec = q_net(obs_t).squeeze(0).cpu().numpy()  # (n_actions, N_OBJ)
            return q_vec.astype(np.float32), "MORL_vec", {"probs": probs}
        else:
            # Proxy Q: Q(s,a) = (probs[a] / mean(probs)) * V(s).
            # V(s) has no action-dependent structure; per-component delta
            # direction is determined by the sign of V(s) components.
            # reference_action = argmax(probs) to avoid sign ambiguity from V(s).
            decomposed_critic = getattr(model, "decomposed_critic", None)
            with torch.no_grad():
                v = decomposed_critic(obs_t).squeeze(0).cpu().numpy()
            prob_weight = probs / (probs.mean() + 1e-8)
            q_mat       = np.outer(prob_weight, v)
            return q_mat.astype(np.float32), "EUPG_decomposed", {"probs": probs}

    elif algo == "DQN":
        with torch.no_grad():
            q = model.policy.q_net(obs_t).cpu().numpy().squeeze()
        return q.astype(np.float32), "Scalar", {}

    elif algo == "A2C":
        a2c_q_net = getattr(model, "a2c_q_net", None)
        # TD-trained Q(s,a) for all actions using A2C's own transitions.
        # reference_action = argmax(logits), consistent with A2C's stochastic
        # policy intent: the independently trained Q network estimates a greedy
        # policy different from A2C's own stochastic policy, so logits better
        # reflect A2C's actual decision criterion.  Q values are retained in
        # aux for delta computation in the explanation.
        with torch.no_grad():
            q_vals = a2c_q_net(obs_t).squeeze(0).cpu().numpy()   # (n_actions,)
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        return q_vals.astype(np.float32), "A2C_Q", {
            "q_vals": q_vals.astype(np.float32),
            "logits": logits.astype(np.float32),
        }

    else:   # PPO
        ppo_q_net = getattr(model, "ppo_q_net", None)
        # TD-trained Q(s,a) for all actions using PPO's own transitions.
        # reference_action = argmax(logits), consistent with PPO's stochastic
        # policy intent: the independently trained Q network estimates a greedy
        # policy different from PPO's own stochastic policy, so logits better
        # reflect PPO's actual decision criterion.  Q values are retained in
        # aux for delta computation in the explanation.
        with torch.no_grad():
            q_vals = ppo_q_net(obs_t).squeeze(0).cpu().numpy()   # (n_actions,)
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        return q_vals.astype(np.float32), "PPO_Q", {
            "q_vals": q_vals.astype(np.float32),
            "logits": logits.astype(np.float32),
        }


# ── Determinism classification ────────────────────────────────────────────────
# Deterministic at inference: mismatch implies exploration or non-convergence.
_DETERMINISTIC_ALGOS = {"Envelope", "DQN"}

# Stochastic at inference: mismatch is expected during normal operation.
_STOCHASTIC_ALGOS = {"EUPG", "PPO", "A2C"}


# ── Action selection ──────────────────────────────────────────────────────────

def _select_actions(q_values, weights_arr, q_type, algo, env_action=None, aux=None):
    """
    Determine best_action, alt_action, reference_action, and match for RDX.

    best_action
        The action the agent actually executed (env_action).  Falls back to
        reference_action when env_action is None (pure pairwise mode).

    reference_action
        Deterministic optimum under each algorithm's criterion:
          Envelope                  → argmax(Q_vec @ weights)
          EUPG (DecomposedQNet)     → argmax(probs)  [stochastic policy reference;
                                       DecomposedQNet uses a greedy TD target that
                                       estimates a different policy than EUPG's own]
          EUPG (fallback proxy)     → argmax(probs)  [avoids V(s) sign ambiguity]
          DQN                       → argmax(Q_scalar)
          PPO / A2C (with QNet)     → argmax(logits) [stochastic policy reference;
                                       QNet uses a greedy TD target that estimates
                                       a different policy than the stochastic one]

    alt_action
        reference_action != best_action → alt = reference_action
            (the greedy/stochastic optimum serves as the natural contrast)
        reference_action == best_action → alt = second-highest scalar_q
            (agent chose the optimum; contrast against the next-best option)

    match
        bool: best_action == reference_action.
        Envelope / DQN  : expected ~100% for a converged deterministic policy.
        EUPG / PPO / A2C: structurally low; stochastic sampling from the policy
                          distribution vs a deterministic reference guarantees
                          frequent mismatches regardless of policy quality.

    Returns
    -------
    best_action, alt_action, scalar_q, reference_action, match
    """
    if aux is None:
        aux = {}

    if q_type in ("MORL_vec", "EUPG_decomposed"):
        scalar_q = q_values @ weights_arr
    else:
        scalar_q = q_values   # already scalar for DQN, PPO_Q, A2C_Q

    # Determine reference_action per algorithm semantics.
    if q_type in ("MORL_vec", "EUPG_decomposed") and algo == "EUPG" and "probs" in aux:
        # Both EUPG paths use argmax(probs) as reference, reflecting EUPG's
        # stochastic policy intent.  For the DecomposedQNet path the greedy TD
        # target estimates a policy different from EUPG's own stochastic policy,
        # making probs the more faithful reference criterion.
        reference_action = int(np.argmax(aux["probs"]))
    elif algo in _STOCHASTIC_ALGOS and "logits" in aux:
        # PPO and A2C use argmax(logits) as reference, reflecting their
        # stochastic policy intent.  The independently trained QNet estimates a
        # greedy policy different from the stochastic policy, so logits are the
        # more faithful reference criterion.
        reference_action = int(np.argmax(aux["logits"]))
    else:
        # Deterministic paths (Envelope, DQN): greedy argmax on scalar_q.
        reference_action = int(np.argmax(scalar_q))

    # best_action: what the agent actually executed.
    if env_action is not None:
        best_action = int(env_action)
    else:
        best_action = reference_action

    # match: did the agent's executed action coincide with the reference?
    match = int(best_action == reference_action) if env_action is not None else None

    # alt_action: the contrast action for the explanation.
    if reference_action != best_action:
        alt_action = reference_action
    else:
        sorted_idx = np.argsort(scalar_q)[::-1]
        alt_action = int(next(idx for idx in sorted_idx if idx != best_action))

    return best_action, alt_action, scalar_q, reference_action, match


# ── Summary builders ──────────────────────────────────────────────────────────

def _build_morl_summary(best_action, alt_action, reference_action,
                        delta, weights_arr, q_type, algo, match):
    """
    Build a human-readable RDX summary for MORL algorithms (Envelope / EUPG).

    Parameters
    ----------
    delta : np.ndarray, shape (3,)
        q_vector[best_action] - q_vector[alt_action].
        Positive delta[i]: best_action gains on objective i.

    match semantics
    ───────────────
    None  : no env_action; pure pairwise comparison.
    True  : agent chose the reference action; alt is the next-best action.
    False : agent did not choose the reference action; alt is the reference.
            For EUPG / A2C / PPO this is structurally expected: stochastic
            sampling frequently deviates from any deterministic reference.
            For Envelope / DQN this implies non-convergence or exploration.
    """
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
    """
    Build a human-readable RDX summary for single-objective algorithms.

    Parameters
    ----------
    delta : float
        scalar_q[best_action] - scalar_q[alt_action].
        For PPO_Q / A2C_Q: delta = q_vals[best] - q_vals[alt]
        (mean cancels out).  Positive delta: best_action scores higher on the
        scalar metric.  Delta can be negative when match=False (agent chose a
        lower-value action via stochastic sampling).
    """
    is_stochastic = algo in _STOCHASTIC_ALGOS

    if q_type == "PPO_Q":
        metric    = "TD-trained Q-value"
        ref_label = f"highest-logit action (action {reference_action})"
        algo_note = (
            "[PPO: Q(s,a) trained via TD on PPO's own transitions. "
            "reference_action = argmax(logits) to reflect PPO's stochastic "
            "policy intent; PPOQNet's greedy TD target estimates a policy "
            "different from PPO's own. Match rate is structurally low due to "
            "stochastic policy sampling. "
            "Reward decomposition not applicable.]"
        )
    elif q_type == "A2C_Q":
        metric    = "TD-trained Q-value"
        ref_label = f"highest-logit action (action {reference_action})"
        algo_note = (
            "[A2C: Q(s,a) trained via TD on A2C's own transitions. "
            "reference_action = argmax(logits) to reflect A2C's stochastic "
            "policy intent; A2CQNet's greedy TD target estimates a policy "
            "different from A2C's own. Match rate is structurally low due to "
            "stochastic policy sampling. "
            "Reward decomposition not applicable.]"
        )
    else:   # Scalar (DQN)
        metric    = "Q-value"
        ref_label = f"highest-Q action (action {reference_action})"
        algo_note = (
            "[DQN: single-objective algorithm. "
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
    """
    Compute a Reward Difference Explanation (RDX) for a single transition.

    Parameters
    ----------
    model      : trained RL model (Envelope, EUPG, DQN, PPO, or A2C)
    obs        : flat observation array (obs_before_step)
    weights    : objective weights; defaults to module-level rewards_coeff
    env_action : action the agent actually executed (None = pairwise mode)
    env        : live MOfiveG_net instance; required for EUPG accrued reward

    Returns
    -------
    dict with explanation fields; structure varies by algorithm type
    (MORL_RDX, SingleObjective_RDX, or SingleObjective_PolicyBased).
    """
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

    # ── MORL: Envelope / EUPG ─────────────────────────────────────────────────
    if q_type in ("MORL_vec", "EUPG_decomposed"):
        best_q = q_values[best_action]
        alt_q  = q_values[alt_action]
        delta  = best_q - alt_q

        summary, improved, degraded = _build_morl_summary(
            best_action, alt_action, reference_action,
            delta, weights_arr, q_type, algo, match
        )

        detail_lines = [
            f"{name}: {'up' if delta[i] > 0 else 'down'} "
            f"{abs(delta[i]):.3f} raw "
            f"(weighted: {delta[i] * weights_arr[i]:+.3f})"
            for i, name in enumerate(OBJ_NAMES)
        ]

        result = {
            "type":               "MORL_RDX",
            "algo":               algo,
            "q_type":             q_type,
            "env_action":         env_action,
            "reference_action":   reference_action,
            "alternative_action": alt_action,
            "match":              match,
            "summary":            summary,
            "best_q_vector":      best_q,
            "alt_q_vector":       alt_q,
            "best_weighted_q":    best_q * weights_arr,
            "alt_weighted_q":     alt_q  * weights_arr,
            "delta_vector":       delta,
            "weighted_delta":     delta  * weights_arr,
            "overall_advantage":  float(np.dot(delta, weights_arr)),
            "improved":           improved,
            "degraded":           degraded,
            "details":            detail_lines,
        }
        if "probs" in aux:
            result["action_probs"] = aux["probs"]
        return result

    # ── DQN ───────────────────────────────────────────────────────────────────
    elif q_type == "Scalar":
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action, reference_action,
            delta, algo, q_type, match
        )
        return {
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

    # ── PPO / A2C ─────────────────────────────────────────────────────────────
    else:
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action, reference_action,
            delta, algo, q_type, match
        )
        return {
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


# ── Log entry builder ─────────────────────────────────────────────────────────

def _build_log_entry(step_counter, action, explanation,
                     obs_array=None, env=None):
    """
    Build a flat dict suitable for CSV export via pd.DataFrame.

    Parameters
    ----------
    step_counter : int
        Current global step index from the environment.
    action : int
        Action the agent executed this step (env_action).
    explanation : dict
        Output of reward_difference_explanation().
    obs_array : np.ndarray or None
        Flat observation array captured *before* the action was applied
        (obs_before_step).  When provided together with env, state features
        are extracted via extract_state_features() and appended as "feat_*"
        columns.  Pass None to omit state features (e.g. for scalar algorithms
        or when replaying old logs without retraining).
    env : MOfiveG_net or None
        Live environment instance needed by extract_state_features() to read
        temporal trackers (last_mtd_step, security_penalty_cumul).
        Must be provided together with obs_array; ignored if obs_array is None.

    Column groups in the returned dict
    ───────────────────────────────────
    Metadata       : step, algo, q_type, exp_type
    Actions        : env_action, reference_action, alt_action, match
    Summary        : summary (human-readable text)
    MORL Q-diffs   : weighted_{resource,network,security}_diff,
                     best_weighted_q_{resource,network,security},
                     alt_weighted_q_{resource,network,security}
    EUPG policy    : prob_action_<i>, max_prob, max_prob_action, policy_entropy
    Scalar Q       : best_q, alt_q, delta          (DQN / PPO / A2C)
    State features : feat_*  (only when obs_array and env are both provided)
    """
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

    exp_type = explanation["type"]

    if exp_type == "MORL_RDX":
        entry.update({
            "weighted_resource_diff":   explanation["weighted_delta"][0],
            "weighted_network_diff":    explanation["weighted_delta"][1],
            "weighted_security_diff":   explanation["weighted_delta"][2],
            "best_weighted_q_resource": explanation["best_weighted_q"][0],
            "best_weighted_q_network":  explanation["best_weighted_q"][1],
            "best_weighted_q_security": explanation["best_weighted_q"][2],
            "alt_weighted_q_resource":  explanation["alt_weighted_q"][0],
            "alt_weighted_q_network":   explanation["alt_weighted_q"][1],
            "alt_weighted_q_security":  explanation["alt_weighted_q"][2],
        })
        # EUPG-specific policy distribution fields.
        if "action_probs" in explanation:
            probs = explanation["action_probs"]
            for i, p in enumerate(probs):
                entry[f"prob_action_{i}"] = round(float(p), 6)
            entry["max_prob"]        = round(float(probs.max()), 6)
            entry["max_prob_action"] = int(probs.argmax())
            entry["policy_entropy"]  = round(
                float(-np.sum(probs * np.log(probs + 1e-8))), 6
            )

    elif exp_type == "SingleObjective_RDX":
        entry.update({
            "best_q": explanation["best_q"],
            "alt_q":  explanation["alt_q"],
            "delta":  explanation["delta"],
        })

    elif exp_type == "SingleObjective_PolicyBased":
        entry.update({
            "best_q": explanation["best_q"],
            "alt_q":  explanation["alt_q"],
            "delta":  explanation["delta"],
        })

    # ── State features (MORL only; optional for scalar algorithms) ────────────
    # Appended as "feat_*" columns so that the CSV remains backward-compatible:
    # rows produced without obs_array / env simply lack these columns, which
    # pandas handles gracefully as NaN when DataFrames are concatenated.
    if obs_array is not None and env is not None:
        state_feats = extract_state_features(obs_array, env)
        entry.update(state_feats)

    return entry