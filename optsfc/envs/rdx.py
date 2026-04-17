import numpy as np
import torch

rewards_coeff = [0.4, 0.3, 0.3]
OBJ_NAMES     = ["Resource", "Network", "Security"]


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


def _get_q_values(model, obs_np, weights_arr, algo, env=None):
    """
    Return Q-values and algorithm-specific auxiliary info.

    Returns: (q_values, q_type, aux)

    q_type / q_values shape:
      "MORL_vec"        (n_actions, 3)  Envelope true vector Q, or EUPG
                                        DecomposedQNet vector Q
      "EUPG_decomposed" (n_actions, 3)  EUPG proxy vector Q (fallback when
                                        DecomposedQNet is not available)
      "Scalar"          (n_actions,)    DQN scalar Q
      "PPO_Q"           (n_actions,)    PPO TD-trained scalar Q
      "A2C_Q"           (n_actions,)    A2C TD-trained scalar Q

    aux per algorithm:
      Envelope        : {}
      EUPG (QNet)     : {"probs": (n_actions,)}  policy probabilities;
                        reference_action uses argmax(probs) to reflect
                        EUPG's stochastic policy intent
      EUPG (fallback) : {"probs": (n_actions,)}  reference_action uses
                        argmax(probs) to avoid V(s) sign ambiguity
      DQN             : {}
      PPO (QNet)      : {"q_vals": (n_actions,), "logits": (n_actions,)}
                        raw Q values and policy logits; reference_action
                        uses argmax(logits) to reflect PPO's stochastic
                        policy intent
      A2C (A2CQNet)    : {"q_vals": (n_actions,), "logits": (n_actions,)}
                        raw Q values and policy logits; reference_action
                        uses argmax(logits) to reflect A2C's stochastic
                        policy intent
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
            # reference_action = argmax(probs), consistent with EUPG's
            # stochastic policy: the greedy Q target estimates a policy
            # different from EUPG's own, so probs better reflects EUPG's
            # actual decision criterion.
            with torch.no_grad():
                q_vec = q_net(obs_t).squeeze(0).cpu().numpy()  # (n_actions, N_OBJ)
            return q_vec.astype(np.float32), "MORL_vec", {"probs": probs}
        else:
            # Proxy: Q(s,a) = (probs[a] / mean(probs)) * V(s).
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
        # TD-trained Q(s,a) for all actions, using A2C's own transitions.
        # reference_action = argmax(logits), consistent with A2C's stochastic
        # policy intent: the independently trained Q network estimates a greedy
        # policy different from A2C's own stochastic policy, so logits better
        # reflect A2C's actual decision criterion. Q values are retained in aux
        # for delta computation in the explanation.
        with torch.no_grad():
            q_vals = a2c_q_net(obs_t).squeeze(0).cpu().numpy()  # (n_actions,)
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        return q_vals.astype(np.float32), "A2C_Q", {
            "q_vals": q_vals.astype(np.float32),
            "logits": logits.astype(np.float32),
        }

    else:  # PPO 
        ppo_q_net = getattr(model, "ppo_q_net", None)
        # TD-trained Q(s,a) for all actions, using PPO's own transitions.
        # reference_action = argmax(logits), consistent with PPO's stochastic
        # policy intent: the independently trained Q network estimates a greedy
        # policy different from PPO's own stochastic policy, so logits better
        # reflect PPO's actual decision criterion. Q values are retained in aux
        # for delta computation in the explanation.
        with torch.no_grad():
            q_vals = ppo_q_net(obs_t).squeeze(0).cpu().numpy()  # (n_actions,)
            dist   = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.squeeze().cpu().numpy()
        return q_vals.astype(np.float32), "PPO_Q", {
            "q_vals": q_vals.astype(np.float32),
            "logits": logits.astype(np.float32),
        }
        
# Deterministic at inference: mismatch implies exploration or non-convergence.
_DETERMINISTIC_ALGOS = {"Envelope", "DQN"}

# Stochastic at inference: mismatch is expected during normal operation.
_STOCHASTIC_ALGOS = {"EUPG", "PPO", "A2C"}


def _select_actions(q_values, weights_arr, q_type, algo, env_action=None, aux=None):
    """
    Determine best_action, alt_action, reference_action, and match for RDX.

    best_action      = env_action (the action the agent actually executed).
                       Falls back to reference_action when env_action is None.

    reference_action = deterministic optimum under each algorithm's criterion:
      Envelope                 -> argmax(Q_vec @ weights)
      EUPG with DecomposedQNet -> argmax(probs)  [stochastic policy reference;
                                  DecomposedQNet uses a greedy TD target that
                                  estimates a different policy than EUPG's own]
      EUPG fallback            -> argmax(probs)  [avoids V(s) sign ambiguity]
      DQN                      -> argmax(Q_scalar)
      PPO with PPOQNet         -> argmax(logits)  [stochastic policy reference;
                                  PPOQNet uses a greedy TD target that estimates
                                  a different policy than PPO's own stochastic one]

    alt_action:
      reference_action != best_action -> alt = reference_action
        (the greedy/stochastic optimum serves as the natural contrast)
      reference_action == best_action -> alt = second-highest scalar_q
        (agent chose the optimum; contrast against next-best option)

    match = (best_action == reference_action)
      Envelope / DQN : expected ~100% for a converged deterministic policy
      EUPG / PPO     : structurally low; stochastic sampling from the policy
                       distribution vs a deterministic reference guarantees
                       frequent mismatches regardless of policy quality

    Returns: best_action, alt_action, scalar_q, reference_action, match
    """
    if aux is None:
        aux = {}

    if q_type in ("MORL_vec", "EUPG_decomposed"):
        scalar_q = q_values @ weights_arr
    else:
        scalar_q = q_values   # already scalar: DQN, PPO_Q, PPO_advantage

    # Determine reference_action per algorithm semantics.
    if q_type in ("MORL_vec", "EUPG_decomposed") and algo == "EUPG" and "probs" in aux:
        # Both EUPG paths use argmax(probs) as reference, reflecting EUPG's
        # stochastic policy intent. For the DecomposedQNet path, the greedy
        # TD target estimates a policy different from EUPG's own stochastic
        # policy, making probs the more faithful reference criterion.
        reference_action = int(np.argmax(aux["probs"]))
    elif algo in _STOCHASTIC_ALGOS and "logits" in aux:
        # PPO paths use argmax(logits) as reference, reflecting PPO's
        # stochastic policy intent. For the PPOQNet path, the greedy TD target
        # estimates a policy different from PPO's own stochastic policy, making
        # logits the more faithful reference criterion. 
        reference_action = int(np.argmax(aux["logits"]))
    else:
        # Deterministic paths (Envelope, DQN): greedy argmax on scalar_q.
        reference_action = int(np.argmax(scalar_q))

    # best_action: what the agent actually did.
    if env_action is not None:
        best_action = int(env_action)
    else:
        best_action = reference_action

    # match: did the agent's action coincide with the reference?
    match = int(best_action == reference_action) if env_action is not None else None

    # alt_action: the contrast action for the explanation.
    if reference_action != best_action:
        alt_action = reference_action
    else:
        sorted_idx = np.argsort(scalar_q)[::-1]
        alt_action = int(next(idx for idx in sorted_idx if idx != best_action))

    return best_action, alt_action, scalar_q, reference_action, match


def _build_morl_summary(best_action, alt_action, reference_action,
                        delta, weights_arr, q_type, algo, match):
    """
    Build a human-readable RDX summary for MORL algorithms (Envelope / EUPG).

    delta = q_vector[best_action] - q_vector[alt_action], shape (3,).
    Positive delta[i]: best_action is better on objective i.

    match=None  : no env_action; pure pairwise comparison.
    match=True  : agent chose the reference action;
                  alt is the next-best action (by scalar_q).
    match=False : agent did not choose the reference action;
                  alt is the reference action.
                  For EUPG / PPO this is structurally expected: stochastic
                  sampling from the policy distribution will frequently
                  deviate from any fixed deterministic reference, regardless
                  of policy quality.
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

    # For EUPG (both paths), the reference is argmax(probs).
    is_eupg = algo == "EUPG"
    ref_label = (
        "highest-probability action"
        if is_eupg
        else "highest-Q action"
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

    delta = scalar_q[best_action] - scalar_q[alt_action].
    For PPO_Q: delta = q_vals[best] - q_vals[alt] (mean cancels out).
    Positive delta: best_action scores higher on the scalar metric.
    Delta can be negative when match=False (agent chose a lower-value action).
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
    else:  # Scalar (DQN)
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


def reward_difference_explanation(model, obs, weights=None, env_action=None, env=None):
    weights_arr = np.array(weights if weights is not None else rewards_coeff, dtype=np.float32)
    obs_np      = np.array(obs, dtype=np.float32)
    algo        = _detect_algo(model)

    q_values, q_type, aux = _get_q_values(
        model, obs_np, weights_arr, algo, env=env
    )

    best_action, alt_action, scalar_q, reference_action, match = _select_actions(
        q_values, weights_arr, q_type, algo,
        env_action=env_action, aux=aux
    )

    # --- MORL: Envelope / EUPG ---
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

    # --- DQN ---
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

    # --- PPO / A2C ---
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


def _build_log_entry(step_counter, action, explanation):
    """
    Build a flat log entry for CSV export.

    Columns:
      env_action       : action the agent actually executed
      reference_action : stochastic policy reference under each algorithm's criterion
                         (argmax probs for EUPG; argmax logits for PPO both paths;
                          argmax Q for Envelope/DQN)
      alt_action       : contrast action used in the explanation
      match            : whether env_action == reference_action
                         (structurally low for EUPG/PPO due to stochastic sampling)
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
            "best_q":  explanation["best_q"],
            "alt_q":   explanation["alt_q"],
            "delta":   explanation["delta"],
        })

    elif exp_type == "SingleObjective_PolicyBased":
        entry.update({
            "best_q":  explanation["best_q"],
            "alt_q":   explanation["alt_q"],
            "delta":   explanation["delta"],
        })

    return entry
