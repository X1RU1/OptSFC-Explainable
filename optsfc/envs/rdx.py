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
    if "PPO" in name or "A2C" in name:
        return type(model).__name__
    return "Unknown"


def _get_q_values(model, obs_np, weights_arr, algo, env=None):
    """
    Return Q-values and algorithm-specific auxiliary info.

    Returns: (q_values, q_type, aux)

    q_type / q_values:
      "MORL_vec"        (n_actions, 3)  Envelope true vector Q
      "EUPG_decomposed" (n_actions, 3)  EUPG proxy vector Q
      "Scalar"          (n_actions,)    DQN scalar Q
      "PPO_advantage"   (n_actions,)    PPO/A2C advantage approximation

    aux:
      Envelope / DQN : {}
      EUPG           : {"probs": np.ndarray (n_actions,)}
      PPO / A2C      : {"logits": np.ndarray (n_actions,)}
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
        # Use DecomposedQNet if available: true per-action per-objective Q values
        # trained on EUPG's own transitions, no proxy construction needed.
        q_net = getattr(model, "decomposed_q_net", None)
        probs = model.get_action_probabilities(
            obs_np,
            getattr(env, "accrued_reward", None)
        )
        if q_net is not None:
            with torch.no_grad():
                q_vec = q_net(obs_t).squeeze(0).cpu().numpy()  # (n_actions, N_OBJ)
            return q_vec.astype(np.float32), "MORL_vec", {"probs": probs}
        else:
            # Fallback: proxy construction (retains sign of V for interpretability)
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

    else:  # PPO / A2C
        with torch.no_grad():
            features             = model.policy.extract_features(obs_t)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            value                = model.policy.value_net(latent_vf).item()
            dist                 = model.policy.get_distribution(obs_t)
            logits               = dist.distribution.logits.squeeze().cpu().numpy()

        advantage = logits - logits.mean()
        q_approx  = value + advantage
        return q_approx.astype(np.float32), "PPO_advantage", {"logits": logits.astype(np.float32)}


# Algorithms that use deterministic (argmax) action selection at inference.
# These never produce stochastic sampling mismatches.
_DETERMINISTIC_ALGOS = {"Envelope", "DQN"}

# Algorithms that sample stochastically from a distribution at inference.
_STOCHASTIC_ALGOS = {"EUPG", "PPO", "A2C"}


def _select_actions(q_values, weights_arr, q_type, algo, env_action=None, aux=None):
    """
    Unified action selection for RDX.

    best_action      = env_action (what the agent actually did).
                       Falls back to reference_action when env_action is None.

    reference_action = deterministic optimum under each algorithm's criterion:
      Envelope / DQN  -> argmax(scalar_q)   [true Q-based greedy]
      EUPG            -> argmax(probs)       [highest-prob action, consistent
                                              with Categorical(probs).sample()]
      PPO / A2C       -> argmax(logits)      [highest-logit action, consistent
                                              with predict(deterministic=True)]

    alt_action:
      reference_action != best_action -> alt = reference_action
        (natural contrast: what the greedy/high-prob policy would have done)
      reference_action == best_action -> alt = second-highest scalar_q
        (agent chose the optimum; contrast against next-best)

    match = (best_action == reference_action)
      Envelope / DQN : expected ~100% for a converged policy (deterministic)
      EUPG           : True when agent sampled the highest-prob action
      PPO / A2C      : True when agent sampled the highest-logit action

    Returns: best_action, alt_action, scalar_q, reference_action, match
    """
    if aux is None:
        aux = {}

    if q_type in ("MORL_vec", "EUPG_decomposed"):
        scalar_q = q_values @ weights_arr
    else:
        scalar_q = q_values

    # Reference action: deterministic optimum per algorithm semantics.
    # For EUPG after the v_abs fix, argmax(scalar_q) == argmax(probs),
    # so either would give the same result. We use probs directly to be
    # explicit about the semantic.
    if q_type == "EUPG_decomposed" and "probs" in aux:
        reference_action = int(np.argmax(aux["probs"]))
    elif q_type == "PPO_advantage" and "logits" in aux:
        reference_action = int(np.argmax(aux["logits"]))
    else:
        reference_action = int(np.argmax(scalar_q))

    # best_action: what the agent actually executed.
    if env_action is not None:
        best_action = int(env_action)
    else:
        best_action = reference_action

    # match
    match = int(best_action == reference_action) if env_action is not None else None

    # alt_action
    if reference_action != best_action:
        alt_action = reference_action
    else:
        sorted_idx = np.argsort(scalar_q)[::-1]
        alt_action = int(next(idx for idx in sorted_idx if idx != best_action))

    return best_action, alt_action, scalar_q, reference_action, match


def _build_morl_summary(best_action, alt_action, reference_action,
                        delta, weights_arr, q_type, algo, match):
    """
    Build RDX summary for MORL algorithms (Envelope / EUPG).

    delta = q_vector[best_action] - q_vector[alt_action], shape (3,).
    Positive delta[i]: best_action is better on objective i.

    match=True, alt=second-best:
      Agent chose the optimum. Explain gain over next-best.
      Envelope: deterministic, frame as Q-based choice.
      EUPG: stochastic, frame as lucky sample of the highest-prob action.

    match=False, alt=reference:
      Envelope: should rarely happen (policy not converged or exploration).
                Frame as suboptimal action, no stochastic language.
      EUPG/PPO: expected during training. Frame as stochastic sampling
                away from the highest-probability action.
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
    proxy_note = " [proxy Q]" if q_type == "EUPG_decomposed" else ""

    is_stochastic = algo in _STOCHASTIC_ALGOS

    if match is None:
        # No env_action provided: pure comparison mode.
        summary = (
            f"Action {best_action} vs action {alt_action}{proxy_note}: "
            f"{component_str}."
        )

    elif match:
        # Agent chose the deterministic optimum.
        if is_stochastic:
            chosen_desc = (
                f"Action {best_action} was executed and matches the "
                f"highest-probability action."
            )
        else:
            chosen_desc = (
                f"Action {best_action} was executed and is the "
                f"highest-Q action."
            )

        if all(d >= 0 for d in delta):
            verdict = "better on all objectives"
        elif all(d <= 0 for d in delta):
            # With the v_abs fix this should not occur for EUPG.
            # For Envelope it could happen if the true Q vectors are unusual.
            verdict = "lower proxy Q on all objectives vs next-best"
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
        # Agent did NOT choose the deterministic optimum.
        if q_type == "EUPG_decomposed":
            ref_desc = f"highest-probability action (action {reference_action})"
        else:
            ref_desc = f"highest-Q action (action {reference_action})"

        if all(d >= 0 for d in delta):
            verdict = "better on all objectives"
        elif all(d <= 0 for d in delta):
            verdict = "worse on all objectives"
        else:
            verdict = (
                f"better on {', '.join(improved)}, "
                f"worse on {', '.join(degraded)}"
            )

        if is_stochastic:
            reason = (
                f"Action {best_action} was executed via stochastic sampling; "
                f"the {ref_desc} was not chosen."
            )
        else:
            # Envelope / DQN mismatch: suboptimal action was taken.
            # This should be rare for a converged policy.
            reason = (
                f"Action {best_action} was executed; "
                f"the {ref_desc} was not chosen "
                f"(possible cause: exploration or policy not converged)."
            )

        summary = (
            f"{reason} "
            f"Relative to {ref_desc}{proxy_note}: "
            f"action {best_action} is {verdict}. "
            f"Per-component: {component_str}."
        )

    return summary, improved, degraded


def _build_scalar_summary(best_action, alt_action, reference_action,
                          delta, algo, q_type, match):
    """
    Build RDX summary for single-objective (DQN) and policy-gradient (PPO/A2C).

    delta = scalar_q[best_action] - scalar_q[alt_action].
    Positive delta: best_action scores higher on the scalar metric.
    Delta can be negative when match=False.
    """
    is_stochastic = algo in _STOCHASTIC_ALGOS

    if q_type == "PPO_advantage":
        metric    = "advantage-approximated Q"
        ref_label = f"highest-logit action (action {reference_action})"
        algo_note = (
            "[PPO/A2C: Q(s,a) approx V(s) + A(s,a), "
            "A(s,a) = logit(a) - mean(logits). "
            "Reward decomposition not applicable.]"
        )
    else:
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
        if is_stochastic:
            chosen_desc = (
                f"Action {best_action} was executed and matches the {ref_label}."
            )
        else:
            chosen_desc = (
                f"Action {best_action} was executed and is the {ref_label}."
            )
        summary = (
            f"{chosen_desc} "
            f"{metric} vs next-best action {alt_action}: {delta_str}. "
            f"{algo_note}"
        )

    else:
        if is_stochastic:
            reason = (
                f"Action {best_action} was executed via stochastic sampling; "
                f"the {ref_label} was not chosen."
            )
        else:
            reason = (
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

        return {
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
            "best_advantage":     float(scalar_q[best_action]),
            "alt_advantage":      float(scalar_q[alt_action]),
            "delta_advantage":    delta,
        }


def _build_log_entry(step_counter, action, explanation):
    """
    Build a flat log entry for CSV export.

    env_action      : what the agent actually executed
    reference_action: deterministic optimum under the algorithm's criterion
    alt_action      : contrast action used in the explanation
    match           : whether env_action == reference_action
    best_action and overall_advantage are omitted (redundant).
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

    elif exp_type == "SingleObjective_RDX":
        entry.update({
            "best_q":  explanation["best_q"],
            "alt_q":   explanation["alt_q"],
            "delta":   explanation["delta"],
        })

    elif exp_type == "SingleObjective_PolicyBased":
        entry.update({
            "best_advantage":  explanation["best_advantage"],
            "alt_advantage":   explanation["alt_advantage"],
            "delta_advantage": explanation["delta_advantage"],
        })

    return entry