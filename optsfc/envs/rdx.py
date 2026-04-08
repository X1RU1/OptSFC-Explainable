import numpy as np
import torch
from optsfc.envs.eupg.decomposed_critic import DecomposedCritic

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
    Return Q-value of different RL 
    return (q_values, q_type)
      MORL_vec    → (n_actions, 3)  Decomposed reward (Envelope)
      EUPG_decomposed
      MORL_proxy  → (n_actions, 3)  EUPG proxy
      Scalar      → (n_actions,)    DQN scalarized 
      Scalar_proxy→ (n_actions,)    EUPG fallback
      LogProb     → (n_actions,)    PPO log-prob
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
        return q, "MORL_vec"

    elif algo == "EUPG":
        decomposed_critic = getattr(model, "decomposed_critic", None)

        if decomposed_critic is not None:
            # V(s) + prob -> Q
            probs = model.get_action_probabilities(
                obs_np,
                getattr(env, "accrued_reward", None)
            )
            with torch.no_grad():
                v = decomposed_critic(obs_t).squeeze(0).cpu().numpy()   # (N_OBJ,)

            n_actions = len(probs)
            # Q^c(s,a) ≈ V^c(s) * π(a|s) / mean(π)
            prob_weight = probs / (probs.mean() + 1e-8)  # (n_actions,)
            q_mat = np.outer(prob_weight, v)              # (n_actions, 3)
            return q_mat.astype(np.float32), "EUPG_decomposed"

        else:
            # fallback: action probability proxy
            probs = model.get_action_probabilities(
                obs_np,
                getattr(env, "accrued_reward", None)
            )
            return probs.astype(np.float32), "EUPG_prob"

    elif algo == "DQN":
        with torch.no_grad():
            q = model.policy.q_net(obs_t).cpu().numpy().squeeze()
        return q.astype(np.float32), "Scalar"

    # else:  # PPO / A2C
    #     with torch.no_grad():
    #         dist = model.policy.get_distribution(obs_t)
    #         log_probs = dist.distribution.logits.squeeze().cpu().numpy()
    #     return log_probs.astype(np.float32), "LogProb"
    else:  # PPO / A2C
        with torch.no_grad():
            features = model.policy.extract_features(obs_t)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)

            # V(s)：value network output
            value = model.policy.value_net(latent_vf).item()

            # log π(a|s)：policy logits
            dist      = model.policy.get_distribution(obs_t)
            log_probs = dist.distribution.logits.squeeze().cpu().numpy()

        # A(s,a) ≈ log_prob - mean(log_prob) (centralization)
        # Q(s,a) ≈ V(s) + A(s,a)
        advantage = log_probs - log_probs.mean()
        q_approx  = value + advantage              # (n_actions,)
        return q_approx.astype(np.float32), "PPO_advantage"

def _select_actions(q_values, weights_arr, q_type, algo):
    """
    Choose a* and a' based on different RL policies

    Envelope: scalar_q = q_vectors @ weights  (weighted sum)
    EUPG:     scalar_q = scalarized_values     (weighted in model)
    DQN:      scalar_q = q_values              (scalarized Q)
    PPO:      scalar_q = log_probs             (policy approximation)
    """
    if q_type in ("MORL_vec", "MORL_proxy", "EUPG_decomposed"):
        # Envelope: argmax_a Σ w_c Q_c(s,a)
        scalar_q = q_values @ weights_arr
    else:
        # DQN / EUPG_prob / LogProb scalarized value sorted
        scalar_q = q_values

    best_action = int(np.argmax(scalar_q))
    sorted_idx  = np.argsort(scalar_q)[::-1]
    alt_action  = int(next(
        idx for idx in sorted_idx if idx != best_action
    ))
    return best_action, alt_action, scalar_q

def _build_morl_summary(best_action, alt_action,
                        delta, weights_arr, q_type,
                        env_action, match):
    """
    MORL trade-off summary
    match / mismatch 
    """
    improved        = []
    degraded        = []
    main_reason     = None
    main_cost       = None
    max_pos_contrib = -np.inf
    max_neg_contrib =  np.inf

    for i, name in enumerate(OBJ_NAMES):
        w_diff = delta[i] * weights_arr[i]
        if delta[i] > 0:
            improved.append(name)
            if w_diff > max_pos_contrib:
                max_pos_contrib = w_diff
                main_reason     = name
        else:
            degraded.append(name)
            if w_diff < max_neg_contrib:
                max_neg_contrib = w_diff
                main_cost       = name

    overall    = float(np.dot(delta, weights_arr))
    proxy_note = " [proxy Q-values]" if q_type == "MORL_proxy" else ""

    if main_reason and main_cost:
        core = (
            f"Action {best_action} is preferred over "
            f"action {alt_action} "
            f"primarily due to improvement in {main_reason} "
            f"(+{max_pos_contrib:.3f}), "
            f"despite degradation in {main_cost} "
            f"({max_neg_contrib:.3f}), "
            f"with overall advantage {overall:.3f}"
            f"{proxy_note}"
        )
    elif main_reason and not degraded:
        core = (
            f"Action {best_action} is preferred over "
            f"action {alt_action} "
            f"with improvement across all objectives, "
            f"mainly in {main_reason} "
            f"(overall advantage: {overall:.3f})"
            f"{proxy_note}"
        )
    else:
        core = (
            f"Action {best_action} is preferred over "
            f"action {alt_action} "
            f"with marginal overall advantage {overall:.3f}"
            f"{proxy_note}"
        )

    # match / mismatch 
    if match is None:
        summary = core
    elif match:
        # ✔ match
        summary = f"Action {best_action} was chosen. " + core
    else:
        # ❗ mismatch
        summary = (
            f"The value function prefers action {best_action} "
            f"because " +
            core.replace(
                f"Action {best_action} is preferred over "
                f"action {alt_action} ", ""
            ) +
            f" However, the agent executed action {env_action}."
        )

    return (summary, main_reason, main_cost,
            improved, degraded,
            max_pos_contrib, max_neg_contrib, overall)

def _build_scalar_summary(best_action, alt_action,
                          delta, algo, q_type,
                          env_action, match):
    """
    Single-objective / policy-based / EUPG adaptation summary
    """
    # ── EUPG：MORL but don't have per-action Q, use probability for adaptation ──────
    if q_type == "EUPG_prob":
        metric_name = "action probability"
        note = (
            "[EUPG: multi-objective policy-gradient algorithm. "
            "No per-action Q-vector available. "
            "Action preference derived from policy distribution "
            "π(a | s, r̃). "
            "Full RDX reward decomposition is not applicable; "
            "this is a policy-preference adaptation.]"
        )
    # ── PPO/A2C：Single-objective policy-based ─────────────────────────
    # elif q_type == "LogProb":
    #     metric_name = "log-probability"
    #     note        = (
    #         f"[{algo}: single-objective policy-based algorithm, "
    #         "no Q-decomposition]"
    #     )
    elif q_type == "PPO_advantage":
        metric_name = "advantage-based Q approximation"
        note = (
            "[PPO adaptation: Q(s,a) ≈ V(s) + A(s,a), "
            "where V(s) is from the value network and "
            "A(s,a) is approximated via centralized log-probabilities. "
            "Reward decomposition not applicable.]"
        )
    # ── DQN：Single-objective Q-based, with no reward decomposition ────────────────
    else:
        metric_name = "Q-value"
        note        = (
            "[DQN: single-objective Q-based algorithm, "
            "reward decomposition not applicable]"
        )

    core = (
        f"action {best_action} over action {alt_action} "
        f"with {metric_name} advantage {delta:.4f} {note}"
    )

    if match is None:
        summary = "The policy prefers " + core
    elif match:
        summary = (
            f"Action {best_action} was chosen. "
            "The policy prefers " + core
        )
    else:
        summary = (
            "The policy prefers " + core +
            f" However, the agent executed action {env_action}."
        )
    return summary

def reward_difference_explanation(model, obs, weights=None, top_k=2, env_action=None, env=None):
    weights_arr = np.array(weights if weights is not None else rewards_coeff, dtype=np.float32)
    obs_np = np.array(obs, dtype=np.float32)
    algo   = _detect_algo(model)
    q_values, q_type = _get_q_values(
        model, obs_np, weights_arr, algo, env=env
    )
    best_action, alt_action, scalar_q = _select_actions(
        q_values, weights_arr, q_type, algo
    )
    match = int(env_action == best_action) \
            if env_action is not None else None

    # ── Envelope：MORL RDX ──────────────────────────────
    if q_type in ("MORL_vec", "MORL_proxy", "EUPG_decomposed"):
        best_q = q_values[best_action]
        alt_q  = q_values[alt_action]
        delta  = best_q - alt_q

        (summary, main_reason, main_cost,
         improved, degraded,
         max_pos, max_neg, overall) = _build_morl_summary(
            best_action, alt_action,
            delta, weights_arr, q_type,
            env_action, match
        )
        detail_lines = [
            f"{name}: {'↑' if delta[i] > 0 else '↓'} "
            f"{abs(delta[i]):.3f} "
            f"(weighted: {delta[i] * weights_arr[i]:+.3f})"
            for i, name in enumerate(OBJ_NAMES)
        ]
        return {
            "type":               "MORL_RDX",
            "algo":               algo,
            "q_type":             q_type,
            "best_action":        best_action,
            "alternative_action": alt_action,
            "env_action":         env_action,
            "match":              match,
            "summary":            summary,
            "best_q_vector":      best_q,
            "alt_q_vector":       alt_q,
            "best_weighted_q":    best_q * weights_arr,
            "alt_weighted_q":     alt_q  * weights_arr,
            "delta_vector":       delta,
            "weighted_delta":     delta  * weights_arr,
            "overall_advantage":  overall,
            "main_reason":        main_reason,
            "main_cost":          main_cost,
            "improved":           improved,
            "degraded":           degraded,
            "details":            detail_lines,
        }

    # ── EUPG：MORL but no Q, policy probability adaptation ────────
    elif q_type == "EUPG_prob":
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action,
            delta, algo, q_type,
            env_action, match
        )
        return {
            "type":               "MORL_PolicyAdaptation",
            "algo":               "EUPG",
            "q_type":             q_type,
            "best_action":        best_action,
            "alternative_action": alt_action,
            "env_action":         env_action,
            "match":              match,
            "summary":            summary,
            "best_prob":          float(scalar_q[best_action]),
            "alt_prob":           float(scalar_q[alt_action]),
            "delta_prob":         delta,
            "note": (
                "EUPG is a MORL algorithm but does not maintain "
                "per-action Q-vectors. Policy probability distribution "
                "is used as an adaptation of RDX. "
                "Objective-level reward decomposition is not available."
            ),
        }

    # ── DQN: Single-objective Q-based ───────────────────────────────────
    elif q_type == "Scalar":
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action,
            delta, algo, q_type,
            env_action, match
        )
        return {
            "type":               "SingleObjective_RDX",
            "algo":               "DQN",
            "q_type":             q_type,
            "best_action":        best_action,
            "alternative_action": alt_action,
            "env_action":         env_action,
            "match":              match,
            "summary":            summary,
            "best_q":             float(scalar_q[best_action]),
            "alt_q":              float(scalar_q[alt_action]),
            "delta":              delta,
            "note": (
                "DQN is a single-objective algorithm. "
                "RDX is applied to the scalar Q-value; "
                "reward decomposition is not applicable."
            ),
        }

    # ── PPO / A2C: Single-objective policy-based ───────────────────────
    else:
        delta   = float(scalar_q[best_action] - scalar_q[alt_action])
        summary = _build_scalar_summary(
            best_action, alt_action,
            delta, algo, q_type,
            env_action, match
        )
        return {
            "type":               "SingleObjective_PolicyBased",
            "algo":               algo,
            "q_type":             "LogProb",
            "best_action":        best_action,
            "alternative_action": alt_action,
            "env_action":         env_action,
            "match":              match,
            "summary":            summary,
            "best_logprob":       float(scalar_q[best_action]),
            "alt_logprob":        float(scalar_q[alt_action]),
            "delta_logprob":      delta,
            "note": (
                f"{algo} is a single-objective policy-gradient algorithm. "
                "Log-probability is used as a proxy for action preference. "
                "Q-values and reward decomposition are not available."
            ),
        }

def _build_log_entry(step_counter, action, explanation):
    """explain_log entry"""
    entry = {
        "step":        step_counter,
        "algo":        explanation["algo"],
        "q_type":      explanation.get("q_type", ""),
        "exp_type":    explanation["type"],      
        "env_action":  action,
        "best_action": explanation["best_action"],
        "alt_action":  explanation["alternative_action"],
        "match":       explanation["match"],
        "summary":     explanation["summary"],
    }

    exp_type = explanation["type"]

    # ── Envelope / EUPG_decomposed RDX ────────────────────────────────────
    if exp_type == "MORL_RDX":
        entry.update({
            "weighted_resource_diff":
                explanation["weighted_delta"][0],
            "weighted_network_diff":
                explanation["weighted_delta"][1],
            "weighted_security_diff":
                explanation["weighted_delta"][2],
            "best_weighted_q_resource":
                explanation["best_weighted_q"][0],
            "best_weighted_q_network":
                explanation["best_weighted_q"][1],
            "best_weighted_q_security":
                explanation["best_weighted_q"][2],
            "alt_weighted_q_resource":
                explanation["alt_weighted_q"][0],
            "alt_weighted_q_network":
                explanation["alt_weighted_q"][1],
            "alt_weighted_q_security":
                explanation["alt_weighted_q"][2],
            "overall_advantage":
                explanation["overall_advantage"],
            "main_reason":
                explanation.get("main_reason", ""),
            "main_cost":
                explanation.get("main_cost", ""),
        })

    # ── EUPG Policy Adaptation ───────────────────────────────
    elif exp_type == "MORL_PolicyAdaptation":
        entry.update({
            "best_prob":  explanation["best_prob"],
            "alt_prob":   explanation["alt_prob"],
            "delta_prob": explanation["delta_prob"],
        })

    # ── DQN ──────────────────────────────────────────────────
    elif exp_type == "SingleObjective_RDX":
        entry.update({
            "best_q": explanation["best_q"],
            "alt_q":  explanation["alt_q"],
            "delta":  explanation["delta"],
        })

    # ── PPO / A2C ────────────────────────────────────────────
    elif exp_type == "SingleObjective_PolicyBased":
        entry.update({
            "best_logprob":  explanation["best_logprob"],
            "alt_logprob":   explanation["alt_logprob"],
            "delta_logprob": explanation["delta_logprob"],
        })

    return entry