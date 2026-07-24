import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "apg_silver_env_outputs_raw\silver_apg_envelope_env_assignments.csv"

TOP_K = 10

# Ranking criterion:
# "probability" -> APG transition probability P(source -> target)
# "count"       -> number of observed transitions
RANK_BY = "probability"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "step",
    "leaf_id",
    "abstract_state",
    "env_action",
    "silver_class",
    "g_value",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# Sort by trajectory order
df = df.sort_values("step").reset_index(drop=True)


# ============================================================
# Construct consecutive transitions
# ============================================================

df["next_abstract_state"] = df["abstract_state"].shift(-1)
df["next_g_value"] = df["g_value"].shift(-1)
df["next_env_action"] = df["env_action"].shift(-1)

# Last row has no successor and is excluded
transitions = df.iloc[:-1].copy()

transitions = transitions.rename(
    columns={
        "abstract_state": "source_state",
        "next_abstract_state": "target_state",
        "g_value": "source_g",
        "next_g_value": "target_g",
        "env_action": "source_action",
        "next_env_action": "target_action",
    }
)


# ============================================================
# Count transitions
# ============================================================

edge_counts = (
    transitions
    .groupby(["source_state", "target_state"])
    .size()
    .reset_index(name="transition_count")
)


# ============================================================
# Count visits of each source state
# ============================================================

source_visits = (
    transitions
    .groupby("source_state")
    .size()
    .reset_index(name="source_visits")
)


# ============================================================
# Compute APG transition probabilities
# ============================================================

edges = edge_counts.merge(
    source_visits,
    on="source_state",
    how="left"
)

edges["transition_probability"] = (
    edges["transition_count"]
    / edges["source_visits"]
)


# ============================================================
# Additional statistics
# ============================================================

# Mean g-value of the source state
mean_source_g = (
    transitions
    .groupby("source_state")["source_g"]
    .mean()
    .reset_index(name="mean_source_g")
)

# Mean g-value of the target state
mean_target_g = (
    transitions
    .groupby("target_state")["target_g"]
    .mean()
    .reset_index(name="mean_target_g")
)

edges = edges.merge(
    mean_source_g,
    on="source_state",
    how="left"
)

edges = edges.merge(
    mean_target_g,
    on="target_state",
    how="left"
)


# Difference in mean g-value
edges["delta_mean_g"] = (
    edges["mean_target_g"]
    - edges["mean_source_g"]
)


# ============================================================
# Rank edges
# ============================================================

if RANK_BY == "probability":

    top_edges = edges.sort_values(
        by=[
            "transition_probability",
            "transition_count"
        ],
        ascending=[False, False]
    )

elif RANK_BY == "count":

    top_edges = edges.sort_values(
        by=[
            "transition_count",
            "transition_probability"
        ],
        ascending=[False, False]
    )

else:
    raise ValueError(
        "RANK_BY must be either 'probability' or 'count'"
    )


top_edges = top_edges.head(TOP_K).copy()


# ============================================================
# Print result
# ============================================================

print("\n" + "=" * 100)
print(f"TOP {TOP_K} APG EDGES")
print("=" * 100)

print(
    top_edges[
        [
            "source_state",
            "target_state",
            "transition_count",
            "source_visits",
            "transition_probability",
            "mean_source_g",
            "mean_target_g",
            "delta_mean_g",
        ]
    ].to_string(
        index=False,
        formatters={
            "transition_probability": "{:.3f}".format,
            "mean_source_g": "{:.3f}".format,
            "mean_target_g": "{:.3f}".format,
            "delta_mean_g": "{:+.3f}".format,
        }
    )
)


# ============================================================
# Save result
# ============================================================

output_file = Path(INPUT_FILE).with_name(
    f"top_{TOP_K}_apg_edges.csv"
)

top_edges.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)