# Contextual Shards

A mechanistic interpretability pipeline that operationalizes **Shard Theory** — the hypothesis that RL agents consist of multiple contextually-activated decision circuits ("shards") that together produce behavior. This project detects, validates, and characterizes those shards in trained maze-navigation agents.

---

## Background

[Shard Theory](https://www.lesswrong.com/posts/iCfdcxiyr2Kj8m8mT/the-shard-theory-of-human-values) proposes that values/behaviors emerge from a collection of contextual circuits learned during training. Rather than a single monolithic policy, an agent may contain separate shards for "approach cheese when near", "explore corners", etc., each active in its own context.

This project operationalizes four diagnostic criteria:

| Criterion | Description |
|-----------|-------------|
| **D1 — Contextual Encoding** | A layer encodes concept X highly *in* the relevant context, weakly *out of* context |
| **D2 — Causal Behavioral Role** | Suppressing that layer's encoding of X changes behavior |
| **D3 — Shard Separability** | Two candidate shards can be independently suppressed without cross-interference |
| **D4 — Reinforcement Traceability** | Shard strength correlates with how often X was rewarded during training |

---

## Setup

**Requires Python 3.10** (procgen only ships wheels for 3.7–3.10).

```bash
# Create environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies (procgen needs setuptools pinned first)
uv pip install "setuptools==65.5.0"
uv pip install -e ".[dev]"
```

Place the pretrained IMPALA checkpoint at:
```
data/raw/maze_policy.pt
```

You can find the pretrained checkpoint used for this project at (this link)[https://drive.google.com/drive/folders/1atPaaA6Sr6cscBnihB_pdAEeeAp02REM]. We used `model_rand_region_5.pth`

### Verify installation

```bash
uv run pytest          # 102 tests, all green
uv run python -c "import procgen; from src.agent_module import AgentFactory; print('OK')"
```

---

## Project Structure

```
contextual-shards/
├── src/
│   ├── environment_module/     # Env wrappers (ProcgenMazeEnv, MiniGridEnv)
│   ├── agent_module/           # IMPALA policy + ActivationRecorder hooks
│   ├── data_module/            # Concept labelers, HDF5 dataset, RolloutCollector
│   ├── probe_module/           # Linear + MDL probes, context-split evaluator
│   ├── causal_module/          # Activation patcher, shard vectors, causal tracer
│   ├── shard_module/           # ShardDetector, SeparabilityTester, ShardMetrics
│   ├── sae_module/             # Sparse Autoencoder (Bricken 2023)
│   ├── trainer_module/         # PPO trainer scaffold + cheese distributions
│   └── analysis_module/        # Statistical tests + visualizations
│
├── run/
│   ├── conf/                   # Hydra configs (agent, environment, probes, …)
│   └── pipeline/
│       ├── prepare_data/       # Data collection scripts
│       ├── training/           # Agent + SAE training scripts
│       └── analysis/           # Experiment analysis scripts
│
├── tests/                      # Unit + integration tests
├── data/
│   ├── raw/                    # maze_policy.pt checkpoint
│   └── processed/              # Generated HDF5 activation files
└── outputs/
    ├── tables/                 # JSON results
    └── figures/                # Plots
```

All `src/` modules use the **Factory / Registry / Auto-import** pattern: every component registers itself with a `@register_X("name")` decorator and is discoverable via `XFactory("name")`.

---

## Running the Experiments

### Step 0 — Collect Activations

Runs the IMPALA agent through the maze and saves per-step activations and concept labels to HDF5.

```bash
uv run python run/pipeline/prepare_data/collect_activations.py \
    --checkpoint data/raw/maze_policy.pt \
    --output data/processed/activations_impala.h5 \
    --n_steps 50000
```

Recorded layers: all 10 IMPALA blocks (`embedder.block1` … `embedder.fc`).
Recorded concepts: `cheese_presence`, `cheese_proximity`, `cheese_direction`, `corner_proximity`.

---

### Experiment 1 — Shard Detection (D1 + D2)

**Step 1a — Linear Probing (D1)**

Trains linear probes at each layer. High in-context accuracy + low out-of-context accuracy = contextual encoding evidence.

```bash
uv run python run/pipeline/analysis/experiment1_probing.py
# → outputs/tables/experiment1_probing.json
# → outputs/figures/exp1_probe_heatmap.png
```

**Step 1b — Causal Tracing (D2)**

For each layer, measures how much restoring clean activations during a corrupted run recovers cheese-directed behavior.

```bash
uv run python run/pipeline/analysis/experiment1_causal.py
# → outputs/tables/experiment1_causal.json
```

**Step 1c — Integration: Shard Candidates**

Combines D1 + D2 scores. Layers that pass both thresholds are ranked as shard candidates.

```bash
uv run python run/pipeline/analysis/experiment1_integrate.py
# → outputs/tables/experiment1_candidates.json
```

Combined score formula: `combined = probe_specificity × causal_effect`
where `probe_specificity = in_context_acc − out_context_acc`.

---

### Experiment 2 — Shard Separability (D3)

Tests whether suppressing shard S1 affects shard S2's behavior (independence score).

```bash
uv run python run/pipeline/analysis/experiment2_separability.py
# → outputs/tables/experiment2_separability.json
```

Independence score: `I(S1, S2) = 1 − |leakage| / |self_effect|`
Score of 1.0 = fully independent shards. Score of 0.0 = fully entangled.

---

### Experiment 3 — Reinforcement Traceability (D4)

Trains three agents with different cheese distributions, then measures whether shard strength correlates with training distribution. Requires training new agents from scratch.

```bash
# Train the three agents (corner_biased, uniform, anti_corner)
uv run python run/pipeline/training/train_rl_agent.py --distribution corner_biased
uv run python run/pipeline/training/train_rl_agent.py --distribution uniform
uv run python run/pipeline/training/train_rl_agent.py --distribution anti_corner

# Collect activations for each trained agent, then run correlation analysis
uv run python run/pipeline/analysis/experiment3_dist.py
# → outputs/tables/experiment3_correlation.json
```

---

### Experiment 4 — SAE Feature Discovery

Trains a Sparse Autoencoder (Bricken et al. 2023) on FC-layer activations and finds monosemantic features that align with known concepts.

```bash
uv run python run/pipeline/training/train_sae.py \
    --activations data/processed/activations_impala.h5 \
    --layer embedder.fc

uv run python run/pipeline/analysis/experiment4_sae.py
# → outputs/tables/experiment4_features.json
```

SAE architecture: `x → (x − b_pre) → W_enc → ReLU → W_dec` with L1 sparsity + unit-norm decoder columns.

---

### Experiment 5 — Generalization (MiniGrid)

Applies the same pipeline to a MiniGrid agent to test whether the detection framework generalizes across environments.

```bash
uv run python run/pipeline/analysis/experiment5_generalize.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| HDF5 activation storage | Streams >1M frames without loading all into RAM |
| Bonferroni correction on probe p-values | Controls false positives across 10 layers |
| Gram-Schmidt projection for suppression | Removes a shard direction without zeroing the whole layer |
| Spearman (not Pearson) for D4 | Non-parametric; robust to non-linear shard strength scaling |
| MDL probe as cross-check | Codelength measures information content, not just linearity |

---

## Development

```bash
# Run tests
uv run pytest

# Run a specific module's tests
uv run pytest tests/test_probe_module/ -v

# Check code style
uv run ruff check src/
```

### Adding a new concept labeler

```python
# src/data_module/concept_labeler/my_concept.py
from src.data_module.concept_labeler import register_labeler
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler

@register_labeler("my_concept")
class MyConceptLabeler(BaseConceptLabeler):
    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        ...  # return float in [0, 1]
```

Auto-import picks it up automatically — no registration boilerplate needed elsewhere.

### Adding a new environment

```python
# src/environment_module/my_env/my_env.py
from src.environment_module import register_env
from src.environment_module.base_env import BaseEnv

@register_env("my_env")
class MyEnv(BaseEnv):
    ...
```

---

## Citation

If you use this codebase, please cite:

- Turner et al. (2022) — *Parametrically Retargetable Decision-Makers Tend to Pursue Power* (Shard Theory foundations)
- Bricken et al. (2023) — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (SAE)
- Meng et al. (2022) — *Locating and Editing Factual Associations in GPT* (causal tracing methodology)
- Voita & Titov (2020) — *Information-Theoretic Probing with MDL* (MDL probes)
