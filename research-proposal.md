# Research Proposal: Toward an Empirically Testable Definition of Neural Shards

**Project Title:** Contextual Shards: Operationalizing Shard Theory Through Mechanistic Interpretability of Reinforcement Learning Agents

**Date:** March 2026
**Status:** Draft

---

## 1. Motivation and Problem Statement

Shard Theory [Turner & Pope, 2022] proposes that RL agents are best understood as collections of *contextually activated decision influences* — shards — each formed by reinforcement history and selectively activated by environmental context. The theory offers a compelling alternative to monolithic goal-based models of agent behavior and has direct implications for AI alignment.

Despite its conceptual appeal, Shard Theory remains an informal framework: a shard is defined functionally but not structurally. There is no operationalized definition of a shard, no validated detection method, and no established criteria for what evidence would confirm or falsify the theory.

Prior exploratory work [Aristizabal, 2024] found evidence consistent with shard-like structure in a cheese-seeking maze agent: spatial locality of cheese-seeking behavior, layer-wise encoding differences, and a single neuron candidate for cheese-presence encoding. However, these are correlational observations that do not establish the causal, separable nature that characterizes a shard. Critically, that prior work also identified a key methodological challenge: the maze environment itself induces a strong **navigation shard** (a persistent corner-directed preference shaped by the training distribution of cheese placement). This navigation shard dominates activation patterns in the full maze, making the cheese shard's signal difficult to isolate. Probing and causal tracing experiments conducted naively in the full maze risk attributing navigation-shard variance to the cheese shard, or failing to detect the cheese shard at all due to signal dilution.

This leads to a two-tier methodological requirement:

1. **Controlled setting** (open arena, no walls): Provides clean, low-interference conditions for shard discovery. The navigation shard is minimized, isolating the cheese shard's encoding and causal role.
2. **Realistic setting** (full maze): Tests whether the shards identified in the controlled setting persist and remain behaviorally relevant in the complex environment the agent was actually trained in.

**The core research questions are:**
> Can we define a shard in terms of specific, measurable properties of a neural network, and design experiments that can confirm or falsify the existence of such structures?
> Do shards identified under controlled conditions retain their encoding and causal role in realistic environments, despite competition from other shards?

---

## 2. Proposed Definition of a Shard (Hypothesis)

We propose that a **shard** is a computational sub-structure in a trained neural network that satisfies all four of the following criteria:

### D1. Contextual Encoding
A shard encodes a concept (e.g., cheese presence, corner proximity) that is detectable from its associated sub-network's activations specifically when the relevant context is active, and not (or weakly) when the context is absent.
- *Operationalization*: A linear probe trained on activations when context C is present achieves accuracy significantly above chance; the same probe on context-absent activations achieves near-chance accuracy. Formally: P(concept | context present) >> P(concept | context absent).

### D2. Causal Behavioral Role
The shard's activations are causally responsible for the behavior associated with that concept.
- *Operationalization*: Suppressing/activating the shard's representation via activation patching produces predictable changes in behavior. A "shard suppression" intervention that reduces the concept's encoding should reduce concept-directed behavior; amplifying it should increase it.

### D3. Separability
A shard is quasi-independent from other shards: suppressing one shard's representation does not substantially affect other shards' encoding or behavior.
- *Operationalization*: After applying a shard-S suppression intervention, probes for all other identified shards retain their predictive accuracy (within ε tolerance), and behaviors governed by other shards remain intact.

### D4. Reinforcement Traceability
A shard's formation is traceable to specific reinforcement events during training. Shards formed from more frequent/stronger reinforcement signals should be stronger (higher causal effect).
- *Operationalization*: Varying the training distribution (e.g., frequency of cheese placement at different positions) should produce corresponding variation in shard strength metrics.

---

## 3. Research Objectives

1. **O1**: Design and validate a shard detection pipeline based on D1–D4.
2. **O2**: Apply the pipeline to the existing cheese-maze RL agent under controlled (open arena) conditions to identify candidate shards and test D1–D2 with minimal confounds.
3. **O3**: Validate whether controlled-setting findings transfer to the full maze (realistic setting), and quantify navigation-shard interference.
4. **O4**: Test the multi-shard interaction hypothesis (separability, D3) in both controlled and realistic settings.
5. **O5**: Train new maze agents with controlled reinforcement distributions to test D4 (reinforcement traceability), using open arena for evaluation.
6. **O6**: Test whether an unsupervised decomposition (SAE) recovers the same shards as supervised probing.
7. **O7**: Assess the generalizability of the detection pipeline to a second RL environment.

---

## 4. Experimental Design

### Methodological Note: Two-Tier Experimental Structure

All experiments involving direct shard detection (D1, D2, D3) follow a two-tier structure:

- **Tier A — Controlled discovery (open arena)**: The agent is placed in an empty grid with no walls. Cheese is placed at controlled positions. This eliminates wall-following behavior and minimizes navigation shard activation, allowing the cheese shard's encoding to be detected cleanly. Tier A experiments establish *what* shards exist and *where* they are encoded.

- **Tier B — Realistic validation (full maze)**: The same detection pipeline is applied to the agent in the original maze environment. Given navigation shard dominance, we expect weaker but non-zero cheese shard signals at the same layers identified in Tier A. Tier B experiments test whether identified shards survive in realistic conditions and whether their causal role is preserved despite competing influences.

This structure mirrors controlled-variable methodology in empirical science: Tier A isolates the variable of interest; Tier B tests external validity.

---

### Experiment 1: Controlled Shard Discovery (Open Arena)

**Goal**: Establish which network locations satisfy D1 and D2 for the cheese shard under clean, low-interference conditions.

**Setup**: Place the agent in an open grid (no walls, no corners). Vary cheese position systematically across the open space. Define three conditions: (a) cheese present near agent (<5 steps), (b) cheese present far (>10 steps), (c) no cheese.

**Step 1 — Contextual probing (D1)**:
- For each layer, train linear probes on activations to predict: (i) cheese presence, (ii) cheese proximity (distance), (iii) direction-to-cheese.
- Compare probe accuracy across conditions (a) vs. (c). A shard candidate is a layer/direction where accuracy is substantially higher in condition (a) than (c).
- Use MDL probing or control tasks [Hewitt & Liang, 2019] to control for probe complexity.

**Step 2 — Causal tracing (D2)**:
- Run the network with cheese present (clean run) and with cheese removed from the same arena state (corrupted run; minimal intervention, preserving all other state).
- Restore activations layer-by-layer and measure the change in cheese-directed action probability.
- Identify layers where restoring clean activations most recovers cheese-directed behavior.

**Step 3 — Cross-validation**:
- Verify that probe-detected layers (D1) and causal-tracing layers (D2) overlap. Primary shard candidates satisfy both.

**Expected outputs**: A ranked list of candidate shard locations (layer, direction) for the cheese shard, identified in low-interference conditions.

**Success criterion**: At least one location identified satisfying both D1 and D2 with statistical significance (p < 0.01 after Bonferroni correction for layer comparisons).

---

### Experiment 2: Realistic Shard Validation (Full Maze)

**Goal**: Test whether the cheese shard identified in Experiment 1 retains detectable encoding and causal role in the full maze, and characterize how much the navigation shard attenuates the signal.

**Setup**: Use the same agent in the original maze environment. Cheese removal uses the same minimal intervention from `get_obs_no_cheese()` (preserves maze layout and agent position; removes only the cheese cell).

**Step 1 — Probing in full maze**:
- Apply the same probing protocol from Experiment 1 to the full maze.
- Focus analysis on the layers identified as shard candidates in Experiment 1.
- Compute the reduction in probe accuracy relative to Experiment 1 as a measure of navigation shard interference.

**Step 2 — Causal tracing in full maze**:
- Run the causal tracing protocol (same as Experiment 1, Step 2) in the full maze.
- Compare causal effect sizes to Experiment 1 at the same layers.
- Hypothesis: the same layers should show non-zero causal effects, but smaller in magnitude.

**Step 3 — Navigation shard characterization**:
- To quantify the navigation shard's strength: in no-cheese mazes, probe for corner proximity at each layer; run causal tracing with the agent near vs. far from the corner. This establishes the navigation shard's encoding and causal effect sizes as a baseline for comparison.

**Expected outputs**: Comparison of shard signal strength across open arena and full maze; quantitative estimate of navigation shard interference.

**Success criterion**: Cheese shard causal effect in full maze is >0.1 (non-negligible) at the same layers where Experiment 1 found effects, even if smaller than the open arena result.

---

### Experiment 3: Shard Separability Test

**Goal**: Test D3 — that shards are causally independent — in both controlled and realistic settings.

**Setup**: Use the candidate shard locations from Experiments 1 and 2.

**Step 1 — Define shard vectors**:
- **Cheese shard vector**: Mean activation difference (cheese present minus cheese absent) at the identified shard layer, computed from open arena data.
- **Navigation shard vector**: Mean activation difference (near-corner minus center) at the navigation shard layer, computed from no-cheese maze data.

**Step 2 — Tier A: Controlled suppression (open arena)**:
- Suppress the cheese shard vector (project it out) in open arena; measure: (a) change in cheese-directed behavior, (b) change in any residual directional preference, (c) probe accuracy for navigation encoding.
- This tests cheese-navigation separability in the most favorable conditions.

**Step 3 — Tier B: Realistic suppression (full maze)**:
- Apply the same suppressions in the full maze.
- Suppress the cheese shard vector and measure: (a) change in cheese-directed behavior in full maze, (b) change in corner-seeking behavior, (c) probe accuracy for navigation shard encoding.
- Suppress the navigation shard vector and measure the mirror effects.
- Hypothesis: in the full maze, suppressing the cheese shard may partially affect navigation behavior (due to shard entanglement), whereas in the open arena separability should be higher.

**Step 4 — Quantify interference**:
- Define shard independence score I(S1, S2) = 1 − |Δbehavior(S2) when S1 suppressed| / |Δbehavior(S2) when S2 suppressed|. I = 1 means full independence.
- Compute I separately for Tier A and Tier B to measure how much the realistic environment increases shard entanglement.

**Expected outputs**: Independence scores in open arena and full maze; behavioral trajectories under single and dual suppression in both settings.

**Success criterion**: I > 0.8 in open arena (demonstrating separability exists); comparison of Tier A vs. Tier B I scores quantifies realistic-setting entanglement.

---

### Experiment 4: Reinforcement Distribution and Shard Strength

**Goal**: Test D4 — that shards form in proportion to reinforcement events — using open arena evaluation to measure shard properties cleanly.

**Setup**: Train three new maze-solving agents with different cheese placement distributions:
- **Agent A** (corner-biased): Cheese placed in the top-right 25% of mazes 75% of the time.
- **Agent B** (uniform): Cheese placed uniformly across all maze quadrants.
- **Agent C** (anti-corner): Cheese placed in the top-right only 10% of the time; bottom-left 70%.

**Step 1 — Train agents**: Use the same architecture and training procedure as the baseline agent.

**Step 2 — Open arena evaluation**:
- Apply the open arena shard detection pipeline (from Experiment 1) to each agent independently.
- Measure: (a) cheese shard strength (causal effect size), (b) directional preference strength and orientation (does it match the training distribution?), (c) contextual specificity (D1 score) for each.
- Using the open arena for evaluation avoids confounding navigation shard differences across agents with actual cheese shard differences.

**Step 3 — Full maze evaluation**:
- Apply the full maze pipeline (Experiment 2) to the same agents.
- Measure whether the reinforcement distribution differences that are detectable in the open arena also manifest in the full maze metrics.

**Step 4 — Correlation analysis**:
- Test whether corner preference direction and strength correlates with the training distribution across agents (Spearman correlation on open arena metrics).
- Test whether cheese shard strength is consistent across agents (since cheese is always reinforced, this should be stable regardless of placement distribution).

**Expected outputs**: Shard strength profiles for A, B, C agents in both settings; correlation between training distribution and shard properties.

**Success criterion**: Significant positive correlation (ρ > 0.7) between corner reinforcement frequency and corner preference strength in open arena evaluation across the three conditions.

---

### Experiment 5: Sparse Autoencoder Shard Discovery

**Goal**: Test whether unsupervised feature decomposition (SAE) recovers the same shards identified by supervised probing, and whether it performs differently across the two settings.

**Setup**: Train sparse autoencoders on activations at the candidate shard layer identified in Experiment 1, using data collected from both open arena and full maze.

**Step 1 — SAE training**:
- Collect >1M frames of activations from the open arena and separately from the full maze.
- Train one SAE per setting (expansion factor 8–16x, following Bricken et al. 2023).
- Identify top features by reconstruction importance.

**Step 2 — Feature-to-shard mapping**:
- For each SAE feature in both settings, compute its activation profile across contexts (cheese present/absent, near corner/center).
- Test whether cheese-related and navigation-related features emerge as distinct SAE features in each setting.
- Hypothesis: the open arena SAE should yield cleaner, more separable cheese-related features; the full maze SAE may produce more entangled features.

**Step 3 — Causal validation**:
- For SAE-identified candidate features, perform targeted activation patching using the SAE feature directions.
- Measure behavioral consequences in both settings.

**Expected outputs**: Set of SAE features from each setting; comparison of open arena vs. full maze SAE feature separability; comparison of SAE discovery vs. supervised probing.

**Success criterion**: Open arena SAE recovers at least one feature strongly correlated (r > 0.7) with the supervised probe for each identified shard. Full maze SAE comparison quantifies the practical cost of realistic-setting confounds for unsupervised discovery.

---

### Experiment 6: Generalization to a New Environment

**Goal**: Test whether the shard detection pipeline generalizes beyond the cheese maze.

**Setup**: Apply the pipeline to an agent trained on a different RL task with multiple distinguishable behavioral tendencies. Candidate environments: MiniGrid (supports modular navigation), or a custom 2D environment with two competing reward sources.

**Step 1 — Baseline characterization**: Establish the agent's behavioral tendencies and identify natural "controlled" vs. "realistic" analogs in the new environment (e.g., open field vs. structured room).

**Step 2 — Pipeline application**: Apply the Experiment 1 (controlled) and Experiment 2 (realistic) protocols to detect and test candidate shards.

**Expected outputs**: Detected shard candidates in the new environment; qualitative comparison of open arena vs. full environment signal across settings.

---

## 5. Metrics and Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Probe accuracy (open arena)** | Linear probe accuracy for shard concept in vs. out of context | >90% in-context, <60% out-of-context |
| **Probe accuracy (full maze)** | Same probe, full maze conditions | Expected lower; quantify gap vs. open arena |
| **Causal effect size (open arena)** | Δaction probability from activation patching | >0.3 (substantial effect) |
| **Causal effect size (full maze)** | Same metric in full maze | >0.1 (non-negligible; expected smaller) |
| **Independence score I (open arena)** | 1 − cross-shard behavioral leakage | >0.8 |
| **Independence score I (full maze)** | Same metric in full maze | Compared to open arena; quantifies entanglement |
| **SAE correlation (open arena)** | Pearson r between SAE feature and probe prediction | >0.7 |
| **Reinforcement correlation** | Spearman ρ between training distribution and shard strength | >0.7 |

---

## 6. Expected Outcomes

### If D1–D4 are confirmed in open arena and transfer to full maze:
- We will have established the first empirically validated, operationalized definition of a shard, confirmed to be robust across environmental complexity.
- The detection pipeline will provide a general tool for shard discovery applicable even in complex environments.

### If D1–D2 are confirmed in open arena but not in full maze:
- Shards exist but are overwhelmed by competing influences in the realistic setting. This would imply that shards are real computational structures but their behavioral influence is context-gated in a hierarchical way — the navigation shard gates when the cheese shard has behavioral access. This would be a significant finding, suggesting that "shard theory" needs a hierarchical extension: primary shards (navigation, always active) can suppress secondary shards (cheese, contextually accessible only when primary shards do not dominate).

### If D3 (separability) holds in open arena but fails in full maze:
- Shards that appear independent in isolation become entangled in complex environments. This would suggest that shards share intermediate representations that only become distinguishable under low-competition conditions, and that shard independence is an emergent property of context rather than a fixed architectural fact.

### If D4 (reinforcement traceability) fails:
- Shard formation may depend on architectural inductive biases or initialization more than on reward history — challenging a core prediction of Shard Theory.

---

## 7. Related Work to Track

- Aristizabal (2024) — prior empirical work on cheese-maze shard-like behavior, basis for two-tier design
- Soligo et al. (2025) — functional modularity in RL (closely related, compare methods)
- Anthropic Circuit Tracing (2025) — attribution graphs in LLMs (methodological inspiration)
- Cunningham et al. (2023) — SAE interpretability (tool for Experiment 5)
- Langosco et al. (2022) — goal misgeneralization (theoretical context)

---

## 8. Timeline (Rough)

| Phase | Activities | Duration |
|-------|-----------|----------|
| Phase 1 | Reproduce prior results; set up open arena infrastructure; establish two-tier evaluation scaffold | 2 weeks |
| Phase 2 | Experiment 1: Open arena probing + causal tracing | 3 weeks |
| Phase 3 | Experiment 2: Full maze validation + navigation shard characterization | 2 weeks |
| Phase 4 | Experiment 3: Shard separability (both settings) | 2 weeks |
| Phase 5 | Experiment 4: Reinforcement distribution (training + open arena eval) | 4 weeks |
| Phase 6 | Experiment 5: SAE discovery (both settings) | 2 weeks |
| Phase 7 | Experiment 6: Generalization to new environment | 3 weeks |
| Phase 8 | Analysis, writing, revision | 4 weeks |

---

## 9. References

- Turner, A., Pope, Q. "The Shard Theory of Human Values." AI Alignment Forum, 2022.
- Turner, A., et al. "Understanding and Controlling a Maze-Solving Policy Network." LessWrong, 2023.
- Aristizabal. "Exploring Shard-like Behavior: Empirical Insights into a Cheese-Seeking Maze Agent." LessWrong, 2024.
- Langosco et al. "Goal Misgeneralization in Deep RL." ICML, 2022. arXiv:2105.14111.
- Meng et al. "Locating and Editing Factual Associations in GPT." NeurIPS, 2022. arXiv:2202.05262.
- Bricken et al. "Towards Monosemanticity." Transformer Circuits Thread, 2023.
- Soligo et al. "Inducing, Detecting and Characterising Neural Modules." arXiv:2501.17077, 2025.
- Trim & Grayston. "Mechanistic Interpretability of RL Agents." arXiv:2411.00867, 2024.
- Belinkov. "Probing Classifiers: Promises, Shortcomings, and Advances." CL 48(1), 2022.
- Hewitt & Liang. "Designing and Interpreting Probes with Control Tasks." EMNLP, 2019.
