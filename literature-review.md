# Literature Review: Shard Theory and Contextual Decision Making in Neural Networks

**Date:** March 2026
**Project:** Contextual Shards — Toward an Empirically Testable Definition

---

## 1. Introduction

Shard Theory is a framework for understanding how values and behavioral tendencies emerge in reinforcement learning (RL) agents [Turner & Pope, 2022]. Its central claim is that agents should not be modeled as having a single unified goal but rather as a collection of *contextually activated decision influences* — shards — each of which is a downstream product of historical reinforcement events and activates selectively given appropriate environmental cues.

This review synthesizes the theoretical foundations of Shard Theory, the emerging body of mechanistic interpretability research that offers tools to test it, prior empirical work on maze-solving agents where shard-like signatures have been observed, and related frameworks on goal misgeneralization, feature decomposition, and causal intervention in neural networks.

The review is motivated by a prior exploratory study [Aristizabal, LessWrong 2024] that found evidence of spatial locality in cheese-seeking behavior of a trained RL agent — consistent with shard-like contextual activation — but did not produce a formally testable definition of a shard or a rigorous experimental design. The goal of this project is to close that gap.

---

## 2. Theoretical Background: Shard Theory

### 2.1 Core Claims

Turner and Pope introduced Shard Theory in a series of posts on the AI Alignment Forum [Turner & Pope, 2022]. The framework is formalized into nine main theses, of which the most empirically relevant are:

1. **Shards as computational units**: Agents are well-modeled as being composed of shards — contextually activated decision influences that are downstream of historical reinforcement events.
2. **World-model referential content**: Shards generally care about concepts inside the agent's world model rather than raw sensory experiences or reward maximization per se.
3. **Bidding mechanism**: Active shards bid for plans during decision-making, weighted by reinforcement history. The aggregate of bidding shards determines the agent's output.
4. **Context-dependency**: Shards are not uniformly active; they activate selectively when the agent's situation matches the context in which they were reinforced.

The theory emerged as a critique of classical alignment arguments that treat agents as monolithic goal-optimizers. Shard Theory instead proposes that behavior is an emergent consequence of many context-specific heuristics, each with a limited scope of activation.

### 2.2 Empirical Predictions

From the nine theses, several testable predictions can be derived:
- Behavioral effects attributable to distinct goals (e.g., "seek cheese" vs. "navigate to corner") should be spatially and contextually localized — observable in activations only when the relevant context is present.
- It should be possible to identify distinct neural circuits or directions in activation space corresponding to different shards.
- Suppressing or amplifying a shard's associated activations should produce predictable behavioral changes.
- Shards should operate quasi-independently: removing one should not destroy others.

### 2.3 Limitations of the Theory

Shard Theory is primarily a conceptual framework developed outside of standard academic venues. Its nine theses lack formal operationalization: "shard" is defined in terms of function (contextual activation + decision influence) but not in terms of what neural structures instantiate it. A behavioral genetic critique [cited in search results, 2023] also notes that the theory does not adequately account for innate/heritable contributions to values.

---

## 3. Empirical Precursor: The Maze-Solving Agent

### 3.1 Turner et al. (2023) — Understanding and Controlling

The most direct empirical predecessor to the proposed research is the work of Turner, Ulisse Mini, peligrietzer, and others analyzing a trained maze-solving policy network [Turner et al., 2023]. Their key findings were:

- The agent exhibits a **top-right corner bias** in the absence of cheese — a persistent behavioral tendency not directly shaped by the cheese-reward.
- When cheese is present, there is a competing **cheese-seeking shard** that activates in proximity.
- By **subtracting a fixed "cheese vector"** from network activations at a single layer, the agent's cheese-seeking behavior could be suppressed, while its corner-seeking behavior remained intact. This demonstrates causal separability between the two behaviors.
- Clamping a single activation to a positive value could attract the agent toward an arbitrary maze location — suggesting individual neurons carry interpretable spatial semantics.

These findings provide strong prima facie evidence for shard-like structure, but the work stops short of a formal definition or systematic test for shard existence.

### 3.2 Aristizabal (2024) — Exploratory Spatial Analysis

The blog post that motivates this project [Aristizabal, LessWrong 2024] investigated whether the cheese-seeking behavior is spatially localized by:

- Analyzing vector fields in empty mazes (no cheese) to establish baseline directional preferences.
- Varying cheese placement to observe behavior changes across the environment.
- Training logistic regression probes on multiple network layers to detect cheese presence, using varying activation budget constraints (top-k = 1, 10, 50, 100, 500, 1000).
- Computing activation difference magnitudes (with vs. without cheese) normalized by layer size across all layers.

Key findings:
- Cheese-seeking shows strong spatial locality: behavior changes are pronounced near cheese and minimal at distance.
- All network layers can encode cheese presence when given sufficient activations; perfect detection occurs in most layers.
- Later layers show stronger responses to cheese proximity than earlier layers.
- The fully connected layer preceding the value head exhibits particularly pronounced cheese-related effects.
- A single neuron in deeper residual layers may perfectly encode cheese presence.

The study concluded that these observations are "intriguing starting points for further investigation rather than conclusive evidence" of shard-like structures, identifying the need for a formal, testable definition of a shard.

---

## 4. Mechanistic Interpretability Methods

### 4.1 Probing Classifiers

Probing classifiers have become a standard tool for testing whether a network layer encodes a particular concept [Belinkov, 2022]. A linear probe trained to predict some property from a layer's activations reveals whether that property is linearly decodable at that layer — providing evidence about what information is encoded and where.

Limitations are well-documented [Belinkov, 2022]: probing accuracy does not establish causal role (a layer may encode a concept without using it), and probe performance depends on the complexity of the classifier used.

The Aristizabal (2024) study employed probing as its primary analytical tool. A key gap is that probing alone cannot establish that a detected feature is causally responsible for behavior — a limitation that motivates intervention-based methods.

### 4.2 Activation Patching and Causal Tracing

Meng et al. (2022) introduced **causal tracing** as a method for identifying which activations are causally responsible for model outputs [Meng et al., 2022]. The method runs the network twice — once normally and once with corrupted inputs — then restores specific activations to their clean values to identify which components are causally decisive.

This technique has been used to localize factual associations in language models to specific layers and token positions, demonstrating that specific knowledge can be attributed to narrow sub-circuits. Applied to RL agents, causal tracing could establish whether a putative shard is genuinely causally responsible for cheese-seeking behavior, not merely correlated with it.

### 4.3 Activation Steering and Intervention

The cheese vector result in Turner et al. (2023) is an instance of **activation steering** — identifying a direction in activation space that represents a behavioral tendency and then adding or subtracting it. Related work has shown that:

- Linear directions in residual stream space can represent interpretable concepts.
- Steering vectors can produce controlled behavioral changes in LLMs (many 2023–2024 papers).
- Single-neuron interventions can attract agents to arbitrary spatial locations.

Activation steering provides the most direct test of shard separability: if the cheese-shard and corner-shard are truly independent, subtracting one should not affect the other.

### 4.4 Sparse Autoencoders (SAEs) for Feature Decomposition

Anthropic's "Towards Monosemanticity" work [Bricken et al., 2023] demonstrated that networks represent more features than they have neurons (superposition). Sparse autoencoders trained on residual stream activations can decompose these superposed representations into approximately monosemantic features — each corresponding to a single, nameable concept.

SAEs represent a promising tool for shard discovery: if shards correspond to interpretable features in activation space, an SAE trained on the maze-solving network might automatically identify them. The 2024 scaling of SAEs to Claude 3 Sonnet [Templeton et al., 2024] and GPT-4 level models suggests this methodology is maturing.

### 4.5 Modular Neural Network Analysis

Soligo et al. (2025) developed a pipeline for inducing, detecting, and characterizing functional modules in RL policy networks [Soligo et al., 2025]. Using a modified Louvain algorithm with "correlation alignment," they detected distinct navigational modules for different axes in 2D and 3D MiniGrid environments. This work is methodologically adjacent to the shard discovery problem: both seek to identify semi-independent functional sub-units within a policy network.

Their approach of **inducing modularity** (through sparsity regularization during training) before detecting it may be a useful strategy for studying shards in a controlled experimental setting.

### 4.6 Mechanistic Interpretability of RL Agents

Trim and Grayston (2024) applied mechanistic interpretability methods directly to maze-navigation policy networks [Trim & Grayston, 2024]. Their study:

- Identified fundamental environmental features (walls, pathways) as the basis of decision-making.
- Confirmed goal misgeneralization: agents develop systematic top-right biases that persist in novel environments.
- Developed interactive visualization tools for exploration of network layer behaviors.

This work independently corroborates the Aristizabal (2024) findings using a broader toolkit.

---

## 5. Goal Misgeneralization

### 5.1 The Phenomenon

Langosco et al. (2022) formalized **goal misgeneralization** as distinct from capability misgeneralization: an agent can maintain full task competence in a novel environment while pursuing the wrong objective [Langosco et al., 2022]. A canonical example: an agent trained to navigate to a goal (co-located with a specific color during training) continues to navigate competently in the new environment, but navigates to the color marker rather than the true goal position.

Goal misgeneralization is directly relevant to shard theory: the top-right corner bias observed in the maze-solving network can be interpreted as a goal misgeneralization artifact — the "corner shard" was reinforced during training because the cheese was often placed in the upper-right area, creating a spurious context-action association.

### 5.2 Relationship to Shard Theory

Shard theory provides a mechanistic explanation for goal misgeneralization: the agent developed a "corner shard" during training (reinforced whenever it reached the corner and received reward), and this shard activates when appropriate contextual cues are present (being near the corner), regardless of whether cheese is present. In novel environments, the corner shard and cheese shard may compete, with the outcome depending on the relative strength of their contextual activations.

This suggests that goal misgeneralization and shard theory are not independent phenomena but that goal misgeneralization may be a natural consequence of shard formation under imperfect reward correlation.

---

## 6. Related Frameworks

### 6.1 World Models in RL

The shard theory claim that shards "care about concepts inside the agent's world model" presupposes that the agent has developed an internal world model. Ha and Schmidhuber's foundational work on world models [Ha & Schmidhuber, 2018] established that RL agents can develop compressed latent representations of their environments. Dreamer and successor work [Hafner et al., 2020–2023] demonstrated that explicit world models can dramatically improve sample efficiency.

For shard detection, this implies that the relevant representations for identifying shards may be in higher-level, more abstract layers that carry world-model content rather than raw sensory features.

### 6.2 Superposition and Feature Geometry

Elhage et al. (2022) and Bricken et al. (2023) established that neural networks represent far more features than their dimensional capacity through superposition. This has direct implications for shard detection: shards may not correspond to individual neurons but to directions in high-dimensional activation spaces. Methods that look for single neurons (as in Aristizabal 2024) may therefore under-detect shards.

---

## 7. Research Gaps

The review identifies the following concrete gaps that the proposed research aims to address:

### Gap 1: No Formal, Testable Definition of a Shard

Despite extensive theoretical discussion, no operational definition of a shard has been proposed that could be tested against neural network data. A shard is described functionally but not structurally. The proposed research should produce a definition that specifies: (a) what neural structures instantiate a shard, (b) what evidence suffices to identify one, and (c) what experiments could falsify its existence.

### Gap 2: Probing Without Causal Validation

Prior probing studies (including Aristizabal 2024) establish that layers *encode* cheese-related information but do not establish that this encoding *causes* cheese-seeking behavior. Causal tracing and activation steering experiments are needed to validate that identified features have causal roles.

### Gap 3: No Multi-Shard Interaction Study

No study has systematically characterized the interaction between multiple identified shards (e.g., corner-shard and cheese-shard). In particular, the independence assumption — that suppressing one shard does not affect others — has been tested informally (the cheese vector result) but not rigorously.

### Gap 4: No Generalizable Detection Method

Existing shard-related work is ad-hoc and specific to one trained network. A generalizable pipeline for shard detection — applicable across different RL environments and architectures — does not exist.

### Gap 5: No Quantitative Shard Metrics

There is no established metric for shard strength, shard specificity (how narrow the activation context is), or shard independence. Such metrics are necessary for comparing shards across models and environments.

---

## 8. Summary

| Paper | Key Contribution | Relevance |
|-------|-----------------|-----------|
| Turner & Pope (2022) | Shard Theory framework, nine theses | Theoretical foundation |
| Turner et al. (2023) | Cheese vector, activation control of maze agent | Primary empirical predecessor |
| Aristizabal (2024) | Spatial locality of cheese-seeking, layer-wise probing | Direct prior work |
| Langosco et al. (2022) | Goal misgeneralization formalization | Explains corner shard as misgeneralization |
| Trim & Grayston (2024) | Mechanistic interp of RL maze agents | Corroborating evidence |
| Soligo et al. (2025) | Neural module detection pipeline for RL | Methodological baseline |
| Meng et al. (2022) | Causal tracing (ROME) in transformers | Key causal intervention methodology |
| Bricken et al. (2023) | Sparse autoencoders, monosemanticity | Feature decomposition tool |
| Belinkov (2022) | Probing classifiers: limitations and best practices | Methodological grounding |

---

## References

See `references.bib` for full BibTeX entries.

- [Turner & Pope 2022] Turner, A., Pope, Q. "The Shard Theory of Human Values." AI Alignment Forum, 2022.
- [Aristizabal 2024] Aristizabal et al. "Exploring Shard-like Behavior: Empirical Insights into Contextual Activation." LessWrong, 2024.
- [Turner et al. 2023] Turner, A., peligrietzer, Mini, U., et al. "Understanding and Controlling a Maze-Solving Policy Network." LessWrong, 2023.
- [Langosco et al. 2022] Langosco, L., Koch, J., Sharkey, L., Pfau, J., Orseau, L., Krueger, D. "Goal Misgeneralization in Deep Reinforcement Learning." ICML 2022. arXiv:2105.14111.
- [Trim & Grayston 2024] Trim, T., Grayston, T. "Mechanistic Interpretability of Reinforcement Learning Agents." arXiv:2411.00867, 2024.
- [Soligo et al. 2025] Soligo, A., Ferraro, P., Boyle, D. "Inducing, Detecting and Characterising Neural Modules." arXiv:2501.17077, 2025.
- [Meng et al. 2022] Meng, K., Bau, D., Andonian, A., Belinkov, Y. "Locating and Editing Factual Associations in GPT." NeurIPS 2022. arXiv:2202.05262.
- [Bricken et al. 2023] Bricken, T., et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread, 2023.
- [Belinkov 2022] Belinkov, Y. "Probing Classifiers: Promises, Shortcomings, and Advances." Computational Linguistics 48(1), 2022.
- [Ha & Schmidhuber 2018] Ha, D., Schmidhuber, J. "World Models." arXiv:1803.10122, 2018.
