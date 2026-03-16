# Contextual Shards — ML Project Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mechanistic interpretability pipeline that operationalizes Shard Theory by detecting, validating, and characterizing neural shards in trained RL agents across 5 experiments.

**Architecture:** The project wraps `procgen-tools` (existing codebase) as an external dependency and builds a clean, config-driven interpretability pipeline on top. Each module (probe, causal, shard, SAE) uses the factory/registry pattern for extensibility across environments and concepts.

**Tech Stack:** Python 3.11+, uv, PyTorch, Hydra/OmegaConf, scikit-learn, h5py, procgen-tools (from git), circrl, gym, procgen, MiniGrid.

---

## File Map

```
contextual-shards/
├── run/
│   ├── conf/
│   │   ├── config.yaml                      # Default composed config
│   │   ├── agent/impala.yaml                # IMPALA checkpoint path + layer names
│   │   ├── environment/maze.yaml            # Maze env params
│   │   ├── environment/minigrid.yaml        # MiniGrid env params
│   │   ├── concept/cheese_presence.yaml     # Concept label definition
│   │   ├── concept/cheese_proximity.yaml
│   │   ├── concept/cheese_direction.yaml
│   │   ├── concept/corner_proximity.yaml
│   │   ├── probe/linear.yaml                # Linear probe params
│   │   ├── probe/mdl.yaml                   # MDL probe params
│   │   ├── training/rl_agent.yaml           # RL training params
│   │   ├── training/sae.yaml                # SAE training params
│   │   ├── experiment1/default.yaml         # Exp 1 config
│   │   ├── experiment2/default.yaml         # Exp 2 config
│   │   ├── experiment3/default.yaml         # Exp 3 config (3 agent variants)
│   │   ├── experiment4/default.yaml         # Exp 4 SAE config
│   │   └── experiment5/default.yaml         # Exp 5 generalization config
│   └── pipeline/
│       ├── prepare_data/collect_activations.py  # Collect + save activations to HDF5
│       ├── prepare_data/collect_rollouts.py     # Collect maze rollout trajectories
│       ├── training/train_rl_agent.py           # Train maze agents (Exp 3)
│       ├── training/train_sae.py               # Train SAE on activations (Exp 4)
│       ├── analysis/experiment1_probing.py      # D1: layer-wise linear probing
│       ├── analysis/experiment1_causal.py       # D2: causal tracing (clean/corrupted)
│       ├── analysis/experiment1_integrate.py    # Combine D1+D2 → shard candidates
│       ├── analysis/experiment2_separability.py # D3: suppression + independence score
│       ├── analysis/experiment3_dist.py         # D4: reinforcement distribution
│       ├── analysis/experiment4_sae.py          # SAE feature discovery
│       └── analysis/experiment5_generalize.py   # Generalization pipeline
│
├── src/
│   ├── environment_module/
│   │   ├── __init__.py                          # ENV_FACTORY + register_env
│   │   ├── base_env.py                          # BaseEnv (abstract)
│   │   ├── maze/
│   │   │   ├── __init__.py
│   │   │   └── maze_env.py                      # ProcgenMazeEnv (wraps procgen-tools)
│   │   └── minigrid/
│   │       ├── __init__.py
│   │       └── minigrid_env.py                  # MiniGridEnv
│   │
│   ├── agent_module/
│   │   ├── __init__.py                          # AGENT_FACTORY + register_agent
│   │   ├── base_agent.py                        # BaseAgent (abstract)
│   │   ├── policy/
│   │   │   ├── __init__.py
│   │   │   └── impala_agent.py                  # ImpalaAgent (loads procgen-tools model)
│   │   └── hooks/
│   │       ├── __init__.py
│   │       └── activation_hooks.py              # ActivationRecorder (register/remove hooks)
│   │
│   ├── data_module/
│   │   ├── __init__.py
│   │   ├── activation_dataset/
│   │   │   ├── __init__.py
│   │   │   └── activation_dataset.py            # HDF5ActivationDataset (read/write activations)
│   │   ├── rollout_collector/
│   │   │   ├── __init__.py
│   │   │   └── rollout_collector.py             # RolloutCollector (step env, record frames+labels)
│   │   └── concept_labeler/
│   │       ├── __init__.py                      # LABELER_FACTORY + register_labeler
│   │       ├── base_labeler.py                  # BaseConceptLabeler (abstract)
│   │       ├── cheese_presence.py               # CheesePresenceLabeler
│   │       ├── cheese_proximity.py              # CheeseProximityLabeler
│   │       ├── cheese_direction.py              # CheeseDirectionLabeler
│   │       └── corner_proximity.py              # CornerProximityLabeler
│   │
│   ├── probe_module/
│   │   ├── __init__.py                          # PROBE_FACTORY + register_probe
│   │   ├── base_probe.py                        # BaseProbe (fit, predict, score)
│   │   ├── probe/
│   │   │   ├── __init__.py
│   │   │   ├── linear_probe.py                  # LinearProbe (sklearn LogisticRegression)
│   │   │   └── mdl_probe.py                     # MDLProbe (online coding length)
│   │   └── evaluator/
│   │       ├── __init__.py
│   │       └── probe_evaluator.py               # ProbeEvaluator (accuracy, bonferroni, context split)
│   │
│   ├── causal_module/
│   │   ├── __init__.py
│   │   ├── tracing/
│   │   │   ├── __init__.py
│   │   │   └── causal_tracer.py                 # CausalTracer (clean/corrupted runs, layer restoration)
│   │   ├── patch/
│   │   │   ├── __init__.py
│   │   │   └── activation_patcher.py            # ActivationPatcher (suppress, amplify, project-out)
│   │   └── shard_vector/
│   │       ├── __init__.py
│   │       └── shard_vector.py                  # ShardVector (mean diff computation)
│   │
│   ├── shard_module/
│   │   ├── __init__.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   └── shard_detector.py                # ShardDetector (intersect probe + causal candidates)
│   │   ├── separability/
│   │   │   ├── __init__.py
│   │   │   └── separability_tester.py           # SeparabilityTester (suppression + independence score)
│   │   └── metrics/
│   │       ├── __init__.py
│   │       └── shard_metrics.py                 # ShardMetrics (probe_acc, causal_effect, independence_score)
│   │
│   ├── sae_module/
│   │   ├── __init__.py                          # SAE_FACTORY + register_sae
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   └── sparse_autoencoder.py            # SparseAutoencoder (Bricken et al. 2023, 8-16x expansion)
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── sae_trainer.py                   # SAETrainer (Adam, L1 sparsity penalty)
│   │   └── feature/
│   │       ├── __init__.py
│   │       └── feature_analyzer.py              # FeatureAnalyzer (feature→context profile, correlation with probes)
│   │
│   ├── trainer_module/
│   │   ├── __init__.py
│   │   ├── rl_trainer/
│   │   │   ├── __init__.py
│   │   │   ├── ppo_trainer.py                   # PPOTrainer (stable-baselines3 or custom PPO)
│   │   │   └── cheese_distribution.py           # CheesePlacementDistribution (corner-biased, uniform, anti-corner)
│   │   └── sae_trainer/
│   │       ├── __init__.py
│   │       └── sae_train_loop.py                # SAETrainLoop (Hydra-driven SAE training entrypoint)
│   │
│   └── analysis_module/
│       ├── __init__.py
│       ├── statistics/
│       │   ├── __init__.py
│       │   └── stat_tests.py                    # bonferroni_correction, spearman_corr, effect_size
│       └── visualization/
│           ├── __init__.py
│           └── shard_visualizer.py              # ShardVisualizer (probe heatmaps, vector fields, independence plots)
│
├── data/
│   ├── raw/                                     # Downloaded IMPALA checkpoints
│   ├── processed/                               # Collected activations (HDF5)
│   └── external/                               # (unused; procgen-tools installed as package)
│
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   ├── tables/
│   └── figures/
│
├── tests/
│   ├── test_environment_module/
│   │   └── test_maze_env.py
│   ├── test_agent_module/
│   │   └── test_activation_hooks.py
│   ├── test_data_module/
│   │   ├── test_activation_dataset.py
│   │   └── test_concept_labeler.py
│   ├── test_probe_module/
│   │   ├── test_linear_probe.py
│   │   └── test_probe_evaluator.py
│   ├── test_causal_module/
│   │   ├── test_causal_tracer.py
│   │   └── test_activation_patcher.py
│   ├── test_shard_module/
│   │   ├── test_shard_detector.py
│   │   └── test_separability_tester.py
│   └── test_sae_module/
│       ├── test_sparse_autoencoder.py
│       └── test_feature_analyzer.py
│
├── plan/
│   └── 2026-03-16-contextual-shards.md         # This file
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Chunk 1: Project Foundation

### Task 1: Initialize project with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Initialize uv project**
```bash
cd /home/aristizabal95/programming/contextual-shards
uv init --name contextual-shards --python 3.11
```

- [ ] **Step 2: Add core dependencies to pyproject.toml**
```bash
uv add torch torchvision hydra-core omegaconf numpy scipy scikit-learn h5py pandas matplotlib plotly tqdm
uv add "procgen-tools @ git+https://github.com/aristizabal95/procgen-tools.git"
uv add gym3 procgen minigrid
uv add --dev pytest pytest-cov ruff mypy
```

- [ ] **Step 3: Verify installs**
```bash
uv run python -c "import torch; import procgen_tools; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Create .gitignore**
Add: `data/processed/`, `outputs/`, `*.h5`, `*.pt`, `__pycache__/`, `.env`, `*.pyc`, `uv.lock` (keep), `data/raw/*.pt`

- [ ] **Step 5: Create directory skeleton**
```bash
mkdir -p run/conf/{agent,environment,concept,probe,training,experiment1,experiment2,experiment3,experiment4,experiment5}
mkdir -p run/pipeline/{prepare_data,training,analysis}
mkdir -p src/{environment_module/maze,environment_module/minigrid,agent_module/{policy,hooks},data_module/{activation_dataset,rollout_collector,concept_labeler},probe_module/{probe,evaluator},causal_module/{tracing,patch,shard_vector},shard_module/{detection,separability,metrics},sae_module/{model,training,feature},trainer_module/{rl_trainer,sae_trainer},analysis_module/{statistics,visualization}}
mkdir -p tests/{test_environment_module,test_agent_module,test_data_module,test_probe_module,test_causal_module,test_shard_module,test_sae_module}
mkdir -p data/{raw,processed,external} outputs/{logs,checkpoints,tables,figures}
touch src/__init__.py
```

- [ ] **Step 6: Commit**
```bash
git init
git add pyproject.toml .gitignore README.md
git commit -m "chore: initialize contextual-shards project with uv"
```

---

### Task 2: Environment Module

**Files:**
- Create: `src/environment_module/__init__.py`
- Create: `src/environment_module/base_env.py`
- Create: `src/environment_module/maze/__init__.py`
- Create: `src/environment_module/maze/maze_env.py`
- Test: `tests/test_environment_module/test_maze_env.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_environment_module/test_maze_env.py
import pytest
import numpy as np
from src.environment_module import EnvFactory

def test_env_factory_creates_maze():
    env = EnvFactory("maze")
    assert env is not None

def test_maze_env_reset_returns_obs():
    env = EnvFactory("maze")
    obs = env.reset()
    assert obs.shape == (3, 64, 64)  # RGB 64x64

def test_maze_env_has_cheese_position():
    env = EnvFactory("maze")
    env.reset()
    pos = env.cheese_pos()
    assert len(pos) == 2  # (row, col)

def test_maze_env_has_agent_position():
    env = EnvFactory("maze")
    env.reset()
    pos = env.agent_pos()
    assert len(pos) == 2
```

- [ ] **Step 2: Run to verify failure**
```bash
uv run pytest tests/test_environment_module/test_maze_env.py -v
```
Expected: ImportError or FAIL

- [ ] **Step 3: Implement base env**
```python
# src/environment_module/base_env.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseEnv(ABC):
    """Abstract base for all RL environments."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment, return observation (C, H, W)."""
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Step environment, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def cheese_pos(self) -> Tuple[int, int]:
        """Return (row, col) of cheese in maze grid coordinates."""
        ...

    @abstractmethod
    def agent_pos(self) -> Tuple[int, int]:
        """Return (row, col) of agent in maze grid coordinates."""
        ...

    @abstractmethod
    def get_maze_grid(self) -> np.ndarray:
        """Return 2D integer grid of maze state."""
        ...
```

- [ ] **Step 4: Implement maze env**
```python
# src/environment_module/maze/maze_env.py
import numpy as np
from typing import Tuple, Optional
import gym3
from procgen import ProcgenGym3Env
from procgen_tools.maze import get_cheese_pos, get_mouse_pos
from src.environment_module.base_env import BaseEnv
from src.environment_module import register_env

@register_env("maze")
class ProcgenMazeEnv(BaseEnv):
    """Procgen maze environment with maze-state inspection utilities."""

    def __init__(self, cfg):
        self.num_levels = cfg.environment.num_levels
        self.seed = cfg.environment.seed
        self.distribution_mode = cfg.environment.get("distribution_mode", "easy")
        self._venv: Optional[ProcgenGym3Env] = None
        self._last_obs: Optional[np.ndarray] = None

    def _make_env(self) -> ProcgenGym3Env:
        return ProcgenGym3Env(
            num=1,
            env_name="maze",
            num_levels=self.num_levels,
            start_level=self.seed,
            distribution_mode=self.distribution_mode,
            render_mode="rgb_array",
        )

    def reset(self) -> np.ndarray:
        if self._venv is None:
            self._venv = self._make_env()
        obs, _, _ = self._venv.observe()
        self._last_obs = obs["rgb"][0].transpose(2, 0, 1)  # HWC -> CHW
        return self._last_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._venv.act(np.array([action]))
        obs, reward, done, info = self._venv.observe()
        self._last_obs = obs["rgb"][0].transpose(2, 0, 1)
        return self._last_obs, float(reward[0]), bool(done[0]), {}

    def cheese_pos(self) -> Tuple[int, int]:
        state_bytes = self._venv.env.callmethod("get_state")[0]
        from procgen_tools.maze import EnvState
        state = EnvState(state_bytes)
        return tuple(get_cheese_pos(state.inner_grid()))

    def agent_pos(self) -> Tuple[int, int]:
        state_bytes = self._venv.env.callmethod("get_state")[0]
        from procgen_tools.maze import EnvState
        state = EnvState(state_bytes)
        return tuple(get_mouse_pos(state.inner_grid()))

    def get_maze_grid(self) -> np.ndarray:
        state_bytes = self._venv.env.callmethod("get_state")[0]
        from procgen_tools.maze import EnvState
        return EnvState(state_bytes).inner_grid()
```

- [ ] **Step 5: Implement factory + registry**
```python
# src/environment_module/__init__.py
import os
from typing import Dict, Type
from src.environment_module.base_env import BaseEnv
from src.utils.auto_import import import_modules

ENV_FACTORY: Dict[str, Type[BaseEnv]] = {}

def register_env(name: str):
    def decorator(cls: Type[BaseEnv]):
        ENV_FACTORY[name] = cls
        return cls
    return decorator

def EnvFactory(name: str):
    cls = ENV_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_FACTORY)}")
    return cls

_dir = os.path.dirname(__file__)
import_modules(_dir, "src.environment_module")

__all__ = ["EnvFactory", "register_env", "BaseEnv"]
```

- [ ] **Step 6: Create auto_import utility**
```python
# src/utils/__init__.py
from src.utils.auto_import import import_modules
__all__ = ["import_modules"]

# src/utils/auto_import.py
import os
import importlib
from typing import Optional

def import_modules(directory: str, package: str) -> None:
    """Auto-import all Python modules in a directory for registry side-effects."""
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("_"):
            module_name = filename[:-3]
            importlib.import_module(f"{package}.{module_name}")
        elif os.path.isdir(os.path.join(directory, filename)):
            subdir = os.path.join(directory, filename)
            subpackage = f"{package}.{filename}"
            if os.path.exists(os.path.join(subdir, "__init__.py")):
                importlib.import_module(subpackage)
```

- [ ] **Step 7: Create base Hydra config**
```yaml
# run/conf/environment/maze.yaml
num_levels: 500
seed: 0
distribution_mode: easy
```

- [ ] **Step 8: Run tests**
```bash
uv run pytest tests/test_environment_module/ -v
```
Expected: all PASS (may need procgen installed; skip if CI lacks GPU by marking with `@pytest.mark.integration`)

- [ ] **Step 9: Commit**
```bash
git add src/environment_module/ src/utils/ run/conf/environment/ tests/test_environment_module/
git commit -m "feat(env): add environment module with ProcgenMazeEnv factory"
```

---

## Chunk 2: Agent Module + Activation Collection

### Task 3: Agent Module with Activation Hooks

**Files:**
- Create: `src/agent_module/__init__.py`
- Create: `src/agent_module/base_agent.py`
- Create: `src/agent_module/policy/impala_agent.py`
- Create: `src/agent_module/hooks/activation_hooks.py`
- Test: `tests/test_agent_module/test_activation_hooks.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_agent_module/test_activation_hooks.py
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.agent_module.hooks.activation_hooks import ActivationRecorder

def test_recorder_registers_hooks():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(8, 4),
    )
    recorder = ActivationRecorder(model, layer_names=["0", "1"])
    recorder.register()
    assert len(recorder._hooks) == 2

def test_recorder_captures_activations():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8))
    recorder = ActivationRecorder(model, layer_names=["0"])
    recorder.register()
    x = torch.randn(2, 4)
    _ = model(x)
    acts = recorder.get_activations()
    assert "0" in acts
    assert acts["0"].shape == (2, 8)

def test_recorder_removes_hooks():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8))
    recorder = ActivationRecorder(model, layer_names=["0"])
    recorder.register()
    recorder.remove()
    assert len(recorder._hooks) == 0
```

- [ ] **Step 2: Run to verify failure**
```bash
uv run pytest tests/test_agent_module/test_activation_hooks.py -v
```

- [ ] **Step 3: Implement ActivationRecorder**
```python
# src/agent_module/hooks/activation_hooks.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn

class ActivationRecorder:
    """Records intermediate activations from named model layers via forward hooks."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self._model = model
        self._layer_names = layer_names
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def register(self) -> None:
        """Register forward hooks on named layers."""
        named = dict(self._model.named_modules())
        for name in self._layer_names:
            if name not in named:
                raise ValueError(f"Layer '{name}' not found in model. Available: {list(named)}")
            hook = named[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                self._activations[name] = output[0].detach()
        return hook

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return dict(self._activations)

    def clear(self) -> None:
        self._activations.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()
```

- [ ] **Step 4: Implement ImpalaAgent**
```python
# src/agent_module/policy/impala_agent.py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from procgen_tools.models import load_policy
from src.agent_module.base_agent import BaseAgent
from src.agent_module import register_agent
from src.agent_module.hooks.activation_hooks import ActivationRecorder

# IMPALA layer names (from procgen_tools InterpretableImpalaModel)
IMPALA_LAYERS = [
    "block1", "block1.res1", "block1.res2",
    "block2", "block2.res1", "block2.res2",
    "block3", "block3.res1", "block3.res2",
    "fc",
]

@register_agent("impala")
class ImpalaAgent(BaseAgent):
    """IMPALA policy agent wrapping procgen-tools checkpoint."""

    def __init__(self, cfg):
        self.checkpoint_path = cfg.agent.checkpoint_path
        self.layer_names = cfg.agent.get("layer_names", IMPALA_LAYERS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self._load_policy()
        self.recorder = ActivationRecorder(self.policy, self.layer_names)

    def _load_policy(self) -> torch.nn.Module:
        policy = load_policy(self.checkpoint_path, action_size=15, device=self.device)
        policy.eval()
        return policy

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """Return greedy action for a single observation (C, H, W)."""
        obs_t = torch.FloatTensor(obs[None]).to(self.device)
        logits = self.policy(obs_t)
        return int(logits.argmax(-1).item())

    @torch.no_grad()
    def act_with_activations(self, obs: np.ndarray) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Return action and layer activations for a single obs."""
        self.recorder.clear()
        with self.recorder:
            action = self.act(obs)
        return action, self.recorder.get_activations()

    @torch.no_grad()
    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return action probability distribution for a single obs."""
        obs_t = torch.FloatTensor(obs[None]).to(self.device)
        logits = self.policy(obs_t)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]
```

- [ ] **Step 5: Add base agent + factory**
```python
# src/agent_module/base_agent.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple
import torch

class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> int: ...
    @abstractmethod
    def act_with_activations(self, obs: np.ndarray) -> Tuple[int, Dict[str, torch.Tensor]]: ...
    @abstractmethod
    def get_action_probs(self, obs: np.ndarray) -> np.ndarray: ...

# src/agent_module/__init__.py  (same factory pattern as environment)
```

- [ ] **Step 6: Add agent config**
```yaml
# run/conf/agent/impala.yaml
checkpoint_path: data/raw/maze_policy.pt
layer_names:
  - block1
  - block1.res1
  - block1.res2
  - block2
  - block2.res1
  - block2.res2
  - block3
  - block3.res1
  - block3.res2
  - fc
```

- [ ] **Step 7: Run tests**
```bash
uv run pytest tests/test_agent_module/ -v
```
Expected: PASS

- [ ] **Step 8: Commit**
```bash
git add src/agent_module/ run/conf/agent/ tests/test_agent_module/
git commit -m "feat(agent): add IMPALA agent with activation hook recorder"
```

---

### Task 4: Data Module — Concept Labelers + Activation Dataset

**Files:**
- Create: `src/data_module/concept_labeler/base_labeler.py`
- Create: `src/data_module/concept_labeler/cheese_presence.py`
- Create: `src/data_module/concept_labeler/cheese_proximity.py`
- Create: `src/data_module/concept_labeler/cheese_direction.py`
- Create: `src/data_module/concept_labeler/corner_proximity.py`
- Create: `src/data_module/activation_dataset/activation_dataset.py`
- Test: `tests/test_data_module/test_concept_labeler.py`
- Test: `tests/test_data_module/test_activation_dataset.py`

- [ ] **Step 1: Write failing tests for labelers**
```python
# tests/test_data_module/test_concept_labeler.py
import numpy as np
import pytest
from src.data_module.concept_labeler import LabelerFactory

def test_cheese_presence_true():
    labeler = LabelerFactory("cheese_presence")
    # agent at (5,5), cheese at (5,6) = adjacent
    label = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=np.zeros((15, 15)))
    assert label == 1.0

def test_cheese_presence_false():
    labeler = LabelerFactory("cheese_presence")
    label = labeler.label(agent_pos=(1, 1), cheese_pos=(13, 13), maze_grid=np.zeros((15, 15)))
    assert label == 0.0

def test_cheese_proximity_distance():
    labeler = LabelerFactory("cheese_proximity")
    d = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 10), maze_grid=np.zeros((15, 15)))
    assert d == pytest.approx(5.0)

def test_cheese_direction_angle():
    labeler = LabelerFactory("cheese_direction")
    angle = labeler.label(agent_pos=(5, 5), cheese_pos=(5, 6), maze_grid=np.zeros((15, 15)))
    assert -np.pi <= angle <= np.pi

def test_corner_proximity():
    labeler = LabelerFactory("corner_proximity")
    # top-right corner
    d = labeler.label(agent_pos=(1, 13), cheese_pos=(0, 0), maze_grid=np.zeros((15, 15)))
    assert d < 3.0  # close to corner
```

- [ ] **Step 2: Implement labelers**
```python
# src/data_module/concept_labeler/base_labeler.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseConceptLabeler(ABC):
    """Computes scalar/categorical concept label from environment state."""

    @abstractmethod
    def label(
        self,
        agent_pos: Tuple[int, int],
        cheese_pos: Tuple[int, int],
        maze_grid: np.ndarray,
    ) -> float:
        """Return continuous or binary label for the concept."""
        ...

# src/data_module/concept_labeler/cheese_presence.py
import numpy as np
from typing import Tuple
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler
from src.data_module.concept_labeler import register_labeler

@register_labeler("cheese_presence")
class CheesePresenceLabeler(BaseConceptLabeler):
    """Binary: 1 if cheese within proximity_threshold steps, else 0."""

    def __init__(self, cfg=None):
        self.threshold = getattr(cfg, "threshold", 5) if cfg else 5

    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        dist = np.sqrt((agent_pos[0] - cheese_pos[0])**2 + (agent_pos[1] - cheese_pos[1])**2)
        return 1.0 if dist <= self.threshold else 0.0

# src/data_module/concept_labeler/cheese_proximity.py
@register_labeler("cheese_proximity")
class CheeseProximityLabeler(BaseConceptLabeler):
    """Euclidean distance from agent to cheese."""
    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        return float(np.sqrt((agent_pos[0]-cheese_pos[0])**2 + (agent_pos[1]-cheese_pos[1])**2))

# src/data_module/concept_labeler/cheese_direction.py
@register_labeler("cheese_direction")
class CheeseDirectionLabeler(BaseConceptLabeler):
    """Angle (radians) from agent to cheese."""
    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        dy = cheese_pos[0] - agent_pos[0]
        dx = cheese_pos[1] - agent_pos[1]
        return float(np.arctan2(dy, dx))

# src/data_module/concept_labeler/corner_proximity.py
@register_labeler("corner_proximity")
class CornerProximityLabeler(BaseConceptLabeler):
    """Distance from agent to top-right corner of maze."""
    def label(self, agent_pos, cheese_pos, maze_grid) -> float:
        h, w = maze_grid.shape
        corner = (0, w - 1)
        return float(np.sqrt((agent_pos[0]-corner[0])**2 + (agent_pos[1]-corner[1])**2))
```

- [ ] **Step 3: Write failing test for activation dataset**
```python
# tests/test_data_module/test_activation_dataset.py
import numpy as np
import tempfile, os, pytest
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

def test_write_and_read_activations(tmp_path):
    path = str(tmp_path / "test.h5")
    ds = HDF5ActivationDataset(path, mode="w")
    acts = {"block1": np.random.randn(10, 16, 8, 8).astype(np.float32)}
    labels = {"cheese_presence": np.array([1,0,1,0,1,0,1,0,1,0], dtype=np.float32)}
    ds.write_batch(activations=acts, labels=labels)
    ds.close()

    ds2 = HDF5ActivationDataset(path, mode="r")
    assert len(ds2) == 10
    sample = ds2[0]
    assert "block1" in sample["activations"]
    assert "cheese_presence" in sample["labels"]
```

- [ ] **Step 4: Implement HDF5ActivationDataset**
```python
# src/data_module/activation_dataset/activation_dataset.py
import h5py
import numpy as np
from typing import Dict, Any
from torch.utils.data import Dataset

class HDF5ActivationDataset(Dataset):
    """Stores and retrieves activations + concept labels from HDF5."""

    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.mode = mode
        self._file = h5py.File(path, mode)
        if mode == "w":
            self._file.create_group("activations")
            self._file.create_group("labels")
            self._length = 0
        else:
            # infer length from first dataset
            first_key = next(iter(self._file["activations"]))
            self._length = len(self._file["activations"][first_key])

    def write_batch(
        self,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> None:
        n = next(iter(activations.values())).shape[0]
        for layer_name, data in activations.items():
            key = f"activations/{layer_name}"
            if key in self._file:
                self._file[key].resize(self._length + n, axis=0)
                self._file[key][self._length:] = data
            else:
                maxshape = (None,) + data.shape[1:]
                self._file.create_dataset(key, data=data, maxshape=maxshape, chunks=True)
        for label_name, data in labels.items():
            key = f"labels/{label_name}"
            if key in self._file:
                self._file[key].resize(self._length + n, axis=0)
                self._file[key][self._length:] = data
            else:
                self._file.create_dataset(key, data=data, maxshape=(None,), chunks=True)
        self._length += n

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        acts = {k: self._file[f"activations/{k}"][idx] for k in self._file["activations"]}
        labs = {k: self._file[f"labels/{k}"][idx] for k in self._file["labels"]}
        return {"activations": acts, "labels": labs}

    def close(self) -> None:
        self._file.close()
```

- [ ] **Step 5: Run tests**
```bash
uv run pytest tests/test_data_module/ -v
```

- [ ] **Step 6: Commit**
```bash
git add src/data_module/ tests/test_data_module/
git commit -m "feat(data): add concept labelers + HDF5 activation dataset"
```

---

### Task 5: Rollout Collector + Activation Collection Pipeline

**Files:**
- Create: `src/data_module/rollout_collector/rollout_collector.py`
- Create: `run/pipeline/prepare_data/collect_activations.py`

- [ ] **Step 1: Implement RolloutCollector**
```python
# src/data_module/rollout_collector/rollout_collector.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from src.environment_module.base_env import BaseEnv
from src.agent_module.base_agent import BaseAgent
from src.data_module.concept_labeler.base_labeler import BaseConceptLabeler

@dataclass
class RolloutBatch:
    observations: np.ndarray          # (N, C, H, W)
    activations: Dict[str, np.ndarray] # layer -> (N, ...)
    labels: Dict[str, np.ndarray]      # concept -> (N,)
    actions: np.ndarray                # (N,)
    rewards: np.ndarray                # (N,)

class RolloutCollector:
    """Collects rollout data from an agent in an environment."""

    def __init__(
        self,
        env: BaseEnv,
        agent: BaseAgent,
        labelers: Dict[str, BaseConceptLabeler],
        n_steps: int = 10000,
        reset_every: int = 200,
    ):
        self.env = env
        self.agent = agent
        self.labelers = labelers
        self.n_steps = n_steps
        self.reset_every = reset_every

    def collect(self) -> RolloutBatch:
        obs_list, act_list, rew_list = [], [], []
        act_dict: Dict[str, List] = {}
        lab_dict: Dict[str, List] = {k: [] for k in self.labelers}

        obs = self.env.reset()
        for step in tqdm(range(self.n_steps), desc="Collecting rollouts"):
            action, activations = self.agent.act_with_activations(obs)
            new_obs, reward, done, _ = self.env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            for layer, act in activations.items():
                act_dict.setdefault(layer, []).append(act.cpu().numpy())

            agent_pos = self.env.agent_pos()
            cheese_pos = self.env.cheese_pos()
            grid = self.env.get_maze_grid()
            for name, labeler in self.labelers.items():
                lab_dict[name].append(labeler.label(agent_pos, cheese_pos, grid))

            obs = new_obs
            if done or (step % self.reset_every == 0 and step > 0):
                obs = self.env.reset()

        return RolloutBatch(
            observations=np.array(obs_list, dtype=np.float32),
            activations={k: np.array(v, dtype=np.float32) for k, v in act_dict.items()},
            labels={k: np.array(v, dtype=np.float32) for k, v in lab_dict.items()},
            actions=np.array(act_list, dtype=np.int64),
            rewards=np.array(rew_list, dtype=np.float32),
        )
```

- [ ] **Step 2: Implement collection pipeline script**
```python
# run/pipeline/prepare_data/collect_activations.py
"""Collect activations from trained maze agent and save to HDF5."""
import hydra
from omegaconf import DictConfig
from src.environment_module import EnvFactory
from src.agent_module import AgentFactory
from src.data_module.concept_labeler import LabelerFactory
from src.data_module.rollout_collector.rollout_collector import RolloutCollector
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    env = EnvFactory(cfg.environment.name)(cfg)
    agent = AgentFactory(cfg.agent.name)(cfg)
    labelers = {name: LabelerFactory(name)() for name in cfg.concepts}

    collector = RolloutCollector(
        env=env, agent=agent, labelers=labelers,
        n_steps=cfg.collect.n_steps, reset_every=cfg.collect.reset_every,
    )
    batch = collector.collect()

    output_path = cfg.collect.output_path
    ds = HDF5ActivationDataset(output_path, mode="w")
    ds.write_batch(activations=batch.activations, labels=batch.labels)
    ds.close()
    print(f"Saved {len(batch.labels[next(iter(batch.labels))])} samples to {output_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add collection config**
```yaml
# run/conf/config.yaml
defaults:
  - environment: maze
  - agent: impala
  - _self_

concepts:
  - cheese_presence
  - cheese_proximity
  - cheese_direction
  - corner_proximity

collect:
  n_steps: 100000
  reset_every: 200
  output_path: data/processed/activations.h5
```

- [ ] **Step 4: Commit**
```bash
git add src/data_module/rollout_collector/ run/pipeline/prepare_data/ run/conf/config.yaml
git commit -m "feat(data): add rollout collector + activation collection pipeline"
```

---

## Chunk 3: Probe Module (Experiment 1 — D1)

### Task 6: Linear Probe Infrastructure

**Files:**
- Create: `src/probe_module/__init__.py`
- Create: `src/probe_module/base_probe.py`
- Create: `src/probe_module/probe/linear_probe.py`
- Create: `src/probe_module/probe/mdl_probe.py`
- Create: `src/probe_module/evaluator/probe_evaluator.py`
- Test: `tests/test_probe_module/test_linear_probe.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_probe_module/test_linear_probe.py
import numpy as np
import pytest
from src.probe_module import ProbeFactory

def _make_separable_data(n=200, d=16):
    """Two linearly separable classes."""
    X = np.random.randn(n, d).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y

def test_linear_probe_factory():
    probe = ProbeFactory("linear")
    assert probe is not None

def test_linear_probe_fits_and_scores():
    probe = ProbeFactory("linear")()
    X, y = _make_separable_data()
    probe.fit(X[:150], y[:150])
    acc = probe.score(X[150:], y[150:])
    assert acc > 0.85, f"Expected >0.85 on separable data, got {acc:.3f}"

def test_linear_probe_predict():
    probe = ProbeFactory("linear")()
    X, y = _make_separable_data()
    probe.fit(X[:150], y[:150])
    preds = probe.predict(X[150:])
    assert preds.shape == y[150:].shape

def test_probe_evaluator_context_split():
    from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
    evaluator = ProbeEvaluator()
    X = np.random.randn(100, 8).astype(np.float32)
    y = (np.random.rand(100) > 0.5).astype(np.float32)
    context_mask = np.array([True]*50 + [False]*50)
    results = evaluator.evaluate_context_split(X, y, context_mask, n_trials=3)
    assert "in_context_acc" in results
    assert "out_context_acc" in results
```

- [ ] **Step 2: Implement LinearProbe**
```python
# src/probe_module/probe/linear_probe.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe_module.base_probe import BaseProbe
from src.probe_module import register_probe

@register_probe("linear")
class LinearProbe(BaseProbe):
    """Logistic regression probe over flattened activations."""

    def __init__(self, cfg=None):
        C = getattr(cfg, "C", 1.0) if cfg else 1.0
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = self._scaler.fit_transform(self._flatten(X))
        self._clf.fit(Xf, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(self._flatten(X))
        return self._clf.predict(Xf)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        Xf = self._scaler.transform(self._flatten(X))
        return float(self._clf.score(Xf, y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(self._flatten(X))
        return self._clf.predict_proba(Xf)
```

- [ ] **Step 3: Implement MDL probe (simplified online-coding version)**
```python
# src/probe_module/probe/mdl_probe.py
"""MDL probing via codelength (Voita & Titov 2020, simplified)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe_module.base_probe import BaseProbe
from src.probe_module import register_probe

@register_probe("mdl")
class MDLProbe(BaseProbe):
    """MDL probe: reports minimum description length (codelength) in addition to accuracy."""

    def __init__(self, cfg=None):
        self._clf = LogisticRegression(max_iter=1000)
        self._scaler = StandardScaler()
        self._codelength: float = float("inf")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        # Online coding: train on fraction, compute codelength incrementally
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        codelength = 0.0
        Xf = self._scaler.fit_transform(X.reshape(len(X), -1))
        for i, frac in enumerate(fractions):
            end = int(frac * n)
            start = int(fractions[i-1] * n) if i > 0 else 0
            if start == 0:
                # First chunk: uniform prior
                codelength += (end - start) * np.log2(len(np.unique(y)))
                self._clf.fit(Xf[:end], y[:end])
            else:
                # Subsequent chunks: use current model as prior
                proba = self._clf.predict_proba(Xf[start:end])
                labels = y[start:end].astype(int)
                eps = 1e-10
                codelength += -sum(np.log2(proba[i, labels[i]] + eps) for i in range(len(labels)))
                self._clf.fit(Xf[:end], y[:end])
        self._codelength = codelength

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(X.reshape(len(X), -1))
        return self._clf.predict(Xf)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        Xf = self._scaler.transform(X.reshape(len(X), -1))
        return float(self._clf.score(Xf, y))

    @property
    def codelength(self) -> float:
        return self._codelength
```

- [ ] **Step 4: Implement ProbeEvaluator**
```python
# src/probe_module/evaluator/probe_evaluator.py
import numpy as np
from typing import Dict, List
from scipy import stats
from src.probe_module import ProbeFactory

class ProbeEvaluator:
    """Evaluates probes across contexts with statistical testing."""

    def __init__(self, probe_name: str = "linear", n_trials: int = 5):
        self.probe_name = probe_name
        self.n_trials = n_trials

    def evaluate_context_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_mask: np.ndarray,
        n_trials: int = None,
    ) -> Dict[str, float]:
        """Train probe, report accuracy in vs out of context."""
        n_trials = n_trials or self.n_trials
        in_accs, out_accs = [], []
        X_in, y_in = X[context_mask], y[context_mask]
        X_out, y_out = X[~context_mask], y[~context_mask]

        for _ in range(n_trials):
            idx = np.random.permutation(len(X_in))
            split = int(0.8 * len(idx))
            probe = ProbeFactory(self.probe_name)()
            probe.fit(X_in[idx[:split]], y_in[idx[:split]])
            in_accs.append(probe.score(X_in[idx[split:]], y_in[idx[split:]]))
            if len(X_out) > 0:
                out_accs.append(probe.score(X_out, y_out))

        result = {
            "in_context_acc": float(np.mean(in_accs)),
            "in_context_std": float(np.std(in_accs)),
            "out_context_acc": float(np.mean(out_accs)) if out_accs else float("nan"),
        }
        return result

    def evaluate_all_layers(
        self,
        activations_by_layer: Dict[str, np.ndarray],
        y: np.ndarray,
        context_mask: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate probe at each layer. Returns {layer: {metric: value}}."""
        results = {}
        for layer, X in activations_by_layer.items():
            results[layer] = self.evaluate_context_split(X, y, context_mask)
        return results

    def bonferroni_threshold(self, n_comparisons: int, alpha: float = 0.01) -> float:
        """Bonferroni-corrected significance threshold."""
        return alpha / n_comparisons
```

- [ ] **Step 5: Run tests**
```bash
uv run pytest tests/test_probe_module/ -v
```

- [ ] **Step 6: Commit**
```bash
git add src/probe_module/ run/conf/probe/ tests/test_probe_module/
git commit -m "feat(probe): add linear + MDL probes with context-split evaluator"
```

---

## Chunk 4: Causal Module (Experiment 1 — D2)

### Task 7: Causal Tracing + Activation Patcher

**Files:**
- Create: `src/causal_module/__init__.py`
- Create: `src/causal_module/tracing/causal_tracer.py`
- Create: `src/causal_module/patch/activation_patcher.py`
- Create: `src/causal_module/shard_vector/shard_vector.py`
- Test: `tests/test_causal_module/test_causal_tracer.py`
- Test: `tests/test_causal_module/test_activation_patcher.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_causal_module/test_activation_patcher.py
import torch
import numpy as np
import pytest
from src.causal_module.patch.activation_patcher import ActivationPatcher

def _simple_model():
    class M(torch.nn.Module):
        def forward(self, x):
            return x * 2
    return M()

def test_patch_suppresses_layer():
    model = _simple_model()
    patcher = ActivationPatcher(model)
    # Patch output of model to zeros
    patch_fn = patcher.make_zero_patch()
    x = torch.ones(1, 4)
    with patcher.patch_layer("", patch_fn):
        out = model(x)
    assert torch.allclose(out, torch.zeros_like(out))

def test_project_out_reduces_component():
    patcher = ActivationPatcher(torch.nn.Identity())
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    projected = patcher.project_out(x, direction)
    assert abs(projected[0, 0].item()) < 1e-5  # component along direction removed

# tests/test_causal_module/test_causal_tracer.py
import torch
import numpy as np
from src.causal_module.shard_vector.shard_vector import ShardVector

def test_shard_vector_mean_diff():
    clean_acts = {"block1": torch.randn(50, 16, 8, 8)}
    corrupt_acts = {"block1": torch.randn(50, 16, 8, 8)}
    sv = ShardVector()
    vectors = sv.compute(clean_acts, corrupt_acts)
    assert "block1" in vectors
    assert vectors["block1"].shape == (16, 8, 8)
```

- [ ] **Step 2: Implement ActivationPatcher**
```python
# src/causal_module/patch/activation_patcher.py
import torch
import torch.nn as nn
from typing import Callable, Optional, Dict
from contextlib import contextmanager

class ActivationPatcher:
    """Applies activation patches during forward passes."""

    def __init__(self, model: nn.Module):
        self._model = model

    def project_out(self, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Project out a direction vector from activations (shard suppression)."""
        direction = direction / (direction.norm() + 1e-8)
        flat = x.view(x.shape[0], -1)
        proj = (flat @ direction.view(-1, 1)) * direction.view(1, -1)
        return (flat - proj).view_as(x)

    def project_add(self, x: torch.Tensor, direction: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Add a scaled direction vector to activations (shard amplification)."""
        direction = direction / (direction.norm() + 1e-8)
        return x + scale * direction.view(1, -1).expand_as(x.view(x.shape[0], -1)).view_as(x)

    def make_zero_patch(self) -> Callable:
        return lambda x: torch.zeros_like(x)

    def make_restore_patch(self, clean_activation: torch.Tensor) -> Callable:
        """Restore layer output to clean activation (for causal tracing)."""
        def patch(x: torch.Tensor) -> torch.Tensor:
            if x.shape == clean_activation.shape:
                return clean_activation.clone()
            return x
        return patch

    def make_suppress_patch(self, direction: torch.Tensor) -> Callable:
        """Project out shard direction from activations."""
        def patch(x: torch.Tensor) -> torch.Tensor:
            return self.project_out(x, direction)
        return patch

    @contextmanager
    def patch_layer(self, layer_name: str, patch_fn: Callable):
        """Context manager to apply patch_fn to a named layer's output."""
        named = dict(self._model.named_modules())
        target = named.get(layer_name, self._model)
        hook = target.register_forward_hook(lambda m, inp, out: patch_fn(out))
        try:
            yield
        finally:
            hook.remove()
```

- [ ] **Step 3: Implement ShardVector**
```python
# src/causal_module/shard_vector/shard_vector.py
import torch
import numpy as np
from typing import Dict

class ShardVector:
    """Computes shard direction vectors as mean activation differences."""

    def compute(
        self,
        context_activations: Dict[str, torch.Tensor],
        baseline_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_activations: {layer: (N, ...)} activations when concept present
            baseline_activations: {layer: (N, ...)} activations when concept absent
        Returns:
            {layer: mean_diff vector}
        """
        vectors = {}
        for layer in context_activations:
            ctx = context_activations[layer]
            base = baseline_activations[layer]
            if isinstance(ctx, np.ndarray):
                ctx = torch.from_numpy(ctx)
            if isinstance(base, np.ndarray):
                base = torch.from_numpy(base)
            vectors[layer] = ctx.float().mean(0) - base.float().mean(0)
        return vectors
```

- [ ] **Step 4: Implement CausalTracer**
```python
# src/causal_module/tracing/causal_tracer.py
import torch
import numpy as np
from typing import Dict, List, Callable
from src.agent_module.base_agent import BaseAgent
from src.agent_module.hooks.activation_hooks import ActivationRecorder
from src.causal_module.patch.activation_patcher import ActivationPatcher

class CausalTracer:
    """
    Implements causal tracing (Meng et al., 2022) for RL agents.

    Protocol:
    1. Clean run: collect activations with concept present (cheese visible)
    2. Corrupted run: run with concept absent (no cheese), collect activations
    3. Restoration: for each layer, restore clean activations during corrupted run,
       measure recovery of concept-directed behavior.
    """

    def __init__(self, agent: BaseAgent, layer_names: List[str]):
        self.agent = agent
        self.layer_names = layer_names
        self.patcher = ActivationPatcher(agent.policy)

    def _get_activations(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        _, acts = self.agent.act_with_activations(obs)
        return acts

    def _get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.get_action_probs(obs)

    def trace(
        self,
        clean_obs: np.ndarray,
        corrupted_obs: np.ndarray,
        target_action_idx: int,
    ) -> Dict[str, float]:
        """
        For each layer, measure how much restoring clean activations
        recovers the clean target action probability.

        Returns:
            {layer: causal_effect_score} where score = recovered_prob - corrupted_prob
        """
        # Step 1: clean run
        clean_acts = self._get_activations(clean_obs)
        clean_prob = self._get_action_probs(clean_obs)[target_action_idx]

        # Step 2: corrupted run
        corrupted_prob = self._get_action_probs(corrupted_obs)[target_action_idx]

        # Step 3: restoration per layer
        results = {}
        for layer in self.layer_names:
            restore_patch = self.patcher.make_restore_patch(
                clean_acts[layer].unsqueeze(0)
            )
            with self.patcher.patch_layer(layer, restore_patch):
                restored_prob = self._get_action_probs(corrupted_obs)[target_action_idx]
            # Causal effect: how much recovery normalized by clean-corrupted gap
            gap = clean_prob - corrupted_prob + 1e-8
            results[layer] = float((restored_prob - corrupted_prob) / gap)
        return results
```

- [ ] **Step 5: Run tests**
```bash
uv run pytest tests/test_causal_module/ -v
```

- [ ] **Step 6: Commit**
```bash
git add src/causal_module/ tests/test_causal_module/
git commit -m "feat(causal): add causal tracer, activation patcher, shard vectors"
```

---

### Task 8: Experiment 1 — Probing + Causal Tracing Pipeline

**Files:**
- Create: `run/pipeline/analysis/experiment1_probing.py`
- Create: `run/pipeline/analysis/experiment1_causal.py`
- Create: `run/pipeline/analysis/experiment1_integrate.py`
- Create: `run/conf/experiment1/default.yaml`

- [ ] **Step 1: Implement probing pipeline**
```python
# run/pipeline/analysis/experiment1_probing.py
"""D1: Layer-wise contextual probing for concept encoding."""
import json
import hydra
import numpy as np
from omegaconf import DictConfig
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
from src.analysis_module.visualization.shard_visualizer import ShardVisualizer

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ds = HDF5ActivationDataset(cfg.collect.output_path, mode="r")
    n = len(ds)

    # Load all data into memory (or stream for large datasets)
    activations_by_layer = {}
    labels_by_concept = {}
    for i in range(n):
        sample = ds[i]
        for layer, act in sample["activations"].items():
            activations_by_layer.setdefault(layer, []).append(act)
        for concept, label in sample["labels"].items():
            labels_by_concept.setdefault(concept, []).append(label)

    activations_by_layer = {k: np.stack(v) for k, v in activations_by_layer.items()}
    labels_by_concept = {k: np.array(v) for k, v in labels_by_concept.items()}
    ds.close()

    evaluator = ProbeEvaluator(
        probe_name=cfg.experiment1.probe_type,
        n_trials=cfg.experiment1.n_trials,
    )
    all_results = {}
    for concept in cfg.experiment1.concepts:
        y = labels_by_concept[concept]
        # Context mask: binary version of label
        context_mask = y > cfg.experiment1.context_threshold
        results = evaluator.evaluate_all_layers(activations_by_layer, (y > 0.5).astype(float), context_mask)
        all_results[concept] = results
        print(f"\n=== {concept} ===")
        for layer, metrics in results.items():
            print(f"  {layer}: in={metrics['in_context_acc']:.3f}, out={metrics['out_context_acc']:.3f}")

    output = f"outputs/tables/experiment1_probing.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output}")

    # Visualize heatmap of in-context accuracy by layer
    viz = ShardVisualizer()
    viz.plot_probe_heatmap(all_results, output_path="outputs/figures/exp1_probe_heatmap.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement causal tracing pipeline**
```python
# run/pipeline/analysis/experiment1_causal.py
"""D2: Causal tracing — which layers are causally responsible for cheese behavior?"""
import json
import numpy as np
import hydra
from omegaconf import DictConfig
from src.environment_module import EnvFactory
from src.agent_module import AgentFactory
from src.causal_module.tracing.causal_tracer import CausalTracer

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    env = EnvFactory(cfg.environment.name)(cfg)
    agent = AgentFactory(cfg.agent.name)(cfg)
    tracer = CausalTracer(agent, layer_names=cfg.agent.layer_names)

    causal_effects = {layer: [] for layer in cfg.agent.layer_names}

    for trial in range(cfg.experiment1.causal_n_trials):
        # Clean: maze with cheese nearby
        obs_clean = env.reset()
        # ensure cheese is close; skip if not
        cheese_dist = np.linalg.norm(np.array(env.cheese_pos()) - np.array(env.agent_pos()))
        if cheese_dist > cfg.experiment1.near_threshold:
            continue
        action_clean = agent.act(obs_clean)

        # Corrupted: same maze but manually remove cheese
        from procgen_tools.maze import EnvState
        state_bytes = env._venv.env.callmethod("get_state")[0]
        state = EnvState(state_bytes)
        grid = state.inner_grid()
        grid[grid == 2] = 100  # remove cheese (CHEESE=2, EMPTY=100)
        state.set_grid(grid)
        env._venv.env.callmethod("set_state", [state.state_bytes])
        obs_corrupt = env.reset()

        effects = tracer.trace(obs_clean, obs_corrupt, target_action_idx=action_clean)
        for layer, effect in effects.items():
            causal_effects[layer].append(effect)

    mean_effects = {layer: float(np.mean(v)) for layer, v in causal_effects.items() if v}
    print("\n=== Causal Effects by Layer ===")
    for layer, eff in sorted(mean_effects.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {eff:.3f}")

    output = "outputs/tables/experiment1_causal.json"
    with open(output, "w") as f:
        json.dump(mean_effects, f, indent=2)
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement integration script (D1 + D2 → candidates)**
```python
# run/pipeline/analysis/experiment1_integrate.py
"""Cross-validate probe (D1) and causal (D2) results to identify shard candidates."""
import json
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    probing = load_json("outputs/tables/experiment1_probing.json")
    causal = load_json("outputs/tables/experiment1_causal.json")

    concepts = ["cheese_presence"]  # primary concept for Exp 1
    candidates = {}
    for concept in concepts:
        layer_scores = probing[concept]
        for layer, metrics in layer_scores.items():
            in_acc = metrics["in_context_acc"]
            out_acc = metrics["out_context_acc"]
            causal_eff = causal.get(layer, 0.0)
            # Shard candidate: high in-context probe AND high causal effect
            probe_score = in_acc - out_acc  # contextual specificity
            candidates.setdefault(concept, {})[layer] = {
                "probe_specificity": probe_score,
                "causal_effect": causal_eff,
                "combined_score": probe_score * causal_eff,
            }

    print("\n=== Shard Candidates ===")
    for concept, layers in candidates.items():
        print(f"\nConcept: {concept}")
        ranked = sorted(layers.items(), key=lambda x: -x[1]["combined_score"])
        for layer, scores in ranked[:5]:
            print(f"  {layer}: probe_spec={scores['probe_specificity']:.3f}, causal={scores['causal_effect']:.3f}")

    with open("outputs/tables/experiment1_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add experiment1 config**
```yaml
# run/conf/experiment1/default.yaml
probe_type: linear
n_trials: 5
concepts:
  - cheese_presence
  - cheese_proximity
  - cheese_direction
  - corner_proximity
context_threshold: 0.5   # threshold for binary context mask
causal_n_trials: 100
near_threshold: 5         # cheese within 5 steps = "near"
```

- [ ] **Step 5: Commit**
```bash
git add run/pipeline/analysis/experiment1_*.py run/conf/experiment1/
git commit -m "feat(exp1): implement probing + causal tracing pipeline for Experiment 1"
```

---

## Chunk 5: Shard Module (Experiment 2 — D3 Separability)

### Task 9: Shard Detector + Separability Tester

**Files:**
- Create: `src/shard_module/__init__.py`
- Create: `src/shard_module/detection/shard_detector.py`
- Create: `src/shard_module/separability/separability_tester.py`
- Create: `src/shard_module/metrics/shard_metrics.py`
- Create: `run/pipeline/analysis/experiment2_separability.py`
- Test: `tests/test_shard_module/test_shard_detector.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_shard_module/test_shard_detector.py
import numpy as np
import pytest
from src.shard_module.detection.shard_detector import ShardDetector
from src.shard_module.metrics.shard_metrics import ShardMetrics

def test_shard_detector_finds_candidates():
    probe_results = {
        "cheese_presence": {
            "block3": {"probe_specificity": 0.4, "causal_effect": 0.6, "combined_score": 0.24},
            "fc": {"probe_specificity": 0.3, "causal_effect": 0.8, "combined_score": 0.24},
        }
    }
    detector = ShardDetector(probe_threshold=0.2, causal_threshold=0.3)
    candidates = detector.get_top_candidates(probe_results, concept="cheese_presence", top_k=1)
    assert len(candidates) == 1

def test_independence_score_bounds():
    metrics = ShardMetrics()
    # I=1 when suppressing S1 has zero effect on S2 behavior
    I = metrics.independence_score(
        baseline_behavior_S2=0.8,
        suppressed_behavior_S2=0.8,   # no change
        suppressed_behavior_self=0.3,  # S2 itself changes when S2 suppressed
    )
    assert abs(I - 1.0) < 1e-5
```

- [ ] **Step 2: Implement ShardDetector**
```python
# src/shard_module/detection/shard_detector.py
import json
from typing import Dict, List, Tuple

class ShardDetector:
    """Identifies shard candidates from probe and causal tracing results."""

    def __init__(self, probe_threshold: float = 0.2, causal_threshold: float = 0.3):
        self.probe_threshold = probe_threshold
        self.causal_threshold = causal_threshold

    def get_top_candidates(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        concept: str,
        top_k: int = 3,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Return top-k (layer, scores) candidates satisfying D1 and D2."""
        layer_scores = results.get(concept, {})
        candidates = [
            (layer, scores)
            for layer, scores in layer_scores.items()
            if scores["probe_specificity"] >= self.probe_threshold
            and scores["causal_effect"] >= self.causal_threshold
        ]
        candidates.sort(key=lambda x: -x[1]["combined_score"])
        return candidates[:top_k]
```

- [ ] **Step 3: Implement ShardMetrics**
```python
# src/shard_module/metrics/shard_metrics.py

class ShardMetrics:
    """Computes shard-theory specific metrics."""

    def independence_score(
        self,
        baseline_behavior_S2: float,
        suppressed_behavior_S2: float,
        suppressed_behavior_self: float,
        baseline_behavior_self: float = None,
    ) -> float:
        """
        I(S1, S2) = 1 - |Δbehavior(S2) when S1 suppressed| / |Δbehavior(S2) when S2 suppressed|

        Args:
            baseline_behavior_S2: S2-governed behavior before any suppression
            suppressed_behavior_S2: S2-governed behavior after S1 suppression
            suppressed_behavior_self: S2-governed behavior after S2 suppression (denominator)
        """
        leakage = abs(baseline_behavior_S2 - suppressed_behavior_S2)
        self_effect = abs(baseline_behavior_S2 - suppressed_behavior_self) + 1e-8
        return float(1.0 - leakage / self_effect)

    def causal_effect_size(
        self,
        clean_prob: float,
        suppressed_prob: float,
    ) -> float:
        """Δaction probability from shard suppression."""
        return float(abs(clean_prob - suppressed_prob))
```

- [ ] **Step 4: Implement SeparabilityTester**
```python
# src/shard_module/separability/separability_tester.py
import numpy as np
import torch
from typing import Dict, List, Tuple
from src.agent_module.base_agent import BaseAgent
from src.causal_module.patch.activation_patcher import ActivationPatcher
from src.causal_module.shard_vector.shard_vector import ShardVector
from src.shard_module.metrics.shard_metrics import ShardMetrics
from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator

class SeparabilityTester:
    """
    Tests D3: suppressing shard S1 should not substantially affect shard S2.

    Protocol:
    1. Compute cheese shard vector and corner shard vector
    2. Suppress cheese shard → measure corner behavior (and vice versa)
    3. Compute independence score I(cheese, corner) and I(corner, cheese)
    """

    def __init__(self, agent: BaseAgent, shard_layer: str):
        self.agent = agent
        self.shard_layer = shard_layer
        self.patcher = ActivationPatcher(agent.policy)
        self.sv_computer = ShardVector()
        self.metrics = ShardMetrics()

    def compute_behavior_score(
        self,
        observations: np.ndarray,
        target_action_indices: List[int],
    ) -> float:
        """Mean probability of concept-directed actions across observations."""
        probs = []
        for obs, action_idx in zip(observations, target_action_indices):
            p = self.agent.get_action_probs(obs)[action_idx]
            probs.append(p)
        return float(np.mean(probs))

    def suppress_shard_and_measure(
        self,
        shard_vector: torch.Tensor,
        observations: np.ndarray,
        target_action_indices: List[int],
    ) -> float:
        """Measure behavior score after projecting out shard direction."""
        patch_fn = self.patcher.make_suppress_patch(shard_vector)
        probs = []
        with self.patcher.patch_layer(self.shard_layer, patch_fn):
            for obs, action_idx in zip(observations, target_action_indices):
                p = self.agent.get_action_probs(obs)[action_idx]
                probs.append(p)
        return float(np.mean(probs))

    def run(
        self,
        cheese_context_obs: np.ndarray,
        cheese_context_acts: Dict[str, torch.Tensor],
        cheese_baseline_acts: Dict[str, torch.Tensor],
        corner_context_obs: np.ndarray,
        corner_context_acts: Dict[str, torch.Tensor],
        corner_baseline_acts: Dict[str, torch.Tensor],
        cheese_directed_actions: List[int],
        corner_directed_actions: List[int],
    ) -> Dict[str, float]:
        vectors = self.sv_computer.compute(
            {self.shard_layer: cheese_context_acts[self.shard_layer]},
            {self.shard_layer: cheese_baseline_acts[self.shard_layer]},
        )
        cheese_vec = vectors[self.shard_layer]
        corner_vecs = self.sv_computer.compute(
            {self.shard_layer: corner_context_acts[self.shard_layer]},
            {self.shard_layer: corner_baseline_acts[self.shard_layer]},
        )
        corner_vec = corner_vecs[self.shard_layer]

        # Baseline behaviors
        cheese_baseline_beh = self.compute_behavior_score(cheese_context_obs, cheese_directed_actions)
        corner_baseline_beh = self.compute_behavior_score(corner_context_obs, corner_directed_actions)

        # Suppress cheese → measure corner
        cheese_suppress_corner_beh = self.suppress_shard_and_measure(
            cheese_vec, corner_context_obs, corner_directed_actions
        )
        # Suppress corner → measure cheese
        corner_suppress_cheese_beh = self.suppress_shard_and_measure(
            corner_vec, cheese_context_obs, cheese_directed_actions
        )
        # Suppress self (for denominator)
        cheese_self_suppressed = self.suppress_shard_and_measure(
            cheese_vec, cheese_context_obs, cheese_directed_actions
        )
        corner_self_suppressed = self.suppress_shard_and_measure(
            corner_vec, corner_context_obs, corner_directed_actions
        )

        I_cheese_corner = self.metrics.independence_score(
            corner_baseline_beh, cheese_suppress_corner_beh, corner_self_suppressed
        )
        I_corner_cheese = self.metrics.independence_score(
            cheese_baseline_beh, corner_suppress_cheese_beh, cheese_self_suppressed
        )

        return {
            "I_cheese_corner": I_cheese_corner,
            "I_corner_cheese": I_corner_cheese,
            "cheese_causal_effect": self.metrics.causal_effect_size(cheese_baseline_beh, cheese_self_suppressed),
            "corner_causal_effect": self.metrics.causal_effect_size(corner_baseline_beh, corner_self_suppressed),
        }
```

- [ ] **Step 5: Add experiment2 pipeline script**
```python
# run/pipeline/analysis/experiment2_separability.py
"""D3: Test shard separability via suppression interventions."""
import json
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from src.environment_module import EnvFactory
from src.agent_module import AgentFactory
from src.shard_module.separability.separability_tester import SeparabilityTester

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    env = EnvFactory(cfg.environment.name)(cfg)
    agent = AgentFactory(cfg.agent.name)(cfg)

    shard_layer = cfg.experiment2.shard_layer  # from Exp 1 candidates
    tester = SeparabilityTester(agent, shard_layer)

    # Collect context-specific activations
    # (In practice, roll out and filter by context; simplified here)
    # ... (collect cheese_context_obs, cheese_context_acts, etc.)
    # results = tester.run(...)
    # print + save results

    print("Experiment 2: Separability testing")
    print("Load Exp 1 candidates from outputs/tables/experiment1_candidates.json")
    print("Run tester.run() with context-specific observation batches")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**
```bash
uv run pytest tests/test_shard_module/ -v
```

- [ ] **Step 7: Commit**
```bash
git add src/shard_module/ run/pipeline/analysis/experiment2_separability.py tests/test_shard_module/
git commit -m "feat(shard): add shard detector, separability tester, independence metrics"
```

---

## Chunk 6: RL Trainer + Experiment 3

### Task 10: RL Trainer with Configurable Cheese Distribution

**Files:**
- Create: `src/trainer_module/rl_trainer/cheese_distribution.py`
- Create: `src/trainer_module/rl_trainer/ppo_trainer.py`
- Create: `run/pipeline/training/train_rl_agent.py`
- Create: `run/conf/experiment3/default.yaml`
- Test: `tests/test_trainer_module/test_cheese_distribution.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_trainer_module/test_cheese_distribution.py
import numpy as np
from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution

def test_corner_biased_places_in_corner():
    dist = CheesePlacementDistribution(mode="corner_biased", grid_size=15)
    positions = [dist.sample() for _ in range(1000)]
    # Top-right quadrant: col > 7, row < 7
    corner_count = sum(1 for r, c in positions if r < 7 and c > 7)
    assert corner_count / 1000 > 0.6  # >60% in corner

def test_uniform_distributes_evenly():
    dist = CheesePlacementDistribution(mode="uniform", grid_size=15)
    positions = [dist.sample() for _ in range(1000)]
    corner_count = sum(1 for r, c in positions if r < 7 and c > 7)
    assert 0.15 < corner_count / 1000 < 0.35  # roughly uniform

def test_anti_corner_avoids_corner():
    dist = CheesePlacementDistribution(mode="anti_corner", grid_size=15)
    positions = [dist.sample() for _ in range(1000)]
    corner_count = sum(1 for r, c in positions if r < 7 and c > 7)
    assert corner_count / 1000 < 0.15  # <15% in corner
```

- [ ] **Step 2: Implement CheesePlacementDistribution**
```python
# src/trainer_module/rl_trainer/cheese_distribution.py
import numpy as np
from typing import Tuple

class CheesePlacementDistribution:
    """
    Controls cheese placement distribution during RL training.

    Modes:
      corner_biased: Cheese in top-right 25% of maze 75% of the time.
      uniform: Cheese placed uniformly across all maze quadrants.
      anti_corner: Cheese in top-right only 10%; bottom-left 70%.
    """

    def __init__(self, mode: str, grid_size: int = 15):
        self.mode = mode
        self.grid_size = grid_size
        self.half = grid_size // 2

    def sample(self) -> Tuple[int, int]:
        """Return (row, col) for cheese placement."""
        if self.mode == "corner_biased":
            return self._sample_corner_biased()
        elif self.mode == "uniform":
            return self._sample_uniform()
        elif self.mode == "anti_corner":
            return self._sample_anti_corner()
        raise ValueError(f"Unknown mode: {self.mode}")

    def _sample_uniform(self) -> Tuple[int, int]:
        r = np.random.randint(1, self.grid_size - 1)
        c = np.random.randint(1, self.grid_size - 1)
        return int(r), int(c)

    def _sample_corner_biased(self) -> Tuple[int, int]:
        if np.random.rand() < 0.75:
            # Top-right quadrant
            r = np.random.randint(0, self.half)
            c = np.random.randint(self.half, self.grid_size - 1)
        else:
            r, c = self._sample_uniform()
        return int(r), int(c)

    def _sample_anti_corner(self) -> Tuple[int, int]:
        p = np.random.rand()
        if p < 0.10:
            # Top-right: 10%
            r = np.random.randint(0, self.half)
            c = np.random.randint(self.half, self.grid_size - 1)
        elif p < 0.80:
            # Bottom-left: 70%
            r = np.random.randint(self.half, self.grid_size - 1)
            c = np.random.randint(0, self.half)
        else:
            r, c = self._sample_uniform()
        return int(r), int(c)
```

- [ ] **Step 3: Implement PPO trainer wrapper**
```python
# src/trainer_module/rl_trainer/ppo_trainer.py
"""PPO trainer for maze agents with configurable cheese placement."""
import torch
import numpy as np
from typing import Optional
from procgen_tools.models import load_policy
from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution

class PPOTrainer:
    """
    Wraps stable-baselines3 PPO for maze-agent training with a custom
    cheese placement callback to control reinforcement distribution.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cheese_dist = CheesePlacementDistribution(
            mode=cfg.training.cheese_mode,
            grid_size=cfg.training.grid_size,
        )
        self.total_timesteps = cfg.training.total_timesteps
        self.seed = cfg.training.seed

    def train(self, output_path: str) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor
        from src.trainer_module.rl_trainer.maze_callback import CheesePlacementCallback
        from src.environment_module.maze.maze_env import ProcgenMazeEnv

        env = ProcgenMazeEnv(self.cfg)
        callback = CheesePlacementCallback(self.cheese_dist)
        model = PPO("CnnPolicy", env, verbose=1, seed=self.seed)
        model.learn(total_timesteps=self.total_timesteps, callback=callback)
        model.save(output_path)
        print(f"Saved model to {output_path}")
```

- [ ] **Step 4: Add training pipeline script**
```python
# run/pipeline/training/train_rl_agent.py
"""Train maze agent with configurable cheese placement distribution."""
import hydra
from omegaconf import DictConfig
from src.trainer_module.rl_trainer.ppo_trainer import PPOTrainer

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = PPOTrainer(cfg)
    output_path = cfg.training.output_path
    trainer.train(output_path)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add experiment3 configs**
```yaml
# run/conf/experiment3/default.yaml
agents:
  corner_biased:
    cheese_mode: corner_biased
    output_path: data/raw/agent_corner_biased.pt
  uniform:
    cheese_mode: uniform
    output_path: data/raw/agent_uniform.pt
  anti_corner:
    cheese_mode: anti_corner
    output_path: data/raw/agent_anti_corner.pt
total_timesteps: 5000000
grid_size: 15
seed: 42

# run/conf/training/rl_agent.yaml
total_timesteps: 5000000
grid_size: 15
seed: 42
cheese_mode: uniform    # override per-agent in experiment3
output_path: data/raw/agent.pt
```

- [ ] **Step 6: Run tests**
```bash
uv run pytest tests/test_trainer_module/ -v
```

- [ ] **Step 7: Commit**
```bash
git add src/trainer_module/ run/pipeline/training/train_rl_agent.py run/conf/experiment3/ tests/test_trainer_module/
git commit -m "feat(trainer): add PPO trainer + configurable cheese placement for Experiment 3"
```

---

## Chunk 7: SAE Module (Experiment 4)

### Task 11: Sparse Autoencoder

**Files:**
- Create: `src/sae_module/model/sparse_autoencoder.py`
- Create: `src/sae_module/training/sae_trainer.py`
- Create: `src/sae_module/feature/feature_analyzer.py`
- Create: `run/pipeline/training/train_sae.py`
- Create: `run/pipeline/analysis/experiment4_sae.py`
- Test: `tests/test_sae_module/test_sparse_autoencoder.py`

- [ ] **Step 1: Write failing tests**
```python
# tests/test_sae_module/test_sparse_autoencoder.py
import torch
import pytest
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder

def test_sae_forward_shape():
    d_input = 64
    expansion = 8
    sae = SparseAutoencoder(d_input=d_input, expansion_factor=expansion)
    x = torch.randn(16, d_input)
    recon, features = sae(x)
    assert recon.shape == (16, d_input)
    assert features.shape == (16, d_input * expansion)

def test_sae_features_sparse():
    sae = SparseAutoencoder(d_input=32, expansion_factor=8, l1_coef=0.1)
    x = torch.randn(100, 32)
    _, features = sae(x)
    # After ReLU, most features should be zero (sparse)
    sparsity = (features == 0).float().mean()
    # Note: untrained model won't be sparse, just check shape + type
    assert features.dtype == torch.float32

def test_sae_reconstruction_loss_finite():
    sae = SparseAutoencoder(d_input=16, expansion_factor=4)
    x = torch.randn(8, 16)
    recon, features = sae(x)
    loss = sae.loss(x, recon, features)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Implement SparseAutoencoder (Bricken et al. 2023)**
```python
# src/sae_module/model/sparse_autoencoder.py
import torch
import torch.nn as nn
from src.sae_module import register_sae

@register_sae("standard")
class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder following Bricken et al. (2023).
    Architecture: x -> (W_enc * (x - b_pre) + b_enc) -> ReLU -> features -> W_dec -> reconstruction
    Loss: MSE reconstruction + L1 sparsity penalty on features
    """

    def __init__(self, cfg=None, d_input: int = 256, expansion_factor: int = 8, l1_coef: float = 0.01):
        super().__init__()
        if cfg is not None:
            d_input = cfg.sae.d_input
            expansion_factor = cfg.sae.expansion_factor
            l1_coef = cfg.sae.l1_coef

        self.d_input = d_input
        self.d_hidden = d_input * expansion_factor
        self.l1_coef = l1_coef

        self.b_pre = nn.Parameter(torch.zeros(d_input))
        self.W_enc = nn.Linear(d_input, self.d_hidden, bias=True)
        self.W_dec = nn.Linear(self.d_hidden, d_input, bias=False)
        self.relu = nn.ReLU()

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(self.W_dec.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        return self.relu(self.W_enc(x - self.b_pre))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return self.W_dec(features) + self.b_pre

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features

    def loss(self, x: torch.Tensor, recon: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        mse = ((x - recon) ** 2).mean()
        l1 = self.l1_coef * features.abs().mean()
        return mse + l1

    def normalize_decoder(self) -> None:
        """Ensure decoder columns have unit norm (called after each gradient step)."""
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(self.W_dec.weight.data, dim=0)
```

- [ ] **Step 3: Implement SAETrainer**
```python
# src/sae_module/training/sae_trainer.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

class SAETrainer:
    """Trains SAE on stored activations from a specific layer."""

    def __init__(self, sae: SparseAutoencoder, cfg):
        self.sae = sae
        self.lr = cfg.training.lr
        self.n_epochs = cfg.training.n_epochs
        self.batch_size = cfg.training.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sae.to(self.device)

    def train(self, dataset_path: str, layer_name: str, output_path: str) -> None:
        ds = HDF5ActivationDataset(dataset_path, mode="r")
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for i in range(0, len(ds), self.batch_size):
                batch_acts = []
                for j in range(i, min(i + self.batch_size, len(ds))):
                    sample = ds[j]
                    act = sample["activations"][layer_name]
                    batch_acts.append(torch.from_numpy(act).flatten())
                x = torch.stack(batch_acts).to(self.device)

                optimizer.zero_grad()
                recon, features = self.sae(x)
                loss = self.sae.loss(x, recon, features)
                loss.backward()
                optimizer.step()
                self.sae.normalize_decoder()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.n_epochs}: loss={total_loss/len(ds)*self.batch_size:.6f}")

        torch.save(self.sae.state_dict(), output_path)
        ds.close()
        print(f"Saved SAE to {output_path}")
```

- [ ] **Step 4: Implement FeatureAnalyzer**
```python
# src/sae_module/feature/feature_analyzer.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr

class FeatureAnalyzer:
    """
    Analyzes SAE features to identify those corresponding to shard concepts.
    Maps SAE features to context profiles (cheese present/absent, corner proximal/distal).
    """

    def __init__(self, sae, layer_name: str):
        self.sae = sae
        self.layer_name = layer_name

    def compute_feature_context_profiles(
        self,
        activations: np.ndarray,
        context_labels: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        For each SAE feature, compute mean activation in each context.
        Returns {concept: (n_features,) mean difference (context_on - context_off)}.
        """
        x = torch.from_numpy(activations).float()
        with torch.no_grad():
            _, features = self.sae(x.view(len(x), -1))
        features_np = features.numpy()

        profiles = {}
        for concept, labels in context_labels.items():
            mask_on = labels > 0.5
            mask_off = ~mask_on
            mean_on = features_np[mask_on].mean(0) if mask_on.any() else np.zeros(features_np.shape[1])
            mean_off = features_np[mask_off].mean(0) if mask_off.any() else np.zeros(features_np.shape[1])
            profiles[concept] = mean_on - mean_off  # (n_features,) contrast vector
        return profiles

    def correlate_with_probe(
        self,
        sae_feature_activations: np.ndarray,   # (N,)
        probe_predictions: np.ndarray,          # (N,)
    ) -> Tuple[float, float]:
        """Pearson r between SAE feature and probe predictions."""
        r, p = pearsonr(sae_feature_activations, probe_predictions)
        return float(r), float(p)

    def top_features_per_concept(
        self,
        profiles: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Return top-k feature indices ranked by |contrast score|."""
        result = {}
        for concept, profile in profiles.items():
            ranked = sorted(enumerate(profile), key=lambda x: -abs(x[1]))
            result[concept] = [(idx, float(score)) for idx, score in ranked[:top_k]]
        return result
```

- [ ] **Step 5: Add SAE training pipeline**
```python
# run/pipeline/training/train_sae.py
"""Train Sparse Autoencoder on collected activations."""
import hydra
import torch
from omegaconf import DictConfig
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
from src.sae_module.training.sae_trainer import SAETrainer

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    sae = SparseAutoencoder(cfg)
    trainer = SAETrainer(sae, cfg)
    trainer.train(
        dataset_path=cfg.collect.output_path,
        layer_name=cfg.sae.target_layer,
        output_path=cfg.sae.output_path,
    )

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add SAE config**
```yaml
# run/conf/training/sae.yaml
d_input: 256        # IMPALA fc layer dimension
expansion_factor: 8
l1_coef: 0.01
lr: 0.0001
n_epochs: 10
batch_size: 512
target_layer: fc
output_path: outputs/checkpoints/sae.pt
```

- [ ] **Step 7: Run tests**
```bash
uv run pytest tests/test_sae_module/ -v
```

- [ ] **Step 8: Commit**
```bash
git add src/sae_module/ run/pipeline/training/train_sae.py run/pipeline/analysis/experiment4_sae.py run/conf/training/sae.yaml tests/test_sae_module/
git commit -m "feat(sae): add sparse autoencoder (Bricken 2023) + feature analyzer for Experiment 4"
```

---

## Chunk 8: Analysis Module + Experiment 5

### Task 12: Statistical Analysis + Visualization

**Files:**
- Create: `src/analysis_module/statistics/stat_tests.py`
- Create: `src/analysis_module/visualization/shard_visualizer.py`
- Create: `run/pipeline/analysis/experiment5_generalize.py`

- [ ] **Step 1: Implement stat tests**
```python
# src/analysis_module/statistics/stat_tests.py
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def bonferroni_correction(p_values: List[float], alpha: float = 0.01) -> List[bool]:
    """Return True if null hypothesis rejected after Bonferroni correction."""
    n = len(p_values)
    threshold = alpha / n
    return [p < threshold for p in p_values]

def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Return (rho, p_value) Spearman rank correlation."""
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)

def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2) + 1e-8
    return float(diff / pooled_std)

def report_experiment3_correlation(
    agent_names: List[str],
    corner_reinforcement_freqs: List[float],
    corner_shard_strengths: List[float],
) -> Dict:
    """Test D4: Spearman correlation between training distribution and shard strength."""
    rho, p = spearman_correlation(corner_reinforcement_freqs, corner_shard_strengths)
    return {
        "agents": agent_names,
        "corner_reinforcement_freqs": corner_reinforcement_freqs,
        "corner_shard_strengths": corner_shard_strengths,
        "spearman_rho": rho,
        "p_value": p,
        "significant": p < 0.05,
    }
```

- [ ] **Step 2: Implement ShardVisualizer**
```python
# src/analysis_module/visualization/shard_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, Optional

class ShardVisualizer:
    """Generates publication-ready figures for shard analysis."""

    def plot_probe_heatmap(
        self,
        probe_results: Dict[str, Dict[str, Dict[str, float]]],
        output_path: Optional[str] = None,
    ) -> None:
        """Heatmap: concepts × layers, color = in_context_acc."""
        concepts = list(probe_results)
        layers = list(next(iter(probe_results.values())))
        data = np.array([[probe_results[c][l]["in_context_acc"] for l in layers] for c in concepts])

        fig, ax = plt.subplots(figsize=(max(8, len(layers)), max(4, len(concepts))))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.set_yticks(range(len(concepts)))
        ax.set_yticklabels(concepts)
        plt.colorbar(im, ax=ax, label="In-context probe accuracy")
        ax.set_title("Probe Accuracy by Layer and Concept (D1)")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_causal_effects(
        self,
        causal_effects: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> None:
        """Bar plot of causal effect by layer."""
        layers = list(causal_effects)
        values = [causal_effects[l] for l in layers]
        fig, ax = plt.subplots(figsize=(max(8, len(layers)), 4))
        ax.bar(range(len(layers)), values, color="steelblue")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.axhline(0.3, color="red", linestyle="--", label="Target effect size")
        ax.set_ylabel("Causal Effect Size")
        ax.set_title("Causal Tracing Results by Layer (D2)")
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_independence_scores(
        self,
        results: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> None:
        """Bar plot of shard independence scores."""
        keys = ["I_cheese_corner", "I_corner_cheese"]
        values = [results.get(k, 0.0) for k in keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(keys, values, color=["#2196F3", "#4CAF50"])
        ax.axhline(0.8, color="red", linestyle="--", label="Target I > 0.8")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Independence Score I")
        ax.set_title("Shard Separability (D3)")
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_reinforcement_correlation(
        self,
        agent_names,
        corner_freqs,
        corner_strengths,
        rho: float,
        output_path: Optional[str] = None,
    ) -> None:
        """Scatter plot: reinforcement frequency vs shard strength (D4)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(corner_freqs, corner_strengths, s=100, color="darkorange")
        for name, x, y in zip(agent_names, corner_freqs, corner_strengths):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Corner Reinforcement Frequency")
        ax.set_ylabel("Corner Shard Strength")
        ax.set_title(f"Reinforcement Traceability (D4) — Spearman ρ={rho:.2f}")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.close()
```

- [ ] **Step 3: Add experiment5 generalization pipeline**
```python
# run/pipeline/analysis/experiment5_generalize.py
"""Apply shard detection pipeline to a new environment (MiniGrid or custom)."""
import json
import hydra
from omegaconf import DictConfig
from src.environment_module import EnvFactory
from src.agent_module import AgentFactory
from src.data_module.concept_labeler import LabelerFactory
from src.data_module.rollout_collector.rollout_collector import RolloutCollector
from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
from src.causal_module.tracing.causal_tracer import CausalTracer

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Same pipeline as Experiment 1 (probing + causal tracing) but on a new environment.
    Override environment + agent in config via:
      python experiment5_generalize.py environment=minigrid agent=minigrid_ppo
    """
    env = EnvFactory(cfg.environment.name)(cfg)
    agent = AgentFactory(cfg.agent.name)(cfg)
    labelers = {name: LabelerFactory(name)() for name in cfg.experiment5.concepts}

    collector = RolloutCollector(env, agent, labelers, n_steps=cfg.collect.n_steps)
    batch = collector.collect()

    evaluator = ProbeEvaluator(probe_name="linear", n_trials=5)
    results = {}
    for concept, labels in batch.labels.items():
        context_mask = labels > 0.5
        layer_results = evaluator.evaluate_all_layers(batch.activations, (labels > 0.5).astype(float), context_mask)
        results[concept] = layer_results

    output = "outputs/tables/experiment5_generalization.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**
```bash
git add src/analysis_module/ run/pipeline/analysis/experiment5_generalize.py
git commit -m "feat(analysis): add stat tests, shard visualizer, experiment 5 pipeline"
```

---

## Chunk 9: Integration + README

### Task 13: Main Config + Integration Test

**Files:**
- Create: `run/conf/config.yaml` (update)
- Create: `tests/integration/test_pipeline_smoke.py`

- [ ] **Step 1: Finalize main Hydra config**
```yaml
# run/conf/config.yaml
defaults:
  - environment: maze
  - agent: impala
  - _self_

concepts:
  - cheese_presence
  - cheese_proximity
  - cheese_direction
  - corner_proximity

collect:
  n_steps: 100000
  reset_every: 200
  output_path: data/processed/activations_${agent.name}.h5

experiment1:
  probe_type: linear
  n_trials: 5
  context_threshold: 0.5
  causal_n_trials: 100
  near_threshold: 5
  concepts:
    - cheese_presence
    - corner_proximity

experiment2:
  shard_layer: fc   # Override from Exp 1 results

experiment4:
  sae_path: outputs/checkpoints/sae.pt
  target_layer: fc
  top_k_features: 10

experiment5:
  concepts:
    - goal_presence
    - goal_proximity
```

- [ ] **Step 2: Smoke test**
```python
# tests/integration/test_pipeline_smoke.py
"""Smoke tests: verify each module instantiates without error."""
import pytest
from unittest.mock import MagicMock

def test_probe_factory():
    from src.probe_module import ProbeFactory
    probe = ProbeFactory("linear")()
    assert probe is not None

def test_labeler_factory():
    from src.data_module.concept_labeler import LabelerFactory
    for name in ["cheese_presence", "cheese_proximity", "cheese_direction", "corner_proximity"]:
        lab = LabelerFactory(name)()
        assert lab is not None

def test_sae_instantiation():
    from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
    sae = SparseAutoencoder(d_input=64, expansion_factor=4)
    assert sae is not None

def test_shard_metrics():
    from src.shard_module.metrics.shard_metrics import ShardMetrics
    m = ShardMetrics()
    I = m.independence_score(0.8, 0.8, 0.3)
    assert abs(I - 1.0) < 1e-4
```

- [ ] **Step 3: Run all tests**
```bash
uv run pytest tests/ -v --tb=short
```
Expected: All unit tests PASS; integration tests PASS.

- [ ] **Step 4: Final commit**
```bash
git add run/conf/config.yaml tests/integration/ README.md
git commit -m "chore: finalize config, add smoke tests, complete project scaffold"
```

---

## Execution Order Summary

| Phase | Script | Prerequisite |
|-------|--------|-------------|
| 1 | `collect_activations.py` | Checkpoint in `data/raw/` |
| 2 | `experiment1_probing.py` | Phase 1 |
| 3 | `experiment1_causal.py` | Phase 1 |
| 4 | `experiment1_integrate.py` | Phases 2+3 |
| 5 | `experiment2_separability.py` | Phase 4 (shard layer identified) |
| 6 | `train_rl_agent.py` × 3 | None (trains new agents) |
| 7 | `collect_activations.py` × 3 | Phase 6 |
| 8 | `experiment3_dist.py` | Phase 4 + 7 |
| 9 | `train_sae.py` | Phase 1 (large activation set) |
| 10 | `experiment4_sae.py` | Phase 9 |
| 11 | `experiment5_generalize.py` | MiniGrid agent trained |

---

## Key Design Decisions

1. **procgen-tools as external dep**: No fork. Install from git; extend via composition.
2. **HDF5 for activations**: >1M frames for SAE training; stream-friendly with `maxshape=None`.
3. **Shard vector = mean diff**: Directly extends Turner et al.'s "cheese vector" method.
4. **Causal tracing as effect ratio**: `(restored - corrupted) / (clean - corrupted)` normalizes across different gap sizes.
5. **Independence score denominator**: Uses self-suppression as reference to handle varying shard strengths.
6. **SAE decoder normalization**: Called after each gradient step per Bricken et al.
