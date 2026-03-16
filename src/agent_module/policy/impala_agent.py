"""ImpalaAgent — loads the IMPALA policy from a procgen-tools checkpoint.

Requires procgen-tools (Python 3.10). Gracefully raises ImportError if unavailable.
"""
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.agent_module import register_agent
from src.agent_module.base_agent import BaseAgent


@register_agent("impala")
class ImpalaAgent(BaseAgent):
    """IMPALA policy agent wrapping the procgen-tools model.

    Args:
        cfg: Config with cfg.agent.{checkpoint_path, layer_names}
    """

    def __init__(self, cfg: Any) -> None:
        agent_cfg = cfg.agent
        self._layer_names: List[str] = list(getattr(agent_cfg, "layer_names", []))
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = getattr(agent_cfg, "checkpoint_path", None)
        if checkpoint_path:
            self.load(checkpoint_path)

    def load(self, checkpoint_path: str) -> None:
        """Load IMPALA model from checkpoint."""
        try:
            from procgen_tools.models import load_policy  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "procgen-tools not installed. ImpalaAgent requires Python 3.10 env."
            ) from e

        self._model = load_policy(checkpoint_path, device=self._device)
        assert self._model is not None
        self._model.eval()

    def act(self, obs: np.ndarray) -> int:
        """Select greedy action given observation."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)
            logits = self._model(obs_t)["pi_logits"]
            return int(logits.argmax(dim=-1).item())

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names
