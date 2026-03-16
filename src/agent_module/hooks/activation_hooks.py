"""ActivationRecorder — registers forward hooks to capture intermediate activations."""
from contextlib import contextmanager
from typing import Dict, Generator, List

import torch
import torch.nn as nn


class ActivationRecorder:
    """Records named layer activations during a forward pass.

    Usage:
        recorder = ActivationRecorder(model, ["block1", "fc"])
        with recorder.record() as activations:
            model(obs)
        # activations: {"block1": tensor, "fc": tensor}
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self._model = model
        self._layer_names = layer_names

    @contextmanager
    def record(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Context manager: runs hooks during forward pass, yields activations dict."""
        activations: Dict[str, torch.Tensor] = {}
        named = dict(self._model.named_modules())
        hooks = []

        for name in self._layer_names:
            if name not in named:
                continue
            module = named[name]

            def _make_hook(layer_name: str):
                def hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
                    activations[layer_name] = out.detach().cpu()
                return hook

            hooks.append(module.register_forward_hook(_make_hook(name)))

        try:
            yield activations
        finally:
            for h in hooks:
                h.remove()
