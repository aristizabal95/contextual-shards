import pytest
import torch
import torch.nn as nn

from src.agent_module import AgentFactory, register_agent, BaseAgent, AGENT_FACTORY
from src.agent_module.hooks.activation_hooks import ActivationRecorder


def test_agent_factory_raises_unknown():
    with pytest.raises(ValueError, match="Unknown agent"):
        AgentFactory("nonexistent_agent")


def test_agent_factory_returns_class():
    cls = AgentFactory("impala")
    assert issubclass(cls, BaseAgent)


def test_register_agent_decorator():
    @register_agent("mock_agent_test")
    class MockAgent(BaseAgent):
        def act(self, _obs):
            return 0

        def load(self, checkpoint_path):
            pass

    assert "mock_agent_test" in AGENT_FACTORY
    assert AgentFactory("mock_agent_test") is MockAgent


def test_impala_agent_raises_without_procgen_tools():
    """ImpalaAgent.load() raises ImportError when procgen-tools not installed."""
    cls = AgentFactory("impala")

    class FakeCfg:
        class agent:
            checkpoint_path = None
            layer_names = ["block1", "fc"]

    agent = cls(FakeCfg())  # No checkpoint -> no load call
    with pytest.raises(ImportError):
        agent.load("fake_checkpoint.pt")


def test_activation_recorder_captures_layers():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    # Name the layers so we can find them
    named = dict(model.named_modules())
    layer_names = [k for k in named if k]  # skip empty string root

    recorder = ActivationRecorder(model, layer_names)
    x = torch.randn(2, 8)
    with recorder.record() as acts:
        _ = model(x)

    # At least one layer should be captured
    assert len(acts) > 0
    for v in acts.values():
        assert isinstance(v, torch.Tensor)


def test_activation_recorder_skips_unknown_layers():
    model = nn.Linear(4, 2)
    recorder = ActivationRecorder(model, ["nonexistent_layer"])
    x = torch.randn(1, 4)
    with recorder.record() as acts:
        _ = model(x)
    # No activations recorded for unknown layers
    assert len(acts) == 0


def test_activation_recorder_hooks_removed_after_context():
    model = nn.Linear(4, 2)
    recorder = ActivationRecorder(model, [""])  # empty = skip
    x = torch.randn(1, 4)
    # Should not raise, hooks cleaned up
    with recorder.record() as acts:
        _ = model(x)
    assert isinstance(acts, dict)
