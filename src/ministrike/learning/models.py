"""Local: small policy/value networks; global: centralized training, local action.

Methods: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
Distributions: https://docs.pytorch.org/docs/stable/distributions.html
"""

import torch  # Local: tensor operations; global: differentiable actor and critic.
from torch import nn  # Local: reusable layers; global: compact network definitions.
from torch.distributions import Categorical, Independent  # Local: library probabilities.
from tensordict.nn import TensorDictModule  # Local: named I/O; global: TorchRL integration.
from torchrl.modules import ProbabilisticActor  # Local: library sampling and log probabilities.

from ministrike.sim.constants import ACTION_NVECS, OBS_DIM  # Shared observation/action contract.


class FactorizedCategorical(Independent):
    """Local: join six control heads; global: never join independent player axes."""

    arg_constraints = {}  # Base categorical validates its own finite normalized logits.

    def __init__(self, logits, validate_args=None):
        """Local: reinterpret only the head axis; global: correct per-player PPO ratios."""
        super().__init__(Categorical(logits=logits), 1, validate_args=validate_args)

    @property
    def mode(self):
        """Local: category argmax; global: deterministic evaluation and gameplay."""
        return self.base_dist.logits.argmax(-1)  # Keep all leading arena/player dimensions.


class ActorNetwork(nn.Module):
    """Local: shared two-layer MLP; global: bots use their own local history only."""

    def __init__(self, observation_dim=OBS_DIM, action_nvecs=ACTION_NVECS):
        """Local: allocate layers and action mask; global: fixed published architecture."""
        super().__init__()  # Register parameters with PyTorch.
        self.observation_dim, self.action_nvecs = observation_dim, tuple(action_nvecs)  # Local: explicit saved interface.
        self.network = nn.Sequential(nn.Linear(observation_dim, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh(),
                                     nn.Linear(128, sum(action_nvecs)))
        self.register_buffer("head_sizes", torch.tensor(action_nvecs))  # Save contract with weights.

    def forward(self, observation):
        """Local: pad unequal categorical heads; global: reuse standard distributions."""
        pieces = self.network(observation).split(self.action_nvecs, dim=-1)  # Each enabled slice is one control.
        padded = [torch.nn.functional.pad(piece, (0, max(self.action_nvecs) - piece.shape[-1]),
                                         value=-1e9) for piece in pieces]
        return torch.stack(padded, dim=-2)  # [..., six heads, nine padded categories].


class CentralCritic(nn.Module):
    """Local: combine five local observations; global: centralized team value learning."""

    def __init__(self, observation_dim=OBS_DIM, team_size=5):
        """Local: define a 2x256 critic; global: actor remains independently executable."""
        super().__init__()  # Register the critic network.
        self.network = nn.Sequential(nn.Linear(team_size * observation_dim, 256), nn.Tanh(),
                                     nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, team_size))

    def forward(self, observation):
        """Local: flatten only player/features; global: preserve batch and time axes."""
        return self.network(observation.flatten(-2)).unsqueeze(-1)  # [..., five players, one value].


def make_actor(network=None):
    """Local: bind named TensorDict keys; global: standard TorchRL probability handling."""
    module = TensorDictModule(network or ActorNetwork(), in_keys=[("agents", "observation")],
                              out_keys=[("agents", "logits")])
    return ProbabilisticActor(module, in_keys={"logits": ("agents", "logits")},
                              out_keys=[("agents", "action")],
                              distribution_class=FactorizedCategorical, return_log_prob=True,
                              log_prob_key=("agents", "action_log_prob"))


def make_critic(network=None):
    """Local: bind the central critic; global: let TorchRL compute value losses/GAE."""
    return TensorDictModule(network or CentralCritic(), in_keys=[("agents", "observation")],
                             out_keys=[("agents", "state_value")])
