"""Local: immutable historical actors; global: reproducible self-play curricula.

Snapshot rationale: https://arxiv.org/abs/1711.00832
State copies: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import numpy as np  # Local: dedicated matchup randomness; global: reproducibility.
import torch  # Local: frozen actor inference; global: no moving opponent parameters.

from ministrike.learning.models import ActorNetwork, FactorizedCategorical
from ministrike.sim.constants import decode_actions  # Shared physical action interpretation.


class SnapshotPool:
    """Local: retain initial/latest/eight old actors; global: vary opponents per round."""

    def __init__(self, actor, seed, capacity=8):
        """Local: copy initial weights; global: initial policy remains a permanent control."""
        self.capacity = capacity  # Number of historical entries beyond initial/latest.
        self.observation_dim, self.action_nvecs = actor.observation_dim, actor.action_nvecs  # Local: compatible frozen actors.
        self.rng = np.random.default_rng(seed)  # Separate RNG prevents render interference.
        self.states = {}  # CPU tensor state dictionaries own all parameter storage.
        self.cache = {}  # Lazily instantiated CPU actors avoid rebuilding each tick.
        self.latest = 0  # Zero is the initial immutable policy identity.
        self.add(actor, 0)  # Capture before any optimization.

    def add(self, actor, identity):
        """Local: clone actor weights; global: opponent policies change only at new rounds."""
        if identity in self.states:  # Identity reuse would silently mutate a current opponent.
            raise ValueError(f"Snapshot {identity} already exists")
        self.states[identity] = {key: value.detach().cpu().clone()
                                 for key, value in actor.state_dict().items()}
        self.latest = identity  # New rounds can choose the latest finished learner.

    def prune(self, active=()):
        """Local: keep active snapshots until rounds end; global: avoid mid-round replacement."""
        historical = [key for key in self.states if key not in (0, self.latest)]
        retained = set(historical[-self.capacity:]) | {0, self.latest} | set(active)
        self.states = {key: value for key, value in self.states.items() if key in retained}
        self.cache = {key: value for key, value in self.cache.items() if key in retained}

    def choose(self, count):
        """Local: split latest/history equally; global: sample opponents once per round."""
        old = [key for key in self.states if key not in (0, self.latest)]  # Active retired policies are not candidates.
        history = ([0] if self.latest != 0 else []) + old[-self.capacity:] or [self.latest]
        choices = self.rng.choice(history, count)  # First propose a historical identity.
        choices[self.rng.random(count) < 0.5] = self.latest  # Half use the latest snapshot.
        return choices.astype(np.int64)  # Stable serializable opponent identifiers.

    def act(self, observations, identities):
        """Local: batch arenas by snapshot; global: efficient immutable opponent inference."""
        output = np.zeros((*observations.shape[:-1], 7), dtype=np.float32)
        with torch.no_grad():  # Opponents never contribute gradients.
            for identity in np.unique(identities):  # One network call per active snapshot.
                key = int(identity)  # Convert NumPy scalar for JSON/checkpoint compatibility.
                if key not in self.cache:  # Build once while a snapshot remains active.
                    with torch.random.fork_rng(devices=[]):  # Cache construction must not advance rollout RNG.
                        actor = ActorNetwork(self.observation_dim, self.action_nvecs).eval()  # Local: match-specific actor.
                    actor.load_state_dict(self.states[key])  # Load the immutable copy.
                    self.cache[key] = actor.requires_grad_(False)  # Enforce frozen inference.
                selected = identities == key  # Gather the arenas using this snapshot.
                logits = self.cache[key](torch.from_numpy(observations[selected].copy()))
                output[selected] = decode_actions(FactorizedCategorical(logits).sample().numpy(), self.action_nvecs)
        return output  # Physical controls are consumed by the authoritative simulator.

    def state_dict(self):
        """Local: capture weights and matchup RNG; global: resume the same experiment."""
        return {"states": self.states, "latest": self.latest, "capacity": self.capacity,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        """Local: restore immutable pool; global: continue the original opponent schedule."""
        self.states = state["states"]  # Checkpoint loader already owns these tensor objects.
        self.latest = state["latest"]  # Preserve the saved learner snapshot identity.
        self.capacity = state["capacity"]  # Preserve historical selection capacity.
        self.rng.bit_generator.state = state["rng"]  # Continue matchup randomness.
        self.cache.clear()  # Rebuild networks from restored weight copies.
