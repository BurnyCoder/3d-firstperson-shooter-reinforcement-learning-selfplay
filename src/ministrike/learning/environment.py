"""Local: batched NumPy/TensorDict bridge; global: one simulation for training/play.

Source: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html
"""

import numpy as np  # Local: assemble ten-player physical controls; global: NumPy simulator.
import torch  # Local: copied tensor observations; global: differentiable policy boundary.
from tensordict import TensorDict  # Local: nested player data; global: TorchRL convention.
from torchrl.data import Binary, Composite, MultiCategorical, Unbounded  # Library shape contracts.
from torchrl.envs import EnvBase  # Library collection and selective reset machinery.

from ministrike.sim.constants import ACTION_NVECS, OBS_DIM, decode_actions
from ministrike.sim.simulation import Simulation  # Authoritative state and rules.


class ArenaEnv(EnvBase):
    """Local: five learner slots per arena; global: preserve slots through teammate death."""

    def __init__(self, num_envs, seed, condition, pool, scenario="combat"):
        """Local: initialize independent arenas/specs; global: fixed MAPPO interface."""
        super().__init__(device="cpu", batch_size=[num_envs])  # NumPy physics resides on CPU.
        self.sim = Simulation(num_envs=num_envs, seed=seed)  # Own authoritative arena states.
        self.rng = np.random.default_rng(seed + 1009)  # Role draws have a dedicated stream.
        self.condition = condition  # Self-play or fixed scripted control.
        self.pool = pool  # Snapshot pool owns immutable opponent weights.
        self.scenario = scenario  # Applied only at round reset boundaries.
        self.roles = np.zeros(num_envs, dtype=np.int64)  # Learner team 0 or 1 per arena.
        self.opponent_ids = np.zeros(num_envs, dtype=np.int64)  # Snapshot identity per arena.
        self.rounds = 0  # Completed rounds are reported alongside step budgets.
        self.wins = 0  # Learner victories are a diagnostic, not the held-out evaluation.
        self.draws = 0  # Count simultaneous-draw outcomes explicitly.
        self.last_events = []  # Retain the latest game events for diagnostics.
        if condition == "fixed":  # Import lazily so self-play has no scripted dependency.
            from ministrike.policies.scripted import ScriptedPolicy
            self.fixed_policy = ScriptedPolicy()  # Same observation rules as learned actors.
        player_shape = (num_envs, 5)  # Five permanent learner slots per arena.
        self.observation_spec = Composite(agents=Composite(
            observation=Unbounded(shape=(*player_shape, OBS_DIM)),
            alive=Binary(shape=(*player_shape, 1), dtype=torch.bool), shape=player_shape),
            shape=(num_envs,))
        self.action_spec = Composite(agents=Composite(
            action=MultiCategorical(ACTION_NVECS, shape=(*player_shape, 6)), shape=player_shape),
            shape=(num_envs,))
        self.reward_spec = Composite(agents=Composite(
            reward=Unbounded(shape=(*player_shape, 1)), shape=player_shape), shape=(num_envs,))
        self.done_spec = Composite(done=Binary(shape=(num_envs, 1), dtype=torch.bool),
                                   terminated=Binary(shape=(num_envs, 1), dtype=torch.bool),
                                   shape=(num_envs,))

    def _set_seed(self, seed):
        """Local: reset role RNG; global: EnvBase seeding contract."""
        self.rng = np.random.default_rng(seed + 1009)  # Keep role sampling independent.
        self.sim.reset(seed=seed)  # Explicit seeding starts a reproducible simulation.

    def _indices(self, opponent=False):
        """Local: map fixed slots to global players; global: share one actor across sides."""
        teams = 1 - self.roles if opponent else self.roles  # Roles remain fixed within a round.
        return teams[:, None] * 5 + np.arange(5)[None, :]  # Broadcast team-relative player IDs.

    def current_tensordict(self):
        """Local: copy observations/alive flags; global: no NumPy rollout aliasing."""
        batch = np.arange(self.batch_size[0])[:, None]  # One row per arena.
        selected = self._indices()  # Select each arena's assigned learner team.
        observation = self.sim.observe()[batch, selected].copy()  # Fancy indexing owns storage.
        alive = self.sim.alive[batch, selected].copy()[..., None]  # Alive-before-step actor mask.
        done = torch.tensor(self.sim.done.copy()[:, None], dtype=torch.bool)  # Round termination only.
        agents = TensorDict({"observation": torch.from_numpy(observation),
                             "alive": torch.from_numpy(alive)}, [self.batch_size[0], 5])
        return TensorDict({"agents": agents, "done": done, "terminated": done.clone()}, self.batch_size)

    def _reset(self, tensordict=None, **kwargs):
        """Local: apply only requested arena resets; global: preserve concurrent episodes."""
        requested = None if tensordict is None else tensordict.get("_reset", None)
        mask = np.ones(self.batch_size[0], dtype=bool) if requested is None else requested.cpu().numpy().reshape(-1)
        self.roles[mask] = self.rng.integers(0, 2, size=int(mask.sum()))  # Reassign sides per new round.
        self.opponent_ids[mask] = self.pool.choose(int(mask.sum()))  # Freeze opponent identity until termination.
        self.sim.reset(scenario=self.scenario, mask=mask)  # Other arenas retain all state/history.
        self.pool.prune(self.opponent_ids)  # Retain snapshots still used by unfinished rounds.
        return self.current_tensordict()  # The library merges only requested reset slices.

    def _step(self, tensordict):
        """Local: combine both teams then advance once; global: simultaneous game dynamics."""
        observation = self.sim.observe()  # Opponents receive the same local visibility contract.
        arenas = np.arange(self.batch_size[0])[:, None]  # Advanced indexing selects each team.
        learner_indices = self._indices()  # Team assignment cannot change inside this transition.
        opponent_indices = self._indices(opponent=True)  # Other five slots form the frozen opposition.
        controls = np.zeros((self.batch_size[0], 10, 7), dtype=np.float32)
        controls[arenas, learner_indices] = decode_actions(tensordict["agents", "action"].cpu().numpy())
        opponent_observation = observation[arenas, opponent_indices]  # Dynamic hidden fields stay masked.
        controls[arenas, opponent_indices] = (self.fixed_policy.act(opponent_observation)
            if self.condition == "fixed" else self.pool.act(opponent_observation, self.opponent_ids))
        result = self.sim.step(controls)  # All ten controls enter the same fixed simulation tick.
        state = self.current_tensordict()  # Capture terminal observations before any reset.
        reward = result.rewards[np.arange(self.batch_size[0]), self.roles]  # Team reward persists after death.
        state["agents", "reward"] = torch.tensor(reward[:, None, None], dtype=torch.float32).expand(-1, 5, 1).clone()
        self.rounds += int(result.terminated.sum())  # Count every completed arena once.
        self.wins += int(((result.winner == self.roles) & result.terminated).sum())
        self.draws += int(((result.winner < 0) & result.terminated).sum())
        self.last_events = result.events  # Event payloads support logging without rendering.
        return state  # EnvBase attaches this object under the transition's next key.

    def snapshot(self):
        """Local: save environment identity/state; global: resume without resetting rounds."""
        return {"sim": self.sim.state_dict(), "roles": self.roles.copy(),
                "opponent_ids": self.opponent_ids.copy(), "rng": self.rng.bit_generator.state,
                "scenario": self.scenario, "rounds": self.rounds, "wins": self.wins, "draws": self.draws}

    def restore(self, state):
        """Local: restore in-progress arenas; global: preserve trajectory/opponent continuity."""
        self.sim.load_state_dict(state["sim"])  # Restore dynamics and observation history.
        self.roles = state["roles"].copy()  # Resume each arena on its original learner side.
        self.opponent_ids = state["opponent_ids"].copy()  # Preserve frozen per-round opponents.
        self.rng.bit_generator.state = state["rng"]  # Continue role draws exactly.
        self.scenario = state["scenario"]  # Continue the configured curriculum phase.
        self.rounds, self.wins, self.draws = state["rounds"], state["wins"], state["draws"]
