"""Local: batched NumPy/TensorDict bridge; global: one simulation for training/play.

Source: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html
"""

import numpy as np  # Local: assemble ten-player physical controls; global: NumPy simulator.
import torch  # Local: copied tensor observations; global: differentiable policy boundary.
from tensordict import TensorDict  # Local: nested player data; global: TorchRL convention.
from torchrl.data import Binary, Composite, MultiCategorical, Unbounded  # Library shape contracts.
from torchrl.envs import EnvBase  # Library collection and selective reset machinery.

from ministrike.sim.constants import decode_actions
from ministrike.sim.simulation import Simulation  # Authoritative state and rules.


class ArenaEnv(EnvBase):
    """Local: five learner slots per arena; global: preserve slots through teammate death."""

    def __init__(self, num_envs, seed, condition, pool, scenario="combat", experiment=None, reward="R0"):
        """Local: initialize independent arenas/specs; global: fixed MAPPO interface."""
        super().__init__(device="cpu", batch_size=[num_envs])  # NumPy physics resides on CPU.
        self.sim = Simulation(num_envs=num_envs, seed=seed, experiment=experiment, reward=reward)  # Shared state authority.
        self.team_size = self.sim.team_size  # Local: nested player batch dimension derives from the match.
        if not self.sim.spec.bomb and condition != "selfplay":
            raise ValueError("Stage A permits self-play training only")  # Local: scripts are isolated evaluation controls.
        self._observation = self.sim.observe()  # Cache the authoritative full-team observation between transitions.
        self.rng = np.random.default_rng(seed + 1009)  # Role draws have a dedicated stream.
        self.condition = condition  # Self-play or fixed scripted control.
        self.pool = pool  # Snapshot pool owns immutable opponent weights.
        self.scenario = scenario  # Applied only at round reset boundaries.
        self.roles = np.zeros(num_envs, dtype=np.int64)  # Learner team 0 or 1 per arena.
        self.reset_counts = np.zeros(num_envs, dtype=np.int64)  # Local: alternate spawn ends independently per arena.
        self.telemetry = {}  # Local: cumulative behavior/reward diagnostics survive checkpoint resume.
        self.opponent_ids = np.zeros(num_envs, dtype=np.int64)  # Snapshot identity per arena.
        self.rounds = 0  # Completed rounds are reported alongside step budgets.
        self.wins = 0  # Learner victories are a diagnostic, not the held-out evaluation.
        self.draws = 0  # Count simultaneous-draw outcomes explicitly.
        self.last_events = []  # Retain the latest game events for diagnostics.
        if condition == "fixed":  # Import lazily so self-play has no scripted dependency.
            from ministrike.policies.scripted import ScriptedPolicy
            self.fixed_policy = ScriptedPolicy()  # Same observation rules as learned actors.
        player_shape = (num_envs, self.team_size)  # Permanent learner slots, even after death.
        self.observation_spec = Composite(agents=Composite(
            observation=Unbounded(shape=(*player_shape, self.sim.spec.observation_dim)),
            alive=Binary(shape=(*player_shape, 1), dtype=torch.bool), shape=player_shape),
            shape=(num_envs,))
        self.action_spec = Composite(agents=Composite(
            action=MultiCategorical(self.sim.spec.action_nvecs, shape=(*player_shape, len(self.sim.spec.action_nvecs))), shape=player_shape),
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
        return teams[:, None] * self.team_size + np.arange(self.team_size)[None, :]  # Local: stable team slots.

    def current_tensordict(self, observation=None):
        """Local: copy observations/alive flags; global: no NumPy rollout aliasing."""
        self._observation = self.sim.observe() if observation is None else observation.copy()  # Explicit reads also refresh test interventions.
        batch = np.arange(self.batch_size[0])[:, None]  # One row per arena.
        selected = self._indices()  # Select each arena's assigned learner team.
        selected_observation = self._observation[batch, selected].copy()  # Fancy indexing owns independent rollout storage.
        alive = self.sim.alive[batch, selected].copy()[..., None]  # Alive-before-step actor mask.
        done = torch.tensor(self.sim.done.copy()[:, None], dtype=torch.bool)  # Round termination only.
        agents = TensorDict({"observation": torch.from_numpy(selected_observation),
                             "alive": torch.from_numpy(alive)}, [self.batch_size[0], self.team_size])
        return TensorDict({"agents": agents, "done": done, "terminated": done.clone()}, self.batch_size)

    def _reset(self, tensordict=None, **kwargs):
        """Local: apply only requested arena resets; global: preserve concurrent episodes."""
        requested = None if tensordict is None else tensordict.get("_reset", None)
        mask = np.ones(self.batch_size[0], dtype=bool) if requested is None else requested.cpu().numpy().reshape(-1)
        self.roles[mask] = (self.rng.integers(0, 2, size=int(mask.sum())) if self.sim.spec.bomb else
                            (self.reset_counts[mask] + np.flatnonzero(mask)) % 2)  # Local: balanced alternating duel ends.
        self.reset_counts[mask] += 1  # Local: selective reset never changes an unfinished arena's role counter.
        self.opponent_ids[mask] = self.pool.choose(int(mask.sum()))  # Freeze opponent identity until termination.
        observation = self.sim.reset(scenario=self.scenario, mask=mask)  # Other arenas retain all state/history.
        self.pool.prune(self.opponent_ids)  # Retain snapshots still used by unfinished rounds.
        return self.current_tensordict(observation)  # Reuse the simulator's computed frame; merge requested reset slices.

    def _step(self, tensordict):
        """Local: combine both teams then advance once; global: simultaneous game dynamics."""
        observation = self._observation  # Reuse the previous authoritative observation, avoiding duplicate visibility geometry.
        arenas = np.arange(self.batch_size[0])[:, None]  # Advanced indexing selects each team.
        learner_indices = self._indices()  # Team assignment cannot change inside this transition.
        opponent_indices = self._indices(opponent=True)  # Other five slots form the frozen opposition.
        controls = np.zeros((self.batch_size[0], self.sim.players, 7), dtype=np.float32)
        controls[arenas, learner_indices] = decode_actions(tensordict["agents", "action"].cpu().numpy(), self.sim.spec.action_nvecs)
        opponent_observation = observation[arenas, opponent_indices]  # Dynamic hidden fields stay masked.
        controls[arenas, opponent_indices] = (self.fixed_policy.act(opponent_observation)
            if self.condition == "fixed" else self.pool.act(opponent_observation, self.opponent_ids))
        result = self.sim.step(controls)  # All ten controls enter the same fixed simulation tick.
        state = self.current_tensordict(result.observations)  # Capture terminal observations before any reset or extra geometry pass.
        reward = result.rewards[np.arange(self.batch_size[0]), self.roles]  # Team reward persists after death.
        state["agents", "reward"] = torch.tensor(reward[:, None, None], dtype=torch.float32).expand(-1, self.team_size, 1).clone()
        if not self.sim.spec.bomb:
            self._record_diagnostics(controls, learner_indices, result)  # Local: measure behavior, not just loss curves.
        self.rounds += int(result.terminated.sum())  # Count every completed arena once.
        self.wins += int(((result.winner == self.roles) & result.terminated).sum())
        self.draws += int(((result.winner < 0) & result.terminated).sum())
        self.last_events = result.events  # Event payloads support logging without rendering.
        return state  # EnvBase attaches this object under the transition's next key.

    def snapshot(self):
        """Local: save environment identity/state; global: resume without resetting rounds."""
        return {"sim": self.sim.state_dict(), "roles": self.roles.copy(),
                "opponent_ids": self.opponent_ids.copy(), "rng": self.rng.bit_generator.state,
                "scenario": self.scenario, "rounds": self.rounds, "wins": self.wins, "draws": self.draws,
                "reset_counts": self.reset_counts.copy(), "telemetry": self.telemetry.copy()}

    def restore(self, state):
        """Local: restore in-progress arenas; global: preserve trajectory/opponent continuity."""
        self.sim.load_state_dict(state["sim"])  # Restore dynamics and observation history.
        self._observation = self.sim.observe()  # A restored simulator owns a fresh matching observation cache.
        self.roles = state["roles"].copy()  # Resume each arena on its original learner side.
        self.opponent_ids = state["opponent_ids"].copy()  # Preserve frozen per-round opponents.
        self.rng.bit_generator.state = state["rng"]  # Continue role draws exactly.
        self.scenario = state["scenario"]  # Continue the configured curriculum phase.
        self.rounds, self.wins, self.draws = state["rounds"], state["wins"], state["draws"]
        self.reset_counts = state.get("reset_counts", self.reset_counts).copy()  # Local: legacy files lack duel counters.
        self.telemetry = state.get("telemetry", {}).copy()  # Local: resume does not reset cumulative diagnostics.

    def _record_diagnostics(self, controls, selected, result):
        """Local: accumulate interpretable behavior counts; global: expose reward hacking and inactivity."""
        rows = np.arange(self.sim.num_envs)[:, None]  # Local: gather learner slots by current role.
        chosen = controls[rows, selected]  # Local: count only learner behavior.
        values = {"agent_decisions": chosen.shape[0] * chosen.shape[1],
                  "reload_ticks": int((self.sim.reload_timer[rows, selected] > 0).sum()),
                  "aim_saturated_ticks": int((np.abs(self.sim.pitch[rows, selected]) >= 1.3).sum()),
                  "inactive_ticks": int((np.linalg.norm(chosen, axis=-1) < 1e-6).sum())}
        for key, component in self.sim.reward_components.items():
            values[f"reward_{key}"] = float(component[np.arange(self.sim.num_envs), self.roles].sum())
        for event in result.events:
            learner = int(self.roles[event["arena"]])  # Local: duel slots equal team IDs.
            kind = event["type"]
            if kind == "damage" and event["attacker"] == learner:
                values["damage"] = values.get("damage", 0) + event["amount"]
                values["hits"] = values.get("hits", 0) + 1
            if kind == "shot" and event["player"] == learner:
                values["shots"] = values.get("shots", 0) + 1
            if kind == "kill" and event["player"] != learner:
                values["kills"] = values.get("kills", 0) + 1
            if kind == "round_end":
                reason = event["reason"]
                values[reason] = values.get(reason, 0) + 1
        for key, value in values.items():
            self.telemetry[key] = self.telemetry.get(key, 0) + value  # Local: compact counters, full metrics every update.
