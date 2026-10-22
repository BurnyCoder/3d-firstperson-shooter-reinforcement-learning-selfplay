"""Local: verify batched simulator wrapping; global: prevent training leakage.

Source: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html
"""

import torch  # Local: reset masks; global: TorchRL batch contract.
from tensordict import TensorDict  # Local: structured reset requests; global: library interface.

from ministrike.learning.environment import ArenaEnv
from ministrike.learning.models import ActorNetwork
from ministrike.learning.opponents import SnapshotPool
from ministrike.learning.models import make_actor
from ministrike.learning.trainer import collect_batch
from ministrike.sim.constants import GameConfig
from ministrike.sim.simulation import Simulation


def test_partial_reset_does_not_touch_other_arena():
    """Local: reset only arena zero; global: independent parallel episodes."""
    env = ArenaEnv(2, 5, "selfplay", SnapshotPool(ActorNetwork(), 5))  # Two arenas share no state.
    before = env.reset().clone()  # Own a snapshot of both observations.
    roles = env.roles.copy()  # Role changes must also be selective.
    env.reset(TensorDict({"_reset": torch.tensor([[True], [False]])}, [2]))
    after = env.current_tensordict()  # Observe without advancing simulation.
    assert torch.equal(before["agents", "observation"][1], after["agents", "observation"][1])
    assert env.roles[1] == roles[1]  # The active arena keeps its learner side.


def test_transition_observation_does_not_alias_simulator():
    """Local: advance after capture; global: replay data must not mutate in place."""
    env = ArenaEnv(2, 7, "selfplay", SnapshotPool(ActorNetwork(), 7))  # Independent simulator.
    state = env.reset()  # Allocate the first observation.
    before = state["agents", "observation"].clone()  # Keep a reference answer.
    state["agents", "action"] = torch.zeros(2, 5, 6, dtype=torch.long)  # Valid neutral fixture.
    env.step(state)  # Simulation changes may never overwrite previous observations.
    assert torch.equal(before, state["agents", "observation"])


def test_library_rollout_auto_resets_only_finished_rounds():
    """Local: collect across several tiny rounds; global: no post-terminal padding enters PPO."""
    actor = ActorNetwork()  # Collection uses a genuinely stochastic untrained neural policy.
    env = ArenaEnv(2, 7, "selfplay", SnapshotPool(actor, 7))  # Two independent arenas.
    env.sim = Simulation(2, seed=7, config=GameConfig(round_seconds=0.1))  # Two-tick rounds are a test fixture.
    env.reset()  # Assign sides/opponents before collection.
    batch = collect_batch(env, make_actor(actor), 7)  # Official rollout machinery crosses multiple resets.
    assert batch.batch_size == torch.Size([2, 7])  # Arena/time remain separate for GAE.
    assert batch["agents", "action_log_prob"].shape == (2, 7, 5)  # Probability axes stay per-player.
    assert env.rounds == 6  # Three complete two-tick rounds per arena, then one unfinished tick.
    done = batch["next", "done"].squeeze(-1)  # Preserve every terminal transition in the collected data.
    assert done.sum() == 6 and not torch.any(done[:, 1:] & done[:, :-1])  # No terminal-state padding.


def test_restoring_environment_pool_and_rng_replays_actions_exactly():
    """Local: resume with a cold opponent cache; global: checkpoint restore must preserve future behavior."""
    actor = ActorNetwork()  # Same learner weights are used before and after restore.
    pool = SnapshotPool(actor, 17)  # Own matchup history and RNG stream.
    env = ArenaEnv(2, 17, "selfplay", pool)  # Two in-progress arenas exercise batched restoration.
    env.reset()  # Initialize roles, opponents, and simulation history.
    collect_batch(env, make_actor(actor), 2)  # Warm the original cache before taking the state snapshot.
    state, pool_state, rng = env.snapshot(), pool.state_dict(), torch.get_rng_state()  # Save all rollout state.
    expected = collect_batch(env, make_actor(actor), 3)  # Reference continuation on the original objects.
    restored_pool = SnapshotPool(actor, 999)  # Construction seed must become irrelevant after restore.
    restored_pool.load_state_dict(pool_state)  # Restores snapshot identities and selection RNG.
    restored = ArenaEnv(2, 17, "selfplay", restored_pool)  # Its opponent inference cache starts empty.
    restored.restore(state)  # Restore roles, physical state, observations, and arena RNG.
    torch.set_rng_state(rng)  # Continue categorical draws from the exact saved point.
    actual = collect_batch(restored, make_actor(actor), 3)  # Cold cache creation must not consume RNG.
    assert torch.equal(expected["agents", "action"], actual["agents", "action"])
    assert torch.equal(expected["next", "agents", "observation"], actual["next", "agents", "observation"])


def test_cached_transition_matches_fresh_simulator_observation_and_does_not_alias():
    """Local: compare cached/fresh frames after actions; global: performance changes cannot change training evidence."""
    actor = ActorNetwork()  # Deterministic construction is sufficient because comparison uses the same transition.
    env = ArenaEnv(2, 23, "selfplay", SnapshotPool(actor, 23))  # Two arenas exercise independent team selection.
    state = env.reset()  # Cache receives the same complete observation as a fresh reset.
    state["agents", "action"] = torch.zeros(2, 5, 6, dtype=torch.long)  # Explicit legal control categories.
    transition = env.step(state)  # Production path consumes the cached pre-state and result observations.
    cached = transition["next", "agents", "observation"].clone()  # Retain an independent reference.
    refreshed = env.current_tensordict()  # Recompute geometry explicitly as the uncached path previously did.
    assert torch.equal(cached, refreshed["agents", "observation"])  # Every scalar must be unchanged.
    env._observation.fill(999)  # A later cache overwrite must not mutate already collected observations.
    assert torch.equal(cached, transition["next", "agents", "observation"])
