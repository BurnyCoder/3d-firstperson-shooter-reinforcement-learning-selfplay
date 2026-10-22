"""Local: deterministic simulator contracts. Global: gameplay and training share these rules."""

import numpy as np  # Local: array assertions. Global: simulator's authoritative storage.

from ministrike.sim import Simulation  # Local: production API. Global: test human/RL shared physics.
from ministrike.sim.constants import DT, OBS_DIM, GameConfig  # Local: rule constants, not duplicated magic values.
from ministrike.sim.map import SITES  # Local: objective positions. Global: map/rules use the same source.


def neutral(sim):
    """Local: construct no-op controls. Global: isolate the mechanic under test."""
    return np.zeros((sim.num_envs, 10, 7), dtype=np.float32)  # Local: fixed slots persist after death.


def duel(sim, left=(-5, 0, -8), right=(5, 0, -8)):
    """Local: isolate one opponent pair. Global: hitscan/termination tests avoid unrelated actors."""
    sim.health[:] = 0  # Local: remove unrelated players without changing slot identities.
    sim.health[:, [0, 5]] = 100  # Local: retain one player on each team.
    sim.positions[:, 0] = left  # Local: shooter starts on unobstructed outer lane.
    sim.positions[:, 5] = right  # Local: target shares the lane.
    sim.yaw[:, 0] = np.pi / 2  # Local: positive X aim.
    sim.yaw[:, 5] = -np.pi / 2  # Local: negative X aim.


def test_deterministic_seed_and_observation_shape():
    """Local: seeded trajectories agree. Global: reported experiments can reproduce physics."""
    a, b = Simulation(3, seed=11), Simulation(3, seed=11)  # Local: separate RNG objects, same initial seed.
    assert a.observe().shape == (3, 10, OBS_DIM)  # Local: actor input has stable slot/history dimensions.
    controls = np.random.default_rng(4).uniform(-1, 1, (3, 10, 7))  # Local: independent deterministic controls.
    for _ in range(12):  # Local: compare a trajectory, not merely reset state.
        np.testing.assert_array_equal(a.step(controls).observations, b.step(controls).observations)


def test_partial_reset_preserves_other_arena():
    """Local: reset one arena. Global: batched rollouts must not erase unrelated trajectories."""
    sim = Simulation(2, seed=3)  # Local: two arenas expose reset cross-talk.
    sim.step(neutral(sim))  # Local: give untouched arena noninitial history/time.
    before = sim.snapshot(1)  # Local: capture all public state for comparison.
    sim.reset(seed=42, mask=np.array([True, False]))  # Local: reset only arena zero.
    assert sim.snapshot(1) == before  # Local: untouched arena is byte-for-byte equivalent in JSON form.


def test_shots_resolve_simultaneously_and_dead_slots_persist():
    """Local: mutual lethal shots draw. Global: array ordering must not privilege one team."""
    sim = Simulation(config=GameConfig(damage=100))  # Local: make one hit lethal.
    duel(sim)  # Local: opponents face each other in clear space.
    controls = neutral(sim)  # Local: no movement or interaction confounds.
    controls[:, [0, 5], 4] = 1  # Local: both fire on the same tick.
    result = sim.step(controls)  # Local: one authoritative simultaneous tick.
    assert not sim.alive.any()  # Local: both shots count despite their shooters dying.
    assert result.terminated[0] and result.winner[0] == -1  # Local: mutual elimination before planting is a draw.
    assert result.observations.shape[1] == 10  # Local: death does not remove policy slots.


def test_wall_blocks_hitscan_and_hidden_enemy_fields():
    """Local: walls occlude shots and observations. Global: policy cannot exploit hidden enemy state."""
    sim = Simulation()  # Local: default map and weapons.
    duel(sim, (-5, 0, 0), (5, 0, 0))  # Local: central solid building separates actors.
    controls = neutral(sim)  # Local: isolate firing.
    controls[:, 0, 4] = 1  # Local: fire directly into the building.
    sim.step(controls)  # Local: run collision and visibility from shared geometry.
    assert sim.health[0, 5] == 100  # Local: nearest wall absorbs the shot.
    enemy = sim.observe()[0, 0, -96 + 36 :].reshape(10, 6)[5]  # Local: inspect current-frame enemy slot.
    assert not enemy.any()  # Local: enemy dynamics, visibility, and enemy HP are zero behind cover.


def test_ammunition_and_reload_are_finite():
    """Local: empty magazines need reload. Global: training and human combat obey the same resource limits."""
    sim = Simulation()  # Local: default reload duration and magazine.
    sim.ammo[0, 0] = 0  # Local: establish an empty magazine.
    sim.reserve[0, 0] = 7  # Local: reserve limits the refill.
    controls = neutral(sim)  # Local: idle all other players.
    controls[0, 0, 5] = 1  # Local: request reload.
    for _ in range(round(sim.config.reload_seconds / DT) + 2):  # Local: allow countdown to complete.
        sim.step(controls)  # Local: held reload must not continually restart the timer.
    assert sim.ammo[0, 0] == 7 and sim.reserve[0, 0] == 0  # Local: ammo is transferred, never created.


def test_planted_bomb_survives_attacker_elimination():
    """Local: an armed bomb remains active. Global: delayed team returns exist after attacker death."""
    sim = Simulation()  # Local: default rules.
    sim.bomb_planted[0] = True  # Local: establish post-plant phase.
    sim.bomb_carrier[0] = -1  # Local: planted bomb has no carrier.
    sim.bomb_timer[0] = DT * 2  # Local: short countdown isolates terminal logic.
    sim.health[0, :5] = 0  # Local: all attackers are dead.
    first = sim.step(neutral(sim))  # Local: countdown still has one tick.
    assert not first.terminated[0]  # Local: attacker elimination cannot override an armed bomb.
    second = sim.step(neutral(sim))  # Local: explosion wins for the dead attacking team.
    assert second.terminated[0] and second.winner[0] == 0  # Local: delayed terminal reward is preserved.


def test_defuse_completion_precedes_same_tick_explosion():
    """Local: a completed defuse wins ties. Global: terminal precedence is explicit and stable."""
    sim = Simulation()  # Local: default defuse rules.
    sim.bomb_position[0] = SITES[0]  # Local: place bomb at legal site.
    sim.bomb_planted[0] = True  # Local: enable defusing.
    sim.bomb_carrier[0] = -1  # Local: remove carried-bomb semantics.
    sim.bomb_timer[0] = DT  # Local: explosion would occur this tick.
    sim.positions[0, 5] = SITES[0]  # Local: defender is within interaction radius.
    sim.interaction[0, 5] = sim.config.defuse_seconds - DT  # Local: this tick completes defusing.
    controls = neutral(sim)  # Local: keep everyone stationary.
    controls[0, 5, 6] = 1  # Local: hold the defuse button.
    result = sim.step(controls)  # Local: resolve interaction before explosion.
    assert result.terminated[0] and result.winner[0] == 1  # Local: completed defuse has precedence.


def test_terminal_hold_and_checkpoint_replay():
    """Local: checkpoint restores exact trajectories. Global: paused runs do not silently change experiments."""
    sim = Simulation(2, seed=8, config=GameConfig(round_seconds=DT))  # Local: force quick terminal states.
    result = sim.step(neutral(sim))  # Local: both rounds hit the timeout.
    assert result.terminated.all()  # Local: terminal arenas must wait for explicit reset.
    saved = sim.state_dict()  # Local: copy internal physics/history/RNG state.
    before = sim.snapshot()  # Local: public state before an extra tick.
    assert not sim.step(neutral(sim)).rewards.any()  # Local: terminal rewards cannot repeat.
    assert sim.snapshot() == before  # Local: terminal hold freezes counters and positions.
    restored = Simulation(2, config=sim.config)  # Local: fresh instance owns independent arrays.
    restored.load_state_dict(saved)  # Local: restore exact saved arena state.
    np.testing.assert_array_equal(restored.observe(), sim.observe())  # Local: observation history is checkpointed.


def test_movement_cannot_cross_walls():
    """Local: movement respects building bounds. Global: headless agents cannot walk through rendered geometry."""
    sim = Simulation()  # Local: default map.
    sim.positions[0, 0] = (-3, 0, 0)  # Local: start just west of the central building.
    sim.yaw[0, 0] = np.pi / 2  # Local: move east toward its wall.
    controls = neutral(sim)  # Local: isolate one player's movement.
    controls[0, 0, 0] = 1  # Local: hold forward.
    for _ in range(30):  # Local: sufficient time to cross an unblocked building.
        sim.step(controls)  # Local: test repeated collision resolution.
    assert sim.positions[0, 0, 0] < -2.25  # Local: player radius remains outside the wall.


def test_low_cover_blocks_feet_but_not_eye_height_shots():
    """Local: ray elevation matters. Global: the floor-bound game still uses three-dimensional combat geometry."""
    sim = Simulation()  # Local: default low western cover spans Z=-2 to -1 at height one metre.
    duel(sim, (-9, 0, -5), (-9, 0, 0))  # Local: opponents stand on opposite sides of low cover.
    sim.yaw[0, 0] = 0  # Local: aim toward positive Z through the cover's horizontal footprint.
    controls = neutral(sim)  # Local: no motion is needed to shoot over low cover.
    controls[0, 0, 4] = 1  # Local: one eye-height rifle shot.
    result = sim.step(controls)  # Local: actual 3D slabs determine obstruction.
    assert sim.health[0, 5] == 75  # Local: a wall below the shot does not block the rifle.
    damage = next(event for event in result.events if event["type"] == "damage")  # Local: event attribution is explicit.
    assert damage["attacker"] == 0 and damage["victim"] == 5 and damage["amount"] == 25


def test_observe_is_pure_fresh_and_does_not_expose_enemy_health():
    """Local: read-only observations update only returned current frame. Global: camera reads cannot alter training state."""
    sim = Simulation()  # Local: default visibility setup.
    duel(sim)  # Local: visible enemies face one another in clear space.
    before = sim.state_dict()  # Local: preserve exact internal history and random generator.
    first = sim.observe()  # Local: recompute current observable features after the deliberate test intervention.
    second = sim.observe()  # Local: repeated observation reads should be identical.
    np.testing.assert_array_equal(first, second)  # Local: no stochastic observations.
    np.testing.assert_array_equal(before["arrays"]["history"], sim.history)  # Local: reads do not roll historical frames.
    enemy = first[0, 0, -60:].reshape(10, 6)[5]  # Local: latest frame's defender slot.
    assert enemy[3] == 1 and enemy[5] == 0  # Local: visibility does not disclose enemy HP.
    sim.health[0, 5] = 30  # Local: alter an unobservable enemy attribute without changing visibility.
    np.testing.assert_array_equal(first[0, 0], sim.observe()[0, 0])  # Local: actor observation is invariant to enemy HP.


def test_sustained_rifle_cadence_matches_configured_rate():
    """Local: fixed ticks retain cooldown residuals. Global: stated six-shot cadence is not silently rounded to five."""
    sim = Simulation()  # Local: default six-shot rifle.
    controls = neutral(sim)  # Local: no movement or other actions.
    controls[0, 0, 4] = 1  # Local: continuously hold the trigger.
    sim.yaw[0, 0] = np.pi  # Local: shoot harmlessly into the rear perimeter wall.
    for _ in range(round(1 / DT)):  # Local: sample exactly one simulated second.
        sim.step(controls)  # Local: firing uses ordinary production ticks.
    assert sim.shots_fired[0, 0] == sim.config.shots_per_second  # Local: empirical cadence agrees with declared rules.


def test_plant_completion_beats_round_timeout_and_awards_once():
    """Local: an exactly completed plant starts the bomb phase. Global: objective deadline precedence is reproducible."""
    sim = Simulation(config=GameConfig(round_seconds=DT))  # Local: timeout and plant complete on the same tick.
    sim.positions[0, 0] = SITES[1]  # Local: the carrier stands at the eastern legal site.
    sim.interaction[0, 0] = sim.config.plant_seconds - DT  # Local: one tick remains in held planting progress.
    controls = neutral(sim)  # Local: keep the planter stationary.
    controls[0, 0, 6] = 1  # Local: finish planting.
    result = sim.step(controls)  # Local: objective commit precedes the pre-plant timeout check.
    assert sim.bomb_planted[0] and not result.terminated[0]  # Local: the round continues with a live bomb.
    np.testing.assert_allclose(result.rewards[0], [.1, -.1])  # Local: one bounded plant reward is zero-sum.
    assert not sim.step(controls).rewards.any()  # Local: continuing to hold interaction cannot repeat plant reward.


def test_interrupted_interaction_loses_progress():
    """Local: planting requires uninterrupted stationary input. Global: agents cannot bank partial plant progress."""
    sim = Simulation()  # Local: default interaction rules.
    sim.positions[0, 0] = SITES[0]  # Local: carrier is initially eligible to plant.
    controls = neutral(sim)  # Local: no other actors act.
    controls[0, 0, 6] = 1  # Local: begin planting.
    sim.step(controls)  # Local: first held tick creates positive progress.
    assert sim.interaction[0, 0] > 0  # Local: verify the test actually began interaction.
    controls[0, 0, 0] = 1  # Local: movement intent interrupts the held plant.
    sim.step(controls)  # Local: production objective rules cancel progress.
    assert sim.interaction[0, 0] == 0  # Local: a new stationary hold must start from zero.


def test_checkpoint_restores_rng_for_future_curriculum_resets():
    """Local: checkpoint includes reset randomness. Global: resume preserves future starts, not just current positions."""
    first = Simulation(4, seed=124)  # Local: a multi-arena stream exercises objective subtask sampling.
    state = first.state_dict()  # Local: save generator before further reset draws.
    second = Simulation(4, seed=999)  # Local: unrelated construction seed must be replaced by restored state.
    second.load_state_dict(state)  # Local: restore arrays and generator state.
    for scenario in ("objective", "combat", "standard"):  # Local: test all supported curriculum branches.
        np.testing.assert_array_equal(first.reset(scenario=scenario), second.reset(scenario=scenario))


def test_ray_box_parallel_axis_and_boundary_cases():
    """Local: robust slab intersection handles zero direction components. Global: axis-aligned aim cannot produce NaNs."""
    from ministrike.sim.geometry import ray_boxes  # Local: test the shared primitive used by shots and sight.
    origins = np.array([[0, 1, -2], [2, 1, -2], [1, 1, -2]], dtype=np.float32)  # Local: inside/outside/boundary rays.
    directions = np.array([[0, 0, 1]] * 3, dtype=np.float32)  # Local: two parallel axes exercise safe division.
    distance = ray_boxes(origins, directions, np.array([-1, 0, 0]), np.array([1, 2, 2]))
    np.testing.assert_array_equal(distance, [2, np.inf, 2])  # Local: parallel outside misses; boundary ray intersects.


def test_standard_spawns_occlude_all_cross_team_sightlines_across_seeds():
    """Local: original spawns protect every player. Global: human play must allow navigation before first contact."""
    from ministrike.sim.observation import visible_players  # Local: inspect production FOV and occlusion jointly.
    for seed in range(20):  # Local: cover all player spawn jitter with multiple independent deterministic starts.
        sim = Simulation(seed=seed)  # Local: ordinary round starts use protected team areas.
        visible, _ = visible_players(sim)  # Local: compute the exact visibility presented to actor observations.
        assert not visible[0, :5, 5:].any()  # Local: no attacker sees a defender at spawn.
        assert not visible[0, 5:, :5].any()  # Local: no defender sees an attacker at spawn.


def test_combat_curriculum_keeps_visible_close_opponents():
    """Local: protected standard spawns do not break curriculum. Global: combat lessons still start with aimable targets."""
    from ministrike.sim.observation import visible_players  # Local: use the same visibility contract as training.
    sim = Simulation(seed=7)  # Local: deterministic test initialization.
    sim.reset(scenario="combat")  # Local: request intentionally close-range visible engagements.
    visible, _ = visible_players(sim)  # Local: geometry must not hide the curriculum's opposing row.
    assert visible[0, :5, 5:].any(axis=-1).all()  # Local: every attacker has at least one visible opponent.
    assert visible[0, 5:, :5].any(axis=-1).all()  # Local: every defender also has at least one visible opponent.


def test_dropped_bomb_can_be_recovered_at_scripted_stopping_distance():
    """Local: recovery range includes the baseline's objective stop. Global: bots cannot stall just outside pickup range."""
    sim = Simulation()  # Local: default game rules and objective geometry.
    sim.bomb_carrier[0] = -1  # Local: establish a dropped objective without a live carrier.
    sim.bomb_position[0] = SITES[0]  # Local: use unobstructed ground behind the defender spawn baffle.
    sim.positions[0, 1] = SITES[0] + np.array([-.85, 0, 0])  # Local: scripted navigation stops within0.9m.
    sim.step(neutral(sim))  # Local: automatic pickup occurs through ordinary game mechanics.
    assert sim.bomb_carrier[0] == 1  # Local: the arriving attacker recovers the bomb without tighter navigation.
