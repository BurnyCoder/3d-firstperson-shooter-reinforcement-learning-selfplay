"""Local: bomb state machine and explicit terminal precedence. Global: delayed returns survive attacker death."""

import numpy as np  # Local: evaluate independent objective state machines in one vectorized pass.

from .constants import DT  # Local: gameplay progress uses simulation ticks, never wall-clock time.
from .map import SITES  # Local: rendered objective markers and legal planting positions agree.


def resolve_objectives(sim, controls, active, rewards, events):
    """Local: update carrier, interactions and outcomes. Global: one rule order is used by play and training."""
    ids = np.arange(sim.num_envs)  # Local: select each arena's own carrier safely.
    carrier = np.maximum(sim.bomb_carrier, 0)  # Local: sentinel -1 cannot accidentally index player nine.
    carrying = (sim.bomb_carrier >= 0) & ~sim.bomb_planted & active  # Local: only active unplanted carriers move bombs.
    sim.bomb_position[carrying] = sim.positions[ids[carrying], carrier[carrying]]  # Local: include the final death location.
    dropped = carrying & ~sim.alive[ids, carrier]  # Local: killed carriers immediately drop the objective.
    sim.bomb_carrier[dropped] = -1  # Local: dropped bombs can be recovered by a living attacker.
    distances = np.linalg.norm(sim.positions - sim.bomb_position[:, None], axis=-1)  # Local: pickup/defuse ranges.
    pickup = (distances[:, :5] < 1.0) & sim.alive[:, :5]  # Local: one-metre recovery includes ordinary objective stopping distance.
    can_pickup = pickup.any(-1) & (sim.bomb_carrier < 0) & ~sim.bomb_planted & active  # Local: no planted pickups.
    sim.bomb_carrier[can_pickup] = pickup.argmax(-1)[can_pickup]  # Local: smallest slot breaks simultaneous pickup ties.
    carrier = np.maximum(sim.bomb_carrier, 0)  # Local: refresh carrier after recovery.
    carrying = (sim.bomb_carrier >= 0) & ~sim.bomb_planted & active  # Local: updated carrier validity.
    sim.bomb_position[carrying] = sim.positions[ids[carrying], carrier[carrying]]  # Local: recovered bombs follow carriers.
    near_site = np.linalg.norm(sim.positions[:, :, None] - SITES[None, None], axis=-1).min(-1)
    stationary = np.linalg.norm(controls[..., :2], axis=-1) < .1  # Local: movement intent interrupts held interaction.
    held = (controls[..., 6] > .5) & stationary & sim.alive & active[:, None]  # Local: dead players cannot interact.
    planter = (np.arange(10)[None] == sim.bomb_carrier[:, None]) & ~sim.bomb_planted[:, None]
    planter &= near_site <= sim.config.interaction_radius  # Local: carrier must stand in either legal bomb site.
    defuser = (np.arange(10)[None] >= 5) & sim.bomb_planted[:, None] & (distances <= sim.config.interaction_radius)
    valid = held & (planter | defuser)  # Local: only objective-appropriate stationary interactions progress.
    sim.interaction[:] = np.where(active[:, None], np.where(valid, sim.interaction + DT, 0), sim.interaction)
    planted_now = (planter & held & (sim.interaction >= sim.config.plant_seconds - 1e-5)).any(-1)
    defused_now = (defuser & held & (sim.interaction >= sim.config.defuse_seconds - 1e-5)).any(-1)
    was_planted = sim.bomb_planted.copy()  # Local: new plants receive a full countdown beginning next tick.
    sim.bomb_planted[planted_now] = True  # Local: planting is committed before pre-plant timeout evaluation.
    sim.bomb_timer[planted_now] = sim.config.bomb_seconds  # Local: exact full bomb duration on plant completion.
    sim.bomb_carrier[planted_now] = -1  # Local: planted bomb no longer follows any player.
    sim.interaction[planted_now] = 0  # Local: planting progress cannot count toward a later defuse.
    sim.bomb_timer[:] = np.where(was_planted & active, np.maximum(sim.bomb_timer - DT, 0), sim.bomb_timer)
    rewards[:, 0] += .1 * planted_now  # Local: one bounded objective shaping payment per round.
    rewards[:, 1] -= .1 * planted_now  # Local: objective shaping remains exactly zero-sum.
    attackers = sim.alive[:, :5].any(-1)  # Local: elimination checks retain dead slots but inspect health.
    defenders = sim.alive[:, 5:].any(-1)  # Local: defenders cannot leave the armed bomb unresolved by dying.
    exploded = sim.bomb_planted & (sim.bomb_timer < 1e-5)  # Local: tolerate float32 timer rounding.
    timeout = sim.tick * DT >= sim.config.round_seconds - 1e-5  # Local: fixed-tick pre-plant deadline.
    simultaneous = ~attackers & ~defenders & ~sim.bomb_planted  # Local: mutual unplanted elimination draws.
    attack_win = exploded | ~defenders  # Local: bomb explosion or defender elimination normally wins for attack.
    defense_win = ~sim.bomb_planted & (~attackers | timeout)  # Local: armed bombs survive attacker elimination.
    ends = active & (defused_now | attack_win | defense_win | simultaneous)  # Local: do not repeat terminal outcomes.
    winner = np.where(defused_now, 1, np.where(simultaneous, -1, np.where(attack_win, 0, 1)))
    sim.winner[ends] = winner[ends]  # Local: precedence is defuse, mutual draw, attack win, defense win.
    sim.done[ends] = True  # Local: terminal arenas remain frozen until explicitly reset.
    terminal = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0)) * ends  # Local: primary outcome reward.
    rewards[:, 0] += terminal  # Local: attackers receive their signed round result.
    rewards[:, 1] -= terminal  # Local: defenders receive the opposite signed result.
    for arena in np.flatnonzero(planted_now):  # Local: renderer/report events remain compact.
        events.append({"arena": int(arena), "type": "plant", "team": 0})
    for arena in np.flatnonzero(defused_now):  # Local: distinguish objective victories from eliminations.
        events.append({"arena": int(arena), "type": "defuse", "team": 1})
    for arena in np.flatnonzero(exploded & ends & ~defused_now):  # Local: a same-tick completed defuse prevents explosion.
        events.append({"arena": int(arena), "type": "explosion", "team": 0})
    for arena in np.flatnonzero(ends):  # Local: every terminal state emits exactly one round summary.
        reason = "defuse" if defused_now[arena] else "explosion" if exploded[arena] else "mutual_elimination" if simultaneous[arena] else "elimination" if not (attackers[arena] and defenders[arena]) else "timeout"
        events.append({"arena": int(arena), "type": "round_end", "winner": int(winner[arena]), "team": int(winner[arena]), "reason": reason})
