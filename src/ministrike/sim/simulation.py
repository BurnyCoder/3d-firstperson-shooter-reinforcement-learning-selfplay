"""Local: authoritative batched game orchestration. Global: play, training, evaluation and replay share this API.

NumPy generators supply explicit reproducible reset randomness:
https://numpy.org/doc/stable/reference/random/generator.html
"""

from copy import deepcopy  # Local: checkpoints must not alias mutable runtime state.
from dataclasses import asdict, dataclass  # Local: explicit configuration and step-result contracts.

import numpy as np  # Local: CPU batches keep physics independent of renderer and neural-network device.

from .combat import resolve_combat  # Local: isolated weapon phase retains simultaneous damage semantics.
from .constants import FRAME_DIM, HISTORY_LENGTH, GameConfig  # Local: single rule/layout source.
from .map import SITES, SPAWNS  # Local: original map supplies standard and objective curriculum starts.
from .movement import advance_movement  # Local: isolated collision and weapon-clock phase.
from .objectives import resolve_objectives  # Local: explicit objective and terminal state machine.
from .observation import make_frame  # Local: visibility masking remains independent of policy implementation.


@dataclass
class StepResult:
    """Local: immutable-by-convention copied transition data. Global: one interface serves every caller."""

    observations: np.ndarray  # Local: [arena, player, four stacked observation frames].
    rewards: np.ndarray  # Local: [arena, team] exactly zero-sum reward increments.
    terminated: np.ndarray  # Local: [arena] true until an explicit masked reset.
    winner: np.ndarray  # Local: 0 attack, 1 defense, -1 draw or ongoing; consult terminated.
    events: list[dict]  # Local: sparse JSON-compatible combat/objective events for rendering and evaluation.


class Simulation:
    """Local: own batched numeric state. Global: presentation and policy code cannot define game rules."""

    _STATE_ARRAYS = ("positions", "yaw", "pitch", "health", "ammo", "reserve", "cooldown", "reload_timer",
                     "tick", "bomb_position", "bomb_carrier", "bomb_planted", "bomb_timer", "interaction",
                     "done", "winner", "history", "damage_dealt", "shots_fired")  # Local: exhaustive checkpoint arrays.

    def __init__(self, num_envs=1, seed=0, config=None):
        """Local: allocate stable player slots. Global: the same constructor supports interactive and batched use."""
        if not isinstance(num_envs, int) or num_envs < 1:  # Local: invalid batches cannot yield meaningful arenas.
            raise ValueError("num_envs must be a positive integer")  # Local: fail early with actionable feedback.
        self.num_envs = num_envs  # Local: all state arrays share this leading independent-arena dimension.
        self.config = config or GameConfig()  # Local: rule customization is explicit and checkpointed.
        if any(value <= 0 for value in asdict(self.config).values()):  # Local: normalization/timing require positive rules.
            raise ValueError("All GameConfig values must be positive")  # Local: prevent silent divide-by-zero mechanics.
        self.rng = np.random.default_rng(seed)  # Local: dedicated RNG avoids rendering/library RNG interference.
        shape = (num_envs, 10)  # Local: fixed ten-player slots remain throughout each round.
        self.positions = np.zeros((*shape, 3), dtype=np.float32)  # Local: XYZ feet positions in metres.
        self.yaw = np.zeros(shape, dtype=np.float32)  # Local: zero aims toward positive Z.
        self.pitch = np.zeros(shape, dtype=np.float32)  # Local: positive angles look upward.
        self.health = np.zeros(shape, dtype=np.float32)  # Local: alive is derived directly from positive health.
        self.ammo = np.zeros(shape, dtype=np.int32)  # Local: integer bullets currently loaded.
        self.reserve = np.zeros(shape, dtype=np.int32)  # Local: integer bullets available to reload.
        self.cooldown = np.zeros(shape, dtype=np.float32)  # Local: seconds until the rifle can shoot again.
        self.reload_timer = np.zeros(shape, dtype=np.float32)  # Local: seconds until a pending reload completes.
        self.tick = np.zeros(num_envs, dtype=np.int64)  # Local: integer tick counters avoid clock accumulation drift.
        self.bomb_position = np.zeros((num_envs, 3), dtype=np.float32)  # Local: carrier/drop/plant position.
        self.bomb_carrier = np.zeros(num_envs, dtype=np.int64)  # Local: attacker index, or -1 when dropped/planted.
        self.bomb_planted = np.zeros(num_envs, dtype=bool)  # Local: armed objective ignores the pre-plant clock.
        self.bomb_timer = np.zeros(num_envs, dtype=np.float32)  # Local: seconds remaining until explosion.
        self.interaction = np.zeros(shape, dtype=np.float32)  # Local: uninterrupted stationary hold duration.
        self.done = np.zeros(num_envs, dtype=bool)  # Local: completed arenas hold until masked reset.
        self.winner = np.full(num_envs, -1, dtype=np.int64)  # Local: draw/ongoing sentinel disambiguated by done.
        self.history = np.zeros((*shape, HISTORY_LENGTH, FRAME_DIM), dtype=np.float32)  # Local: oldest frame first.
        self.damage_dealt = np.zeros((num_envs, 2), dtype=np.float32)  # Local: effective damage totals by team.
        self.shots_fired = np.zeros(shape, dtype=np.int64)  # Local: per-player firing totals support evaluation.
        self.reset()  # Local: construction returns a ready-to-step arena batch.

    @property
    def alive(self):
        """Local: derive life from health. Global: no duplicated alive state can disagree with combat."""
        return self.health > 0  # Local: stable slots stay present even when this mask becomes false.

    def reset(self, seed=None, scenario="standard", mask=None):
        """Local: initialize selected arenas only. Global: curriculum and vectorized RL use the same reset path."""
        if scenario not in {"standard", "combat", "objective"}:  # Local: avoid silently misspelled curricula.
            raise ValueError("scenario must be standard, combat, or objective")  # Local: clear supported starts.
        selected = np.ones(self.num_envs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != (self.num_envs,):  # Local: ambiguous broadcast resets could corrupt unrelated arenas.
            raise ValueError("reset mask must have shape (num_envs,)")  # Local: require exact batch selection.
        if seed is not None:  # Local: explicit seeds restart the dedicated reset RNG reproducibly.
            self.rng = np.random.default_rng(seed)  # Local: no global NumPy random state is modified.
        ids = np.flatnonzero(selected)  # Local: all mutation below is indexed by selected arenas.
        self.positions[ids] = SPAWNS  # Local: baseline starts have two separated five-player teams.
        jitter = self.rng.uniform(-.3, .3, (len(ids), 10, 2))  # Local: small valid start diversity, never hidden physics noise.
        self.positions[ids, :, 0] += jitter[..., 0]  # Local: vary initial east-west placement.
        self.positions[ids, :, 2] += jitter[..., 1]  # Local: vary initial north-south placement.
        self.yaw[ids] = np.array([0] * 5 + [np.pi] * 5)  # Local: teams initially face toward the opposing half.
        self.pitch[ids] = 0  # Local: aim begins horizontally.
        self.health[ids] = self.config.health  # Local: every new round starts all players alive.
        self.ammo[ids] = self.config.magazine  # Local: refill rifles exactly once per round.
        self.reserve[ids] = self.config.reserve_ammo  # Local: reset finite spare ammunition.
        self.cooldown[ids] = 0  # Local: rifles are ready on the first tick.
        self.reload_timer[ids] = 0  # Local: no reload carries across rounds.
        self.tick[ids] = 0  # Local: pre-plant clock starts at zero.
        self.bomb_carrier[ids] = 0  # Local: a stable attacker slot starts carrying the bomb.
        self.bomb_planted[ids] = False  # Local: every scenario starts with an unplanted objective by default.
        self.bomb_timer[ids] = 0  # Local: explosion timer starts only when planted.
        self.interaction[ids] = 0  # Local: no held objective progress survives reset.
        self.done[ids] = False  # Local: selected arenas resume stepping.
        self.winner[ids] = -1  # Local: clear previous outcome.
        self.damage_dealt[ids] = 0  # Local: statistics refer only to the current round.
        self.shots_fired[ids] = 0  # Local: reset inactivity/shot counters.
        self._reset_scenario(ids, scenario)  # Local: curriculum placement is a distinct readable phase.
        self.bomb_position[ids] = self.positions[ids, 0]  # Local: carried objective starts at carrier feet.
        planted = ids[self.bomb_planted[ids]]  # Local: objective curriculum can start in a post-plant state.
        self.bomb_position[planted] = SITES[0]  # Local: planted curriculum bomb is fixed at site A.
        frame = make_frame(self)  # Local: reset initializes history without fictitious pre-reset observations.
        self.history[ids] = frame[ids, :, None]  # Local: repeat initial frame into all history slots.
        return self._stack_observations(frame)  # Local: reuse computed vision while returning fresh independently owned observations.

    def _reset_scenario(self, ids, scenario):
        """Local: implement prescribed starts. Global: both experimental conditions use identical curricula."""
        if scenario == "combat":  # Local: unobstructed short-range starts teach elementary aiming and movement.
            self.positions[ids, :5, 0] = np.linspace(-11, -5, 5)  # Local: spread actors across western lane.
            self.positions[ids, 5:, 0] = np.linspace(-11, -5, 5)  # Local: corresponding defenders face attackers.
            self.positions[ids, :5, 2] = -5.5  # Local: attackers start beyond spawn protection for intentionally immediate combat.
            self.positions[ids, 5:, 2] = -3  # Local: opposing row is 2.5m away, before the low cover begins at Z=-2.
        elif scenario == "objective":  # Local: balanced mix of close planting and post-plant defusing starts.
            self.positions[ids, :5, 0] = np.linspace(-8, -4, 5)  # Local: attacker carrier starts exactly at site A.
            self.positions[ids, :5, 2] = 8  # Local: other attackers can defend the objective.
            self.positions[ids, 5:, 0] = np.linspace(-12, -8, 5)  # Local: defenders approach the same objective.
            self.positions[ids, 5:, 2] = 10  # Local: defenders start near, but outside immediate defuse range.
            planted = ids[self.rng.random(len(ids)) < .5]  # Local: predetermined 50/50 subtask distribution.
            self.bomb_planted[planted] = True  # Local: half of objective starts teach post-plant behavior.
            self.bomb_carrier[planted] = -1  # Local: an armed bomb has no carrier.
            self.bomb_timer[planted] = self.config.bomb_seconds  # Local: post-plant starts receive the full countdown.

    def step(self, controls):
        """Local: run one fixed-tick transition. Global: this wrapper exposes clearly named rule phases."""
        controls = np.asarray(controls, dtype=np.float32)  # Local: normalize controller transport dtype.
        if controls.shape != (self.num_envs, 10, 7) or not np.isfinite(controls).all():
            raise ValueError("controls must be finite with shape (num_envs, 10, 7)")  # Local: reject invalid policies.
        active = ~self.done  # Local: terminal arenas stay frozen until their caller explicitly resets them.
        rewards = np.zeros((self.num_envs, 2), dtype=np.float32)  # Local: per-tick reward increments start at zero.
        events = []  # Local: sparse events belong only to this transition.
        advance_movement(self, controls, active)  # Local: rotate/move before shooting and objective interaction.
        resolve_combat(self, controls, active, rewards, events)  # Local: all shots resolve simultaneously.
        self.tick += active  # Local: only running arenas advance their pre-plant clock.
        resolve_objectives(self, controls, active, rewards, events)  # Local: objective state defines terminal precedence.
        for event in events:  # Local: all sparse events use the transition's authoritative integer tick.
            event["tick"] = int(self.tick[event["arena"]])  # Local: replay/report consumers do not infer event timing.
        frame = make_frame(self)  # Local: observations describe the fully resolved transition.
        self.history[active, :, :-1] = self.history[active, :, 1:]  # Local: roll history only for advancing arenas.
        self.history[active, :, -1] = frame[active]  # Local: append current observable state after all rule phases.
        observations = self.history.reshape(self.num_envs, 10, -1).copy()  # Local: consumers cannot mutate history.
        return StepResult(observations, rewards, self.done.copy(), self.winner.copy(), events)

    def observe(self):
        """Local: obtain independent history with fresh current frame. Global: observation reads never advance RNG/time."""
        return self._stack_observations(make_frame(self))  # Local: explicit reads remain fresh after deliberate state interventions.

    def _stack_observations(self, frame):
        """Local: combine owned history with an already computed frame. Global: reset and explicit reads share exact output semantics."""
        history = self.history.copy()  # Local: external controllers cannot corrupt stored history.
        history[:, :, -1] = frame  # Local: refresh all returned current frames without mutating unselected arena history.
        return history.reshape(self.num_envs, 10, -1)  # Local: flatten oldest-to-newest four-frame actor input.

    def snapshot(self, arena=0):
        """Local: expose JSON-serializable public state. Global: renderer/replay/report tooling can inspect one arena."""
        return {name: getattr(self, name)[arena].tolist() for name in self._STATE_ARRAYS if name != "history"}

    def state_dict(self):
        """Local: copy physics, history and RNG. Global: resuming training preserves exact simulator continuation."""
        return {"num_envs": self.num_envs, "config": asdict(self.config), "rng": deepcopy(self.rng.bit_generator.state),
                "arrays": {name: getattr(self, name).copy() for name in self._STATE_ARRAYS}}

    def load_state_dict(self, state):
        """Local: restore a compatible checkpoint. Global: reject implicit architecture/rule changes on resume."""
        if state["num_envs"] != self.num_envs or state["config"] != asdict(self.config):
            raise ValueError("Simulator checkpoint batch size or rules differ")  # Local: reproducibility requires both.
        for name in self._STATE_ARRAYS:  # Local: restore every declared state field, including observation history.
            value = np.asarray(state["arrays"][name])  # Local: accept serialized arrays without aliasing them.
            if value.shape != getattr(self, name).shape:  # Local: corrupt array dimensions must fail visibly.
                raise ValueError(f"Invalid checkpoint shape for {name}")  # Local: identify the incompatible field.
            getattr(self, name)[:] = value  # Local: retain stable object arrays while restoring exact contents.
        self.rng.bit_generator.state = deepcopy(state["rng"])  # Local: next randomized reset follows saved trajectory.
