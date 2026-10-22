"""Local: coordinate input, fixed simulation ticks, graphics and HUD; global: one playable game pipeline.

Ursina's documented manual loop: https://www.ursinaengine.org/api_reference.html#Ursina
Screenshots: https://docs.panda3d.org/1.10/python/reference/panda3d.core.GraphicsOutput#panda3d.core.GraphicsOutput.saveScreenshot
"""

from datetime import datetime, timezone  # Local: timestamp manual screenshots consistently across machines.
import json  # Local: save replay controls and final state beside finite-run screenshots.
import logging  # Local: use the root application's timestamped terminal/file logging configuration.
import math  # Local: convert the requested simulation duration to an exact tick count.
from pathlib import Path  # Local: create output directories without relying on the caller's working shell.
import time  # Local: measure presentation time with a monotonic clock independent of simulation ticks.

import numpy as np  # Local: copy and batch human/bot physical actions for the canonical simulator.

from ministrike.policies.scripted import ScriptedPolicy  # Local: usable, honestly labeled baseline before training.
from ministrike.sim import Simulation  # Local: the only authority for movement, hits, objectives and round outcomes.
from ministrike.sim.constants import DT  # Local: the renderer accumulates time against the same fixed tick duration.
from ministrike.view.controls import HumanControls  # Local: isolate frame-rate-independent human input translation.
from ministrike.view.state import ViewControlState  # Local: switch camera/ownership without modifying match state.

LOGGER = logging.getLogger(__name__)  # Local: viewer output joins the run's timestamped logs.


class ViewSession:
    """Local: coordinate a single native window session; global: leave all gameplay in shared Simulation."""

    def __init__(self, mode, checkpoint, seed, team, seconds, screenshot_path, spectator_first_person=False, policy=None):
        """Local: prepare match and policy before rendering; global: keep provenance explicit and replayable."""
        from ursina import Entity, camera, mouse  # Local: avoid importing a display engine for headless training.
        from ministrike.view.hud import MatchHUD  # Local: defer GUI-only modules until after the window exists.
        from ministrike.view.scene import ArenaScene  # Local: defer graphics primitives until the app is initialized.

        self.mode, self.seed = mode, seed  # Local: preserve the original match mode and reproducible seed.
        self.human = int(team) * 5  # Local: a human replaces one of the requested team's five permanent slots.
        self.control_state = ViewControlState.create(mode, team, spectator_first_person)  # Local: separate selection from ownership.
        self.simulation = Simulation(num_envs=1, seed=seed)  # Local: one authoritative arena feeds the native viewer.
        self.observations = self.simulation.reset(seed=seed, scenario="standard")  # Local: record a known initial state.
        self.policy = policy if policy is not None else self._load_policy(checkpoint)  # Local: reuse pre-window-loaded weights.
        self.controls = HumanControls()  # Local: aggregate mouse movement until the next fixed tick.
        self.arena = ArenaScene()  # Local: shared map arrays provide all visible obstacles.
        policy_label = f"CHECKPOINT / {Path(checkpoint).stem[:24]}" if checkpoint else "SCRIPTED BASELINE / UNTRAINED"
        self.hud = MatchHUD(mode, policy_label, self.human)  # Local: expose policy provenance in the visible interface.
        self.scores = [0, 0]  # Local: retain team round wins across resets inside this window session.
        self.round_number = 1  # Local: the HUD uses one-based numbering for human readers.
        self.round_deadline = None  # Local: a short presentation intermission follows a terminal simulation state.
        self.notice = ""  # Local: show one concise winner announcement during that intermission.
        self.paused = False  # Local: pausing stops local simulation ticks and releases mouse capture.
        self.finished = False  # Local: manual app.step can terminate without calling process-wide sys.exit.
        self.accumulator = 0.0  # Local: retain fractional elapsed time between twenty-Hertz simulation ticks.
        self.last_frame = time.perf_counter()  # Local: a monotonic clock avoids wall-clock adjustment artifacts.
        self.started = self.last_frame  # Local: cosmetic effect clocks share one presentation-time origin.
        self.total_ticks = 0  # Local: finite debugging sessions stop at an exact simulation-step count.
        self.target_ticks = max(1, math.ceil(seconds / DT)) if seconds is not None else None  # Local: exact duration.
        self.screenshot_path = Path(screenshot_path) if screenshot_path else None  # Local: retain requested capture path.
        self.replay = [] if seconds is not None else None  # Local: record only bounded sessions, avoiding unbounded RAM use.
        self.handler = Entity(update=self.update, input=self.input)  # Local: documented entity callbacks enter this wrapper.
        camera.fov = 90  # Local: familiar shooter perspective; observation visibility still follows simulation rules.
        camera.clip_plane_near = 0.03  # Local: the original rifle viewmodel remains visible close to the camera.
        mouse.locked = False  # Local: initialize uncaptured so the ownership synchronizer centers the first FPS capture.
        mouse.visible = True  # Local: spectator sessions leave the system cursor usable from their first frame.
        self._sync(time.perf_counter())  # Local: populate the first rendered frame before the manual loop starts.
        LOGGER.info("view_started mode=%s seed=%s policy=%s", mode, seed, policy_label)  # Local: provenance in logs.

    @staticmethod
    def _load_policy(checkpoint):
        """Local: load explicit learned weights or a baseline; global: never silently substitute failed checkpoints."""
        if checkpoint:  # Local: missing or incompatible user-selected checkpoints must surface as normal errors.
            from ministrike.learning.checkpoint import PolicyCheckpoint  # Local: load torch only for learned inference.
            return PolicyCheckpoint(checkpoint)  # Local: the shared wrapper decodes categorical actions for gameplay.
        return ScriptedPolicy()  # Local: the deterministic scripted baseline makes the initial game immediately usable.

    def update(self):
        """Local: accumulate frame time and step the simulator; global: rendering cannot change simulation physics."""
        from ursina import held_keys, mouse  # Local: read the frame's pointer delta and spectator movement keys.

        now = time.perf_counter()  # Local: use presentation time solely for tick scheduling and cosmetic lifetimes.
        elapsed = min(now - self.last_frame, 0.25)  # Local: cap catch-up after window dragging or a graphics stall.
        self.last_frame = now  # Local: the next frame measures its own elapsed duration.
        if self.paused or self.finished:  # Local: neither pause nor completion advances canonical match state.
            self._sync(now)  # Local: keep the pause text and cursor state visible while frozen.
            return  # Local: do not accrue a backlog of paused simulation time.
        if self.control_state.controlled is not None and mouse.locked:
            self.controls.add_mouse(float(mouse.velocity.x), float(mouse.velocity.y))  # Local: defer input to a tick.
        elif self.control_state.followed is None:  # Local: spectator movement only moves the cosmetic free camera.
            delta = (float(mouse.velocity.x), float(mouse.velocity.y)) if held_keys["right mouse"] else (0, 0)
            self.arena.move_free_camera(elapsed, held_keys, delta)  # Local: no actor or physical state changes here.
        if self.round_deadline is not None:  # Local: terminal states remain visible briefly before the next reset.
            if now >= self.round_deadline:  # Local: intermission duration affects presentation only.
                self._new_round()  # Local: all gameplay state is reset through the shared simulation API.
            self._sync(now)  # Local: display the final round state or the newly reset state.
            return  # Local: no terminal-state step is accidentally counted as new gameplay.
        self.accumulator += elapsed  # Local: integrate real elapsed time, independent of rendering frame rate.
        while self.accumulator >= DT and not self.finished and self.round_deadline is None:
            self.accumulator -= DT  # Local: consume one fixed-size simulation interval.
            self._step(now)  # Local: exactly one shared transition owns movement, combat, and objectives.
        self._sync(now)  # Local: copy the latest canonical state to graphics after all due ticks complete.

    def _step(self, now):
        """Local: compose bot and optional human controls; global: every actor uses the same physical action API."""
        from ursina import held_keys  # Local: read held actions only at simulation-tick boundaries.

        actions = np.asarray(self.policy.act(self.observations[0]), dtype=np.float32).copy()  # Local: bot controls.
        controlled = self.control_state.controlled  # Local: F may have transferred human input to any live operator.
        if controlled is not None:  # Local: replace exactly one policy slot while keeping both teams at five players.
            actions[controlled] = self.controls.consume(held_keys)  # Local: preserve this operator's existing physical state.
        previous_ammo = self.simulation.ammo[0].copy()  # Local: a confirmed ammo decrease drives cosmetic muzzle flashes.
        if self.replay is not None:  # Local: finite runs preserve exact inputs for rendered/headless parity checks.
            self.replay.append({"round": self.round_number, "actions": actions.tolist()})  # Local: no rounding/truncation.
        result = self.simulation.step(actions[None])  # Local: add the arena batch dimension required by Simulation.
        self.observations = result.observations  # Local: policies receive only the simulator's masked actor observations.
        self.control_state.refresh(self.simulation.alive[0])  # Local: death hands control back and selects a living spectator target.
        self.total_ticks += 1  # Local: count actual transitions, never render frames.
        self.arena.register_shots(previous_ammo, self.simulation.ammo[0], now)  # Local: display actual successful fire.
        for event in result.events:  # Local: record meaningful visible game events in timestamped application logs.
            if event.get("type") not in {"shot", "fire"}:  # Local: avoid flooding ordinary play with repeated rifle requests.
                LOGGER.info("match_event %s", json.dumps(event, default=str))  # Local: preserve complete event contents.
        if bool(result.terminated[0]):  # Local: the simulation decides the winner before the viewer starts intermission.
            winner = int(result.winner[0])  # Local: canonical zero/one team IDs determine scoreboard updates.
            if winner in (0, 1):  # Local: a draw increments neither team's round win total.
                self.scores[winner] += 1  # Local: retain round wins separately from resettable simulation fields.
            self.notice = ("ATTACKERS WIN" if winner == 0 else "DEFENDERS WIN") if winner >= 0 else "ROUND DRAW"
            self.round_deadline = now + 2.5  # Local: display outcome before the next reproducible round starts.
        if self.target_ticks is not None and self.total_ticks >= self.target_ticks:
            self.finished = True  # Local: deterministic finite-run completion permits exact screenshot/replay checks.

    def _new_round(self):
        """Local: start another seeded match; global: round transitions never reset physics by hand."""
        self.round_number += 1  # Local: derive the next seed from a stable, logged round index.
        seed = self.seed + self.round_number - 1  # Local: each round remains independently reproducible.
        self.observations = self.simulation.reset(seed=seed, scenario="standard")  # Local: canonical full reset.
        self.round_deadline, self.notice = None, ""  # Local: clear presentation-only terminal state.
        self.accumulator = 0.0  # Local: do not apply an intermission's elapsed time to the next round.
        self.controls.clear()  # Local: discard aim motion captured before the reset.
        self.control_state.new_round(self.simulation.alive[0])  # Local: explicit play respawns; watch takeovers remain temporary.
        LOGGER.info("round_started round=%s seed=%s", self.round_number, seed)  # Local: reproduce round transitions.

    def _sync(self, now):
        """Local: refresh view and overlay; global: all synchronization reads simulation fields without writing."""
        self.control_state.refresh(self.simulation.alive[0])  # Local: never display a dead first-person selection.
        self._sync_mouse()  # Local: pointer capture follows current ownership or explicit free-camera look input.
        self.arena.sync(self.simulation, self.control_state.followed, now)  # Local: synchronize graphical entities only.
        self.hud.sync(self.simulation, self.scores, self.round_number, self.control_state.followed, self.paused,
                      self.notice, self.control_state.controlled, self.control_state.selected)  # Local: show current live mode.

    def _sync_mouse(self):
        """Local: capture mouse only for human aim or held spectator look; global: camera changes preserve actor state."""
        from ursina import held_keys, mouse  # Local: use the engine's current key/cursor state.

        free_look = self.control_state.followed is None and bool(held_keys["right mouse"])  # Local: deliberate RMB look.
        locked = not self.paused and (self.control_state.controlled is not None or free_look)  # Local: no passive capture.
        if mouse.locked != locked:  # Local: avoid resetting the cursor position unnecessarily every frame.
            self.controls.clear()  # Local: ownership/capture changes must not apply the previous mode's mouse motion.
            if locked:  # Local: Ursina 8 confines without recentering when its locked property is first set.
                mouse.position = (0, 0)  # Local: prevent the pre-capture desktop cursor offset from becoming a large aim jump.
            mouse.locked = locked  # Local: lock/unlock the system pointer only on a real state transition.
            mouse.visible = not locked  # Local: preserve a usable pointer while passively watching bots.

    def input(self, key):
        """Local: handle window-level controls; global: gameplay keys still pass through HumanControls and Simulation."""
        if key == "escape":  # Local: pause/unpause is a predictable reversible local UI action.
            self.paused = not self.paused  # Local: the update loop owns whether simulation ticks may run.
            self.controls.clear()  # Local: avoid applying stale movement when resuming.
            self._sync_mouse()  # Local: capture follows current input ownership after a pause transition.
        elif key == "f10":  # Local: give users an explicit exit shortcut while Escape is reserved for pause.
            self.finished = True  # Local: the manual loop exits and cleanly destroys the graphics window.
        elif key == "tab":  # Local: spectators cycle every living player across both teams.
            self.control_state.cycle(self.simulation.alive[0])  # Local: human ownership deliberately prevents detached selection.
        elif key == "v":  # Local: switch between a selected bot's eyes and the freely movable spectator camera.
            self.control_state.toggle_view()  # Local: this changes presentation only and is ignored while controlling a player.
        elif key == "f":  # Local: take over the selected live bot or return that bot to the policy immediately.
            self.control_state.toggle_control(self.simulation.alive[0])  # Local: never reset position, health, ammo or bomb state.
            self.controls.clear()  # Local: clear old aiming deltas when the action source changes.
            self._sync_mouse()  # Local: newly controlled operators immediately receive ordinary FPS mouse input.
            LOGGER.info("input_ownership controlled=%s selected=%s", self.control_state.controlled,
                        self.control_state.selected)  # Local: record live takeover/hand-back events without altering the match.
        elif key == "f12":  # Local: let users capture gameplay without external screenshot tooling.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # Local: portable, chronologically sortable filename.
            self.capture(Path("artifacts/screenshots") / f"ministrike-{stamp}.png")  # Local: stay inside this workspace.

    def capture(self, path):
        """Local: write a real framebuffer PNG and optional replay; global: make visual verification reproducible."""
        from panda3d.core import Filename  # Local: use Panda3D's portable path wrapper for the screenshot API.
        from ursina import application  # Local: access the current native graphics buffer without a second window.

        path = Path(path).resolve()  # Local: log and return the exact output destination.
        path.parent.mkdir(parents=True, exist_ok=True)  # Local: create only the caller's requested screenshot folder.
        application.base.graphicsEngine.renderFrame()  # Local: flush the newest synchronized entities to the framebuffer.
        if not application.base.win.saveScreenshot(Filename.from_os_specific(str(path))):
            raise RuntimeError(f"Could not save screenshot to {path}")  # Local: do not claim a capture that failed.
        metadata = {"seed": self.seed, "mode": self.mode, "ticks": self.total_ticks,
                    "round": self.round_number, "controlled": self.control_state.controlled,
                    "selected": self.control_state.selected, "snapshot": self.simulation.snapshot(arena=0)}  # Local: provenance.
        if self.replay is not None:  # Local: bounded debug captures include untruncated inputs for deterministic verification.
            metadata["replay"] = self.replay  # Local: retain every actor action from every recorded transition.
        path.with_suffix(".replay.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")  # Local: exact replay.
        LOGGER.info("screenshot_saved path=%s ticks=%s", path, self.total_ticks)  # Local: expose the inspectable output.
        return str(path)  # Local: the CLI can print the real file path to its user.


def run_view(mode="play", checkpoint=None, seed=0, team=0, seconds=None, screenshot_path=None, spectator_first_person=False):
    """Local: launch the native viewer phases; global: provide one small public play/watch entry point."""
    if mode not in {"play", "watch"} or team not in {0, 1}:  # Local: fail early on unsupported public options.
        raise ValueError("mode must be play/watch and team must be 0/1")  # Local: keep user errors explicit.
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):  # Local: bounded debug duration must be valid.
        raise ValueError("seconds must be a finite positive number")  # Local: reject accidental indefinite smoke runs.
    policy = ViewSession._load_policy(checkpoint)  # Local: load torch's native DLLs before Panda3D opens a Windows graphics context.
    # Global: native import-order failure context: https://github.com/pytorch/pytorch/issues/166628
    from panda3d.core import loadPrcFileData  # Local: configure the native graphics engine only for GUI commands.
    loadPrcFileData("", "audio-library-name null")  # Local: no sound assets are needed; avoid unavailable audio drivers.
    from ursina import Ursina, color, mouse, window  # Local: keep training/import checks independent of a display server.

    app = Ursina(title="MiniStrike | Self-play Lab", size=(1440, 810), fullscreen=False,
                 borderless=False, development_mode=False, editor_ui_enabled=False, vsync=True)  # Local: native window.
    window.color = color.rgb32(24, 37, 47)  # Local: coherent navy background beyond the original industrial arena.
    window.exit_button.visible = False  # Local: F10 and the native titlebar are the documented exits.
    window.fps_counter.enabled = False  # Local: keep debugging counters out of the playable game's interface.
    try:  # Local: always release native resources, including on a policy-loading or screenshot error.
        session = ViewSession(mode, checkpoint, seed, team, seconds, screenshot_path, spectator_first_person, policy)  # Local: phases.
        while not session.finished:  # Local: documented manual loop permits returning normally to the CLI.
            app.step()  # Local: Ursina dispatches input/update callbacks and draws the next frame.
            time.sleep(0.001)  # Local: yield briefly to the OS rather than saturating a CPU with event-loop polling.
        result = session.capture(session.screenshot_path) if session.screenshot_path else None  # Local: final verified frame.
        LOGGER.info("view_finished ticks=%s rounds=%s", session.total_ticks, session.round_number)  # Local: completion evidence.
        return result  # Local: an optional screenshot file is the viewer's only public return value.
    finally:  # Local: cleanup must also happen when users close the native window or an exception occurs.
        mouse.locked = False  # Local: never leave the user's pointer captured after the viewer ends.
        app.destroy()  # Local: release Panda3D window, graphics and native input resources.
