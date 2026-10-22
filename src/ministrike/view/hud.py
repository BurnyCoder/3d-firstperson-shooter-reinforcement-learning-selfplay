"""Local: display match information and controls; global: make the starter game usable without docs.

UI primitives: https://www.ursinaengine.org/api_reference.html#Text
The human minimap deliberately excludes opponent positions behind walls.
"""

import numpy as np  # Local: project known world coordinates onto the optional tactical minimap.
from ursina import Entity, Text, camera, color  # Local: construct readable screen-space panels and labels.
from ursina.shaders import unlit_shader  # Local: keep interface colors independent of arena lighting.

from ministrike.sim.constants import DT  # Local: turn canonical tick counters into the displayed round time.
from ministrike.sim.map import SITES, WALLS  # Local: show only public, fixed map geometry.
from ministrike.view.scene import ATTACK, DEFEND, INK  # Local: share one team palette with the 3D operators.


def panel(position, scale, tint=INK, parent=None):
    """Local: create a simple flat UI panel; global: reuse consistent HUD contrast and layering."""
    return Entity(parent=parent or camera.ui, model="quad", position=position, scale=scale,
                  color=tint, shader=unlit_shader)  # Local: screen-space geometry needs no game collider.


def label(text, position, scale=0.8, tint=None, centered=False, parent=None):
    """Local: create a text label; global: keep every HUD region aligned through one reusable helper."""
    return Text(text=text, parent=parent or camera.ui, position=position, scale=scale,
                origin=(0, 0) if centered else (-0.5, 0.5), color=tint or color.rgb32(225, 235, 238))  # Local: readable.


class MatchHUD:
    """Local: manage all overlay elements; global: state is read from the authoritative simulation only."""

    def __init__(self, mode, policy_label, human):
        """Local: create static headings and dynamic labels; global: communicate controls and bot provenance."""
        self.mode, self.human = mode, human  # Local: distinguish spectator information from human information.
        panel((-0.849, 0.445, 0), (0.007, 0.06), ATTACK)  # Local: small brand accent frames the main title.
        label("MINISTRIKE", (-0.827, 0.478), 1.2)  # Local: original game name; global: no Counter-Strike assets.
        label("SELF-PLAY LAB  /  SECTOR 01", (-0.825, 0.435), 0.53, color.rgb32(150, 174, 185))  # Local: subtitle.
        panel((0, 0.44, 0.1), (0.60, 0.083))  # Local: scoreboard background does not cover gameplay below.
        self.attack_score = label("ATTACK  0", (-0.245, 0.465), 0.88, ATTACK)  # Local: persistent team score.
        self.defend_score = label("0  DEFEND", (0.245, 0.452), 0.88, DEFEND, True)  # Local: opposing score.
        self.clock = label("01:00", (0, 0.447), 1.1, centered=True)  # Local: round or bomb countdown.
        self.alive = label("5 ALIVE       ROUND 01       5 ALIVE", (0, 0.393), 0.57, centered=True)  # Local: teams.
        self.mode_label = label(mode.upper(), (0.50, 0.474), 0.7, DEFEND)  # Local: show camera and control ownership live.
        self.policy = label(policy_label, (0.50, 0.445), 0.52)  # Local: never imply the scripted baseline is learned.
        panel((-0.689, -0.414, 0.1), (0.33, 0.096))  # Local: lower-left player status panel.
        self.health = label("100", (-0.824, -0.384), 1.65)  # Local: health is the largest survival indicator.
        self.health_caption = label("HEALTH", (-0.712, -0.405), 0.53, DEFEND)  # Local: textual status meaning.
        self.player_name = label("ATTACKER 01", (-0.824, -0.449), 0.50, ATTACK)  # Local: current followed operator.
        panel((0.689, -0.414, 0.1), (0.33, 0.096))  # Local: symmetrical ammunition panel.
        self.ammo = label("30 / 90", (0.55, -0.386), 1.35)  # Local: magazine and reserve remain distinct.
        self.weapon = label("RIFLE  /  R RELOAD", (0.55, -0.441), 0.50, color.rgb32(162, 183, 191))  # Local: action hint.
        self.objective = label("ATTACK: HOLD E AT A SITE TO PLANT", (0, -0.352), 0.64, centered=True)  # Local: task.
        self.interaction_bg = panel((0, -0.397, 0), (0.26, 0.009), color.rgb32(44, 65, 73))  # Local: progress track.
        self.interaction_fill = panel((-0.13, -0.397, -0.01), (0.001, 0.009), ATTACK)  # Local: held E progress.
        self.interaction_fill.origin_x = -0.5  # Local: growth proceeds from the left edge of the progress track.
        self.notice = label("", (0, 0.14), 1.35, centered=True)  # Local: round winner and pause notification.
        self.subnotice = label("", (0, 0.09), 0.65, centered=True)  # Local: short next-action instruction.
        self.hint = label("", (0, -0.481), 0.51, color.rgb32(172, 192, 201), True)  # Local: always-visible key legend.
        self.crosshair = Entity(parent=camera.ui)  # Local: group crosshair strokes for one visibility toggle.
        for x, y, width, height in ((-0.008, 0, 0.007, 0.0015), (0.008, 0, 0.007, 0.0015),
                                     (0, -0.008, 0.0015, 0.007), (0, 0.008, 0.0015, 0.007)):
            panel((x, y, 0), (width, height), color.white, self.crosshair)  # Local: open center preserves aiming detail.
        self._build_minimap()  # Local: orient the user with a compact schematic derived from the shared map.

    def _build_minimap(self):
        """Local: draw known map geometry; global: avoid revealing hidden opponent dynamics to a human."""
        self.map_root = Entity(parent=camera.ui, position=(0.71, 0.265))  # Local: upper-right tactical inset.
        panel((0, 0, 0.15), (0.26, 0.26), INK, self.map_root)  # Local: dark panel keeps tiny markers readable.
        self.low = np.min(WALLS[:, [0, 2]], axis=0)  # Local: extract only horizontal map bounds.
        self.span = np.max(WALLS[:, [3, 5]], axis=0) - self.low  # Local: normalize world X/Z into panel units.
        for wall in WALLS:  # Local: show each public obstacle without creating a separate map definition.
            midpoint = (wall[[0, 2]] + wall[[3, 5]]) / 2  # Local: derive the top-down obstacle center.
            location = (midpoint - self.low) / self.span * 0.23 - 0.115  # Local: keep an inner padding margin.
            size = (wall[[3, 5]] - wall[[0, 2]]) / self.span * 0.23  # Local: scale to consistent map coordinates.
            panel((*location, 0.05), tuple(size), color.rgb32(74, 94, 105), self.map_root)  # Local: static wall footprint.
        for name, site in zip(("A", "B"), SITES, strict=True):  # Local: label the two objectives on the inset.
            location = (site[[0, 2]] - self.low) / self.span * 0.23 - 0.115  # Local: use the same projection as walls.
            label(name, (*location, -0.02), 0.45, ATTACK, True, self.map_root)  # Local: visible public objective.
        self.dots = [panel((0, 0, -0.05), (0.006, 0.006), ATTACK if i < 5 else DEFEND, self.map_root)
                     for i in range(10)]  # Local: retain reusable markers for all permanent player slots.

    def sync(self, simulation, scores, round_number, followed, paused, notice="", controlled=None, selected=None):
        """Local: update HUD text and markers; global: display canonical timers, health, and objective state."""
        attacker_count = int(simulation.alive[0, :5].sum())  # Local: count living permanent attacker slots.
        defender_count = int(simulation.alive[0, 5:].sum())  # Local: count living permanent defender slots.
        self.attack_score.text = f"ATTACK  {scores[0]}"  # Local: keep match wins across round resets.
        self.defend_score.text = f"{scores[1]}  DEFEND"  # Local: report the defender's independently accumulated wins.
        self.alive.text = f"{attacker_count} ALIVE       ROUND {round_number:02d}       {defender_count} ALIVE"  # Local: team status.
        remaining = float(simulation.bomb_timer[0]) if simulation.bomb_planted[0] else (
            simulation.config.round_seconds - int(simulation.tick[0]) * DT)  # Local: use bomb time after planting.
        seconds = max(0, int(np.ceil(remaining)))  # Local: never show negative time at the terminal tick.
        self.clock.text = f"{seconds // 60:02d}:{seconds % 60:02d}"  # Local: familiar MM:SS countdown.
        self.clock.color = ATTACK if simulation.bomb_planted[0] else color.white  # Local: armed bomb warning.
        player = (selected if selected is not None else self.human) if followed is None else followed  # Local: current target.
        team = player // 5  # Local: canonical slot ordering decides attacker/defender labels.
        operator = f"{'ATTACKER' if team == 0 else 'DEFENDER'} {player % 5 + 1:02d}"  # Local: name the selected bot explicitly.
        viewpoint = "YOU" if controlled is not None else ("BOT POV" if followed is not None else "FREE CAMERA")
        self.mode_label.text = f"{viewpoint} / {operator}"  # Local: show whether the user is watching or driving this actor.
        self.health.text = str(max(0, int(simulation.health[0, player])))  # Local: display actual health only.
        self.player_name.text = f"{'ATTACKER' if team == 0 else 'DEFENDER'} {player % 5 + 1:02d}"  # Local: operator ID.
        self.player_name.color = ATTACK if team == 0 else DEFEND  # Local: consistent affiliation across modes.
        self.ammo.text = f"{int(simulation.ammo[0, player]):02d} / {int(simulation.reserve[0, player]):02d}"  # Local: ammo.
        reloading = float(simulation.reload_timer[0, player])  # Local: read the simulation-controlled reload state.
        self.weapon.text = f"RELOADING  {reloading:.1f}s" if reloading > 0 else "RIFLE  /  R RELOAD"  # Local: feedback.
        if followed is None:  # Local: the tactical view summarizes both teams instead of an arbitrary dead operator.
            self.health.text = f"{attacker_count} / 5"  # Local: overview status counts attackers still in the round.
            self.health_caption.text = "ALIVE"  # Local: do not mislabel a team count as an individual health value.
            self.player_name.text = "ATTACKERS  /  TACTICAL VIEW"  # Local: make the overview's context explicit.
            self.ammo.text = f"{defender_count} / 5"  # Local: the matching right panel counts living defenders.
            self.weapon.text = "DEFENDERS  /  TACTICAL VIEW"  # Local: avoid showing arbitrary ammunition while not following.
        else:  # Local: first-person and follow modes keep the selected actor's health label.
            self.health_caption.text = "HEALTH"  # Local: restore the label after a spectator leaves the overview.
        self.objective.text = self._objective_text(simulation, player)  # Local: surface the currently relevant objective.
        duration = simulation.config.plant_seconds if team == 0 else simulation.config.defuse_seconds  # Local: action duration.
        progress = min(1.0, float(simulation.interaction[0, player]) / duration)  # Local: represent held interaction time.
        self.interaction_fill.scale_x = max(0.001, progress * 0.26)  # Local: avoid a zero-scale graphics transform.
        self.interaction_bg.enabled = self.interaction_fill.enabled = progress > 0  # Local: hide idle progress bar.
        self.crosshair.enabled = followed is not None and not paused  # Local: crosshair is useful only in FPS view.
        self.notice.text = "PAUSED" if paused else notice  # Local: pause state takes precedence over victory messages.
        self.subnotice.text = "ESC TO RESUME  /  F10 TO QUIT" if paused else ("NEXT ROUND IN A MOMENT" if notice else "")
        if controlled is not None:  # Local: human ownership exposes both combat keys and the immediate hand-back action.
            self.hint.text = "WASD MOVE   MOUSE AIM   LMB FIRE   R RELOAD   HOLD E INTERACT   F HAND BACK   ESC PAUSE   F10 QUIT"
        elif followed is not None:  # Local: first-person bot watching exposes selection, camera choice, and takeover.
            self.hint.text = "TAB NEXT BOT   V FREE CAMERA   F TAKE OVER THIS BOT   ESC PAUSE   F12 SCREENSHOT   F10 QUIT"
        else:  # Local: free camera controls are cosmetic and never change a selected bot's movement.
            self.hint.text = "WASD FLY   Q/E DOWN/UP   RMB LOOK   TAB NEXT BOT   V FIRST PERSON   F TAKE OVER   ESC PAUSE   F10 QUIT"
        for index, dot in enumerate(self.dots):  # Local: update only the small retained marker objects.
            known = controlled is None or index // 5 == controlled // 5  # Local: a watch-mode takeover also loses enemy radar.
            dot.enabled = bool(simulation.alive[0, index]) and known  # Local: dead operators have no active marker.
            point = (simulation.positions[0, index, [0, 2]] - self.low) / self.span * 0.23 - 0.115  # Local: projection.
            dot.x, dot.y = float(point[0]), float(point[1])  # Local: copy coordinates, never write into simulation arrays.

    def _objective_text(self, simulation, player):
        """Local: choose a short objective instruction; global: teach planting and defusing during play."""
        if simulation.bomb_planted[0]:  # Local: after planting, both roles need an explicit bomb announcement.
            return "BOMB PLANTED  /  HOLD E NEAR BOMB TO DEFUSE" if player >= 5 else "BOMB PLANTED  /  DEFEND THE BOMB"
        if player >= 5:  # Local: defenders protect both sites before the plant.
            return "DEFEND A AND B  /  PREVENT THE PLANT"  # Local: direct role-specific objective.
        carrier = int(simulation.bomb_carrier[0])  # Local: identify possession using authoritative objective state.
        if carrier == player:  # Local: the human/observed attacker can plant while carrying the bomb.
            return "YOU CARRY THE BOMB  /  HOLD E AT A OR B TO PLANT"  # Local: make the interaction discoverable.
        return "BOMB DROPPED  /  HOLD E NEAR BOMB TO PICK UP" if carrier < 0 else "ESCORT THE CARRIER TO A OR B"  # Local: team task.
