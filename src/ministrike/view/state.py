"""Local: model camera selection and human takeover; global: input ownership never edits simulation state.

Dataclass reference: https://docs.python.org/3/library/dataclasses.html
"""

from dataclasses import dataclass  # Local: keep the small presentation state explicit and testable without graphics.

import numpy as np  # Local: select from the simulation's copied/read-only ten-player alive masks.


@dataclass
class ViewControlState:
    """Local: track the selected, followed, and human-controlled slots; global: keep those concepts independent."""

    initial_player: int | None  # Local: explicit play mode respawns this human slot; watch takeover is temporary.
    selected: int | None  # Local: Tab selects any living bot, including those on the other team.
    first_person: bool  # Local: V chooses player-eye viewing or the free spectator camera.
    controlled: int | None  # Local: only this slot has its policy output replaced by human controls.

    @classmethod
    def create(cls, mode, team=0, first_person=False, team_size=5):
        """Local: choose the initial interface state; global: keep the public play/watch modes backward compatible."""
        initial = int(team) * team_size if mode == "play" else None  # Local: active match defines permanent slots.
        return cls(initial, int(team) * team_size, mode == "play" or first_person, initial)  # Local: explicit modes.

    @property
    def followed(self):
        """Local: derive the first-person target; global: a living human always sees their own canonical camera."""
        return self.controlled if self.controlled is not None else (self.selected if self.first_person else None)

    def refresh(self, alive):
        """Local: resolve a death or invalid selection; global: never take control of or follow a dead actor."""
        living = np.flatnonzero(alive).tolist()  # Local: allocate an independent list without mutating simulation data.
        if self.controlled is not None and not alive[self.controlled]:  # Local: a controlled death ends human ownership.
            self.controlled = None  # Local: subsequent action batches restore policy ownership automatically.
            self.first_person = True  # Local: death continues in a living bot's first-person view where possible.
        if self.selected not in living:  # Local: a dead selection cannot remain a takeover target.
            self.selected = living[0] if living else None  # Local: use a stable next target or no target if all are dead.
        if not living:  # Local: a simultaneous all-dead outcome has no valid first-person viewpoint.
            self.first_person = False  # Local: show the final arena in the free camera instead.

    def cycle(self, alive):
        """Local: cycle across all living operators; global: only spectators may change the viewed actor."""
        if self.controlled is not None:  # Local: Tab cannot detach a living human's view from their operator.
            return  # Local: the human must first hand control back with F.
        living = np.flatnonzero(alive).tolist()  # Local: include both teams without stale dead slots.
        if living:  # Local: a terminal all-dead state leaves selection unset until the next round.
            index = living.index(self.selected) + 1 if self.selected in living else 0  # Local: cyclic next index.
            self.selected = living[index % len(living)]  # Local: wrap after the final living defender.

    def toggle_view(self):
        """Local: switch between free camera and bot eyes; global: human control never gains a detached camera."""
        if self.controlled is None and self.selected is not None:  # Local: a valid bot selection is required for POV.
            self.first_person = not self.first_person  # Local: this changes presentation only.

    def toggle_control(self, alive):
        """Local: hand control to/from the selected bot; global: preserve position, aim, health, ammo and bomb state."""
        if self.controlled is not None:  # Local: a second F press returns the exact same live operator to the policy.
            self.controlled = None  # Local: no simulation function is called for input ownership changes.
            self.first_person = True  # Local: continue watching that bot after handing back control.
        elif self.selected is not None and alive[self.selected]:  # Local: dead/no-target selections cannot be controlled.
            self.controlled = self.selected  # Local: the app changes only the action source for this one slot.
            self.first_person = True  # Local: taking over immediately uses the selected bot's own eye-level view.

    def new_round(self, alive):
        """Local: restore the session's chosen ownership policy; global: explicit play respawns, watch stays spectator."""
        self.controlled = self.initial_player  # Local: transient watch-mode takeovers end at each round boundary.
        if self.initial_player is not None:  # Local: explicit play returns the user to their original team slot.
            self.selected, self.first_person = self.initial_player, True  # Local: restore the human's first-person camera.
        self.refresh(alive)  # Local: verify selection against the newly reset arena without modifying it.
