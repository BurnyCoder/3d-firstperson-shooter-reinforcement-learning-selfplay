"""Local: test takeover transitions without a window; global: switching input must preserve the running match."""

import numpy as np  # Local: use the simulator's ten-slot alive-mask convention.
from types import SimpleNamespace  # Local: isolate native-shortcut dispatch from graphics.

from ministrike.view.state import ViewControlState  # Local: import only presentation-control state, never graphics.
from ministrike.view.app import ViewSession  # Local: this module defers graphics imports until a window is opened.


def test_takeover_and_hand_back_preserve_selected_bot():
    """Local: swap input ownership twice; global: no round reset or player-state mutation is involved."""
    alive = np.ones(10, dtype=bool)  # Local: all ten slots are eligible for selection in this scenario.
    state = ViewControlState.create("watch", team=0, first_person=True)  # Local: start watching one bot's eyes.
    state.cycle(alive)  # Local: select a different operator before taking control.
    assert state.selected == 1 and state.followed == 1  # Local: first-person camera follows the selection.
    state.toggle_control(alive)  # Local: the first F press changes only which slot receives human actions.
    assert state.controlled == 1 and state.followed == 1  # Local: camera and chosen identity remain stable.
    state.toggle_control(alive)  # Local: the second F press restores learned/scripted policy control.
    assert state.controlled is None and state.followed == 1  # Local: remain watching the same live bot.
    np.testing.assert_array_equal(alive, np.ones(10, dtype=bool))  # Local: transitions do not modify simulation-owned masks.


def test_dead_takeover_returns_to_spectating_and_cannot_control_dead_slot():
    """Local: resolve a death while controlling a bot; global: dead actors cannot accept takeover commands."""
    alive = np.ones(10, dtype=bool)  # Local: begin with a valid live selection.
    state = ViewControlState.create("watch", team=0, first_person=False)  # Local: a free-camera watcher can take over too.
    state.toggle_control(alive)  # Local: control the currently selected slot zero.
    alive[0] = False  # Local: emulate the simulator killing that controlled player.
    state.refresh(alive)  # Local: update presentation after the authoritative transition.
    assert state.controlled is None and state.followed == 1  # Local: automatically watch the next living operator.
    alive[:] = False  # Local: exercise the terminal all-dead edge case.
    state.refresh(alive)  # Local: no valid first-person follow target remains.
    state.toggle_control(alive)  # Local: F must be harmless when no living target can be selected.
    assert state.controlled is None and state.followed is None  # Local: remain safely in the free camera.


def test_all_teams_cycle_and_round_respawn_policy_is_explicit():
    """Local: cycle all teams and reset only viewer ownership; global: initial play and transient takeover differ."""
    alive = np.ones(10, dtype=bool)  # Local: use all ten permanent player slots.
    watcher = ViewControlState.create("watch", team=0, first_person=True)  # Local: no permanent human slot exists.
    for _ in range(5):  # Local: cycling past attacker five must reach defenders.
        watcher.cycle(alive)  # Local: choose the next living operator regardless of team.
    assert watcher.selected == 5  # Local: spectator access includes the defending team.
    watcher.toggle_control(alive)  # Local: temporarily control defender one.
    watcher.new_round(alive)  # Local: a transient watch takeover does not silently become permanent play.
    assert watcher.controlled is None  # Local: the next watch round resumes bot control.
    player = ViewControlState.create("play", team=1, first_person=False)  # Local: explicit play reserves defender one.
    player.toggle_control(alive)  # Local: the human may hand the slot back during the current round.
    player.new_round(alive)  # Local: an explicit play session restores the initial slot on the next round.
    assert player.controlled == 5 and player.followed == 5  # Local: human respawn ownership is predictable.


def test_camera_toggle_cannot_reveal_hidden_opponents_while_human():
    """Local: prevent a human free-camera toggle; global: watching privileges must not leak into live human play."""
    state = ViewControlState.create("play", team=0, first_person=False)  # Local: begin controlling the attacker slot.
    state.toggle_view()  # Local: V is deliberately ignored while the human owns a live operator.
    assert state.first_person and state.followed == 0  # Local: camera remains at the actual player's eyes.


def test_function_key_aliases_use_the_same_live_ownership_transitions():
    """Local: support alternate native keys; global: aliases cannot reset or otherwise mutate the match."""
    alive = np.ones((1, 2), dtype=bool)  # Local: the smallest supported combat match.
    state = ViewControlState.create("watch", team_size=1)
    session = SimpleNamespace(control_state=state, simulation=SimpleNamespace(alive=alive),
                              controls=SimpleNamespace(clear=lambda: None), _sync_mouse=lambda: None)
    ViewSession.input(session, "f1")  # Local: the alternate camera key uses the normal V branch.
    assert state.followed == 0 and state.controlled is None
    ViewSession.input(session, "f2")  # Local: the alternate ownership key uses the normal F branch.
    assert state.controlled == 0
    ViewSession.input(session, "f2")  # Local: returning ownership preserves the same selection.
    assert state.controlled is None and state.followed == 0
    np.testing.assert_array_equal(alive, np.ones((1, 2), dtype=bool))
