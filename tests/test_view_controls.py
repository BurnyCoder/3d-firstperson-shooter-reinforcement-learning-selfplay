"""Local: exercise input accumulation; global: keep FPS input independent of frame rate.

Source: https://www.ursinaengine.org/api_reference.html#mouse
"""

import numpy as np  # Local: compare numeric controls; global: use the simulation's array interface.

from ministrike.view.controls import HumanControls  # Local: import no graphics; global: permit headless tests.


def test_mouse_delta_is_consumed_once_and_keys_remain_held():
    """Local: consume accumulated mouse movement; global: prevent duplicate turn input on catch-up ticks."""
    controls = HumanControls(sensitivity=90.0)  # Local: one normalized mouse unit is 90 degrees.
    controls.add_mouse(0.1, -0.2)  # Local: collect movement from the first rendered frame.
    controls.add_mouse(0.2, 0.1)  # Local: collect movement before the next simulation tick.
    first = controls.consume({"w": 1, "d": 1, "left mouse": 1, "e": 1})  # Local: read held actions.
    second = controls.consume({"w": 1})  # Local: consume again without another mouse frame.
    np.testing.assert_allclose(first[2:4], np.deg2rad([27.0, -9.0]))  # Local: verify additive radians.
    np.testing.assert_array_equal(first[[0, 1, 4, 5, 6]], [1, 1, 1, 0, 1])  # Local: verify action order.
    np.testing.assert_array_equal(second, [1, 0, 0, 0, 0, 0, 0])  # Local: keys persist, deltas do not.


def test_clear_prevents_turn_after_unpause():
    """Local: discard stale input; global: resuming must not rotate the human unexpectedly."""
    controls = HumanControls()  # Local: use the normal sensitivity for this lifecycle check.
    controls.add_mouse(0.5, 0.5)  # Local: simulate accumulated motion before pausing.
    controls.clear()  # Local: pause discards those deltas rather than applying them later.
    np.testing.assert_array_equal(controls.consume({}), np.zeros(7))  # Local: idle input remains neutral.
