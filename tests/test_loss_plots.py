"""Local: validate plotted evidence; global: malformed logs never create invented or duplicated curves."""

import numpy as np  # Local: missing numeric observations must remain NaN gaps in Matplotlib.
import pytest  # Local: explicit failure assertions for ambiguous canonical update order.

from ministrike.experiments.loss_plots import metric_series, parse_metrics  # Local: test data integrity, not visual styling.


def test_loss_data_preserves_missingness_and_partial_tail():
    """Local: distinguish zero, missing and interrupted writes; global: plots must not imply fabricated improvement."""
    raw = b'{"frames":8,"loss":0}\n{"frames":16,"loss":NaN}\n{"frames":24}\n{"frames":32'
    rows, warnings = parse_metrics(raw)
    values = metric_series(rows, "loss")
    assert values[0] == 0 and np.isnan(values[1:]).all()
    assert len(rows) == 3 and len(warnings) == 1  # Local: complete records survive an unfinished final write.


def test_loss_data_refuses_duplicate_frames_or_broken_interior():
    """Local: reject ambiguous recovery records; global: neither concatenation nor sorting hides repeated updates."""
    with pytest.raises(ValueError, match="increasing"):
        parse_metrics(b'{"frames":8}\n{"frames":8}\n')
    with pytest.raises(ValueError, match="line 2"):
        parse_metrics(b'{"frames":8}\nbroken\n{"frames":16}\n')
