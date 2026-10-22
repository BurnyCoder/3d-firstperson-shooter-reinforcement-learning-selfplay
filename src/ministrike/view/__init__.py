"""Local: expose the viewer lazily; global: training never imports a graphics engine."""


def run_view(*args, **kwargs):
    """Local: load the app only when requested; global: keep headless workflows display independent."""
    from ministrike.view.app import run_view as launch  # Local: defer Ursina and GPU window imports.
    return launch(*args, **kwargs)  # Local: forward the public viewer options unchanged.


def run_launcher(*args, **kwargs):
    """Local: load the watch/play start menu lazily; global: keep training free of native GUI initialization."""
    from ministrike.view.launcher import run_launcher as launch  # Local: defer tkinter until a user opens the game.
    return launch(*args, **kwargs)  # Local: preserve checkpoint selection and seed from the CLI.
