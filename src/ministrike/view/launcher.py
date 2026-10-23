"""Local: offer watch or first-person play; global: make trained checkpoints immediately usable.

The standard-library GUI closes before Ursina starts, so there is one active native event loop.
Source: https://docs.python.org/3/library/tkinter.html
File selection: https://docs.python.org/3/library/dialog.html#module-tkinter.filedialog
"""

from pathlib import Path  # Local: validate the explicit checkpoint path before opening the game window.


class LauncherWindow:
    """Local: present arena, bot-POV, and human-play actions; global: preserve explicit bot provenance and team choice."""

    def __init__(self, root, checkpoint, seed):
        """Local: assemble a compact start menu; global: require no browser, server, or additional UI library."""
        import tkinter as tk  # Local: the standard-library GUI is imported only during an interactive launch.

        self.root, self.seed = root, seed  # Local: retain the one window and reproducible match seed.
        self.selection = None  # Local: closing the menu without an action exits normally.
        self.path = tk.StringVar(value=str(checkpoint or ""))  # Local: show exactly the supplied recommended checkpoint.
        self.team = tk.IntVar(value=0)  # Local: start human play as an attacker unless the user selects defenders.
        self.status = tk.StringVar()  # Local: update the visible provenance as the user edits/browses a checkpoint.
        self.root.title("MiniStrike | Choose how to play")  # Local: native titlebar identifies the same game as Ursina.
        self.root.configure(bg="#18252f")  # Local: match the navy/amber/teal arena palette.
        self.root.geometry("1060x530")  # Local: keep the arena, bot POV and human play choices equally visible.
        self.root.resizable(False, False)  # Local: this small fixed layout remains clear without responsive complexity.
        self.root.protocol("WM_DELETE_WINDOW", self.close)  # Local: native close cancels instead of launching a mode.
        body = tk.Frame(root, bg="#18252f", padx=32, pady=25)  # Local: consistent outer spacing frames all three choices.
        body.pack(fill="both", expand=True)  # Local: occupy the complete fixed-size native content area.
        tk.Label(body, text="MINISTRIKE", bg="#18252f", fg="#e1ebee", font=("Segoe UI", 27, "bold")).pack(anchor="w")
        tk.Label(body, text="LOCAL SELF-PLAY LAB  /  CHECKPOINT SELECTS ARENA AND RULES", bg="#18252f", fg="#9badb9",
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 20))  # Local: communicate the runnable game's scope.
        tk.Label(body, text="BOT POLICY", bg="#18252f", fg="#9badb9", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        row = tk.Frame(body, bg="#18252f")  # Local: keep the path entry and native file picker on one line.
        row.pack(fill="x", pady=(5, 6))  # Local: align checkpoint controls with the width of the action cards.
        tk.Entry(row, textvariable=self.path, bg="#101a23", fg="#e1ebee", insertbackground="#e1ebee",
                 relief="flat", font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True, ipady=8)  # Local: explicit path.
        tk.Button(row, text="Browse...", command=self.browse, bg="#344957", fg="#e1ebee", activebackground="#456070",
                  activeforeground="#ffffff", relief="flat", padx=14, pady=6).pack(side="left", padx=(9, 0))  # Local: file picker.
        tk.Label(body, textvariable=self.status, bg="#18252f", fg="#f29e4f", font=("Segoe UI", 9)).pack(anchor="w")
        cards = tk.Frame(body, bg="#18252f")  # Local: equal-width side-by-side actions prevent a hidden preferred mode.
        cards.pack(fill="x", pady=(21, 15))  # Local: place the requested viewing and playing options at the visual center.
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="action")  # Local: three explicit modes have equal width.
        self.watch_button = self._card(  # Local: expose the button reference for a small native smoke check.
            cards, 0, "Watch arena", "Fly over the complete match.\nV: bot POV / Tab: next bot / F: take over.",
            "watch", "#4ccabe", "#65dacd", tk,
        )  # Local: watch mode provides a direct way to inspect trained behavior.
        self.watch_first_button = self._card(  # Local: make watching a bot's actual eyes a direct menu choice.
            cards, 1, "Watch bot first-person", "See exactly where a bot is looking.\nTab: next bot / F: take over immediately.",
            "watch", "#4ccabe", "#65dacd", tk, first_person=True,
        )  # Local: this spectator option starts without human input ownership.
        self.play_button = self._card(  # Local: the final action replaces one bot with the human player from spawn.
            cards, 2, "Play first-person", "Take one operator's place in the match.\nMove, aim, shoot and reload.",
            "play", "#f29e4f", "#ffb66f", tk,
        )  # Local: native FPS input starts only after the launcher is closed.
        options = tk.Frame(body, bg="#18252f")  # Local: team choice matters only when selecting human play.
        options.pack(anchor="w", pady=(0, 14))  # Local: place the optional team choice directly below the action cards.
        tk.Label(options, text="YOUR TEAM", bg="#18252f", fg="#9badb9", font=("Segoe UI", 9, "bold")).pack(side="left")
        for value, text, tint in ((0, "Team 0", "#f29e4f"), (1, "Team 1", "#4ccabe")):
            tk.Radiobutton(options, text=text, variable=self.team, value=value, bg="#18252f", fg=tint,
                           selectcolor="#101a23", activebackground="#18252f", activeforeground=tint,
                           font=("Segoe UI", 10)).pack(side="left", padx=(15, 0))  # Local: explicit canonical team IDs.
        tk.Label(body, text="WASD move   Mouse aim   Left click fire   R reload   Hold E interact\n"
                 "F takes over a watched bot or hands it back. Esc pauses. F10 closes the game.", bg="#18252f", fg="#9badb9",
                 font=("Segoe UI", 9), justify="left").pack(anchor="w")  # Local: show essential FPS controls before capture.
        self.path.trace_add("write", self._update_status)  # Local: a changed path immediately updates visible provenance.
        self._update_status()  # Local: make the initial trained/baseline status explicit before a button is pressed.

    def _card(self, parent, column, heading, detail, mode, tint, active, tk, first_person=False):
        """Local: reuse one action-card layout; global: avoid duplicating watch/play behavior and styling."""
        frame = tk.Frame(parent, bg="#101a23", padx=17, pady=17)  # Local: frame separates each distinct starting action.
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))  # Local: equal gutter.
        button = tk.Button(frame, text=heading, command=lambda: self.choose(mode, first_person), bg=tint, fg="#101a23",
                           activebackground=active, activeforeground="#101a23", relief="flat", cursor="hand2",
                           font=("Segoe UI", 13, "bold"), pady=12)  # Local: mode and initial camera are bound per card.
        button.pack(fill="x")  # Local: the full card width is an obvious click target.
        tk.Label(frame, text=detail, bg="#101a23", fg="#b3c5ce", justify="left", font=("Segoe UI", 9),
                 pady=13).pack(anchor="w")  # Local: clarify what each action does before the user commits to it.
        return button  # Local: a small smoke test can invoke the real native button without manual clicking.

    def _update_status(self, *_):
        """Local: describe the selected policy; global: never silently claim a scripted baseline has trained."""
        selected = self.path.get().strip()  # Local: surrounding whitespace must not turn an empty choice into a false path.
        self.status.set(f"Checkpoint selected: {Path(selected).name}" if selected else
                        "Scripted baseline selected. Choose a trained checkpoint when one is available.")  # Local: provenance.

    def browse(self):
        """Local: select an existing local checkpoint; global: keep model choice explicit and inspectable."""
        from tkinter import filedialog  # Local: reuse the operating system's normal file-selection experience.

        choice = filedialog.askopenfilename(parent=self.root, title="Choose a MiniStrike policy checkpoint",
                                          filetypes=(("Policy checkpoints", "*.pt *.pth"), ("All files", "*.*")))
        if choice:  # Local: cancelling the picker preserves the previously visible selection.
            self.path.set(choice)  # Local: the trace callback updates the trained-checkpoint label automatically.

    def choose(self, mode, first_person=False):
        """Local: validate the chosen path and record one mode; global: close Tk before starting Ursina."""
        from tkinter import messagebox  # Local: invalid paths deserve a readable correction instead of silent fallback.

        checkpoint = self.path.get().strip() or None  # Local: an empty path explicitly selects the scripted baseline.
        if checkpoint is not None and not Path(checkpoint).is_file():  # Local: verify the selected artifact exists.
            messagebox.showerror("Checkpoint not found", "Choose an existing checkpoint, or clear the path to use the scripted baseline.",
                                 parent=self.root)  # Local: keep the menu open so the user can correct their selection.
            return  # Local: do not start a game with an unintended policy.
        self.selection = {"mode": mode, "checkpoint": checkpoint, "seed": self.seed, "team": self.team.get(),
                          "spectator_first_person": first_person}  # Local: distinguish watching bot eyes from human play.
        self.root.destroy()  # Local: end the native menu event loop before initializing the rendering engine.

    def close(self):
        """Local: cancel without selecting a mode; global: closing the menu must not capture the mouse or start a match."""
        self.root.destroy()  # Local: selection remains None, making normal cancellation explicit.


def run_launcher(checkpoint=None, seed=0):
    """Local: launch selection, then the chosen game mode; global: expose arena watching, bot POV, and human play."""
    import tkinter as tk  # Local: defer the only native GUI dependency until the CLI requests the launcher.
    from ministrike.view import run_view  # Local: import the lazy public wrapper, preserving headless module imports.

    root = tk.Tk()  # Local: create one small native menu window.
    chooser = LauncherWindow(root, checkpoint, seed)  # Local: present recommended weights without hidden model discovery.
    root.mainloop()  # Local: wait for an explicit watch/play choice or normal menu cancellation.
    return run_view(**chooser.selection) if chooser.selection is not None else None  # Local: start exactly one requested mode.
