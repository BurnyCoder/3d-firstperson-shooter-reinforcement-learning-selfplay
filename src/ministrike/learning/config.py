"""Local: immutable run parameters; global: checkpoint the complete training protocol.

Numerical defaults are retained engineering decisions in docs/duel-protocol.md.
Algorithm API: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

from dataclasses import dataclass  # Local: explicit typed configuration; global: portable run metadata.


@dataclass
class TrainConfig:
    """Local: CLI/orchestrator input; global: arena frames count once regardless of player count."""

    output_dir: str = "runs/smoke"  # Local: artifacts stay in the current project.
    condition: str = "selfplay"  # Local: historical self-play or fixed scripted opposition.
    seed: int = 101  # Local: independent model, action, arena, and matchup seed root.
    num_envs: int = 16  # Local: independent arenas stepped in one NumPy batch.
    total_frames: int = 65536  # Local: arena transitions, not player action count.
    max_seconds: float = 600.0  # Local: includes collection, optimization, and checkpoint overhead.
    device: str = "cpu"  # Local: optimizer device; inference_device selects rollout placement and opponents use CPU.
    resume: str | None = None  # Local: optional complete training-state checkpoint.
    rollout_steps: int = 128  # Local: keep each arena's temporal sequence intact for GAE.
    epochs: int = 3  # Local: bounded PPO reuse per collected batch.
    minibatch_size: int = 256  # Local: arena-time rows, each containing all five players.
    learning_rate: float = 3e-4  # Local: fixed Adam learning rate across conditions.
    gamma: float = 0.999  # Local: 0.999**1200 ~= 0.30 retains sixty-second outcomes at twenty Hertz.
    gae_lambda: float = 0.95  # Local: bias/variance tradeoff for library GAE.
    clip_epsilon: float = 0.2  # Local: symmetric PPO importance-ratio clipping.
    entropy_coefficient: float = 0.01  # Local: encourage exploration in six categorical controls.
    critic_coefficient: float = 0.5  # Local: weight the central value regression.
    max_grad_norm: float = 1.0  # Local: limit unstable joint actor/critic updates.
    snapshot_interval: int = 10  # Local: completed updates between immutable opponent snapshots.
    milestone_updates: tuple[int, ...] = ()  # Local: registered immutable evaluation endpoints; never change opponent snapshot cadence.
    log_interval: int = 1  # Local: report every update by default with timestamps supplied by root logger.
    torch_threads: int = 4  # Local: avoid excessive CPU overhead for small dense networks.
    source_revision: str | None = None  # Local: persist the precise Git commit that started the run.
    experiment: str = "legacy-5v5-v1"  # Local: select a versioned match without changing legacy checkpoint defaults.
    reward: str = "R0"  # Local: registered duel reward; legacy shaping remains unchanged.
    inference_device: str = "cpu"  # Local: calibration chooses placement using complete throughput.
    observation_transform: str = "identity"  # Local: actor preprocessing is registered and checkpointed; critic inputs stay unchanged.
    registration: str | None = None  # Local: stable roadmap experiment ID required for Stage A optimizer work.
    run_config: str | None = None  # Local: published JSON freezes this execution's resolved parameters.
    budget_manifest: str = "runs/duel-fixed-v1/manifest.json"  # Local: a single absolute Stage A clock.

    def validate(self):
        """Local: reject impossible budgets; global: scientific protocol cannot silently drift."""
        if self.condition not in {"selfplay", "fixed"}:  # Both conditions share all other machinery.
            raise ValueError("condition must be selfplay or fixed")
        from ministrike.learning.observation_transform import TRANSFORMS  # Local: share the actual supported representation identities.
        if self.observation_transform not in TRANSFORMS:
            raise ValueError("Unknown observation transform")
        counts = (self.num_envs, self.total_frames, self.rollout_steps, self.epochs,
                  self.minibatch_size, self.snapshot_interval, self.log_interval, self.torch_threads)
        if any(value <= 0 for value in counts) or self.max_seconds <= 0:  # Prevent empty/infinite runs.
            raise ValueError("Training counts and wall-clock budget must be positive")
        if self.total_frames % self.num_envs:  # Never silently collect more than the frozen frame budget.
            raise ValueError("total_frames must be divisible by num_envs")
        batches = (self.total_frames + self.num_envs * self.rollout_steps - 1) // (self.num_envs * self.rollout_steps)
        if (any(type(value) is not int or not 1 <= value <= batches for value in self.milestone_updates)
                or list(self.milestone_updates) != sorted(set(self.milestone_updates))):
            raise ValueError("milestone updates must be unique increasing reachable positive integers")
        if self.experiment == "duel-fixed-v1" and self.condition != "selfplay":
            raise ValueError("Stage A trains exclusively from self-play")  # Local: evaluation scripts cannot train.
