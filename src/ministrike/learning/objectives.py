"""Local: configure library PPO/GAE; global: MAPPO without handwritten RL loss formulas.

Source: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

import torch  # Local: masking/reduction; global: retain team returns after individual death.
from torchrl.objectives import ClipPPOLoss  # Local: maintained clipped surrogate/value/entropy implementation.
from torchrl.objectives.value import GAE  # Local: maintained temporally ordered advantage estimation.


def make_objectives(actor, critic, config):
    """Local: bind multi-agent names; global: use a shared actor and centralized critic."""
    loss = ClipPPOLoss(actor_network=actor, critic_network=critic,
                       clip_epsilon=config.clip_epsilon, entropy_coeff=config.entropy_coefficient,
                       critic_coeff=config.critic_coefficient, loss_critic_type="smooth_l1",
                       normalize_advantage=False, reduction="none", functional=False)
    keys = {"reward": ("agents", "reward"), "value": ("agents", "state_value"),
            "done": ("agents", "done"), "terminated": ("agents", "terminated"),
            "advantage": ("agents", "advantage"), "value_target": ("agents", "value_target")}
    loss.set_keys(**keys, action=("agents", "action"), sample_log_prob=("agents", "action_log_prob"))
    advantage = GAE(gamma=config.gamma, lmbda=config.gae_lambda, value_network=critic,
                    average_gae=False, time_dim=1, vectorized=True)
    advantage.set_keys(**keys)  # Time is explicit: [arena, time, player, scalar].
    return loss, advantage  # Callers use library losses then apply only survival reduction.


def prepare_advantages(batch, estimator):
    """Local: broadcast round termination and run GAE; global: death never truncates team credit."""
    reward = batch["next", "agents", "reward"]  # Team reward already repeats across player slots.
    for key in ("done", "terminated"):  # GAE needs player-shaped flags but round-based semantics.
        batch["next", "agents", key] = batch["next", key].unsqueeze(-2).expand_as(reward)
    with torch.no_grad():  # Returns are fixed targets for all PPO epochs on this batch.
        estimator(batch)  # Crucially, temporal recursion runs before actor death masking.
    value = batch["agents", "advantage"]  # Preserve all dead-slot advantages for diagnostics.
    alive = batch["agents", "alive"]  # This is survival before each action, never after damage.
    selected = value[alive]  # Normalize only actions that will contribute to the actor loss.
    if selected.numel() > 1:  # A fully dead team can still receive critic-only learning.
        batch["agents", "advantage"] = (value - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6)
    return batch  # Critic targets were computed before normalization and remain untouched.


def masked_total_loss(losses, alive):
    """Local: mask actor/entropy only; global: terminal team outcomes still train all critic slots."""
    mask = alive.squeeze(-1).to(losses["loss_objective"].dtype)  # Match per-player library loss shape.
    actor = ((losses["loss_objective"] + losses["loss_entropy"]) * mask).sum() / mask.sum().clamp_min(1)
    return actor + losses["loss_critic"].mean()  # Central critic sees every retained time/player slot.
