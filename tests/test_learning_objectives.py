"""Local: synthetic library loss/GAE checks; global: verify scientific training invariants.

Source: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

import numpy as np  # Local: serialized state fixtures; global: safe resume compatibility.
import torch  # Local: synthetic differentiable tensors; global: no real training in these tests.
from tensordict import TensorDict  # Local: explicit arena/time/player dimensions.
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ministrike.learning.checkpoint import PolicyCheckpoint, load_checkpoint, save_checkpoint
from ministrike.learning.config import TrainConfig
from ministrike.learning.models import ActorNetwork, CentralCritic, make_actor, make_critic
from ministrike.learning.objectives import make_objectives, masked_total_loss, prepare_advantages
from ministrike.sim.constants import ACTION_NVECS, OBS_DIM


def terminal_batch(rewards, done):
    """Local: make independent team trajectories; global: provide symbolic ground truth for library loss checks."""
    shape = rewards.shape[:-1]  # Local: rewards retain arena, time, and five-player axes.
    observation = torch.zeros(*shape, OBS_DIM)  # Local: identical inputs prevent feature-dependent confounding.
    agents = TensorDict({"observation": observation, "alive": torch.ones_like(rewards, dtype=torch.bool)}, shape)
    next_agents = TensorDict({"observation": observation.clone(), "reward": rewards}, shape)
    return TensorDict({"agents": agents, "next": TensorDict({"agents": next_agents,
                       "done": done, "terminated": done.clone()}, shape[:-1])}, shape[:-1])


def test_gae_matches_exact_returns_and_stops_at_each_round_boundary():
    """Local: check numeric signs, discounting, and axes; global: resets must not transfer rewards between rounds."""
    critic = CentralCritic()  # Local: use the same centralized network wrapper as real training.
    with torch.no_grad():  # Local: a zero baseline makes the GAE result exactly calculable.
        for parameter in critic.parameters():
            parameter.zero_()  # Local: all residuals now equal the explicit reward fixture.
    _, estimator = make_objectives(make_actor(), make_critic(critic), TrainConfig(gamma=.9, gae_lambda=.8))
    rewards = torch.zeros(2, 4, 5, 1)  # Local: arenas have different termination patterns.
    rewards[0, 1], rewards[0, 3], rewards[1, 3] = 1, -1, 1  # Local: include both success and failure.
    done = torch.zeros(2, 4, 1, dtype=torch.bool)  # Local: a complete round can end inside a rollout.
    done[0, 1], done[:, 3] = True, True  # Local: the first arena has two independent two-step rounds.
    batch = terminal_batch(rewards, done)  # Local: production-compatible nested TensorDict axes.
    prepare_advantages(batch, estimator)  # Local: use production GAE and normalization in their real order.
    expected = torch.tensor([[.72, 1, -.72, -1], [.72 ** 3, .72 ** 2, .72, 1]])
    assert torch.allclose(batch["agents", "value_target"], expected[..., None, None].expand_as(rewards), atol=1e-6)


def test_ppo_gradient_increases_probability_of_rewarded_fire_action():
    """Local: verify the sign of an actual PPO gradient; global: positive outcomes must reinforce their actions."""
    network, critic = ActorNetwork(), CentralCritic()  # Local: use the production modules and categorical distribution.
    with torch.no_grad():  # Local: zero logits give a symmetric baseline with zero entropy gradient.
        for parameter in list(network.parameters()) + list(critic.parameters()):
            parameter.zero_()  # Local: any directional preference must arise from the signed rewards.
    actor = make_actor(network)  # Local: preserve the exact six-head probability contract.
    loss, estimator = make_objectives(actor, make_critic(critic), TrainConfig())
    rewards = torch.ones(2, 2, 5, 1)  # Local: every transition is an independent terminal team outcome.
    rewards[0] = -1  # Local: no-fire actions lose while otherwise identical fire actions win.
    batch = terminal_batch(rewards, torch.ones(2, 2, 1, dtype=torch.bool))
    actions = torch.zeros(2, 2, 5, 6, dtype=torch.long)  # Local: all non-fire controls are matched between teams.
    actions[1, ..., 3] = 1  # Local: the only discriminating action is the rifle trigger.
    with torch.no_grad():  # Local: behavior probabilities are fixed targets, never differentiable PPO denominators.
        distribution = actor.get_dist(batch)  # Local: compute the same initial action distribution used by the loss.
        batch["agents", "action"] = actions  # Local: explicit actions give an exact controlled intervention.
        batch["agents", "action_log_prob"] = distribution.log_prob(actions)  # Local: valid old-policy log probabilities.
    prepare_advantages(batch, estimator)  # Local: positive and negative outcomes pass through the production return pipeline.
    losses = loss(batch.reshape(-1))  # Local: flatten arena/time while retaining five-agent credit assignment.
    masked_total_loss(losses, batch["agents", "alive"].reshape(-1, 5, 1)).backward()
    fire_offset = sum(ACTION_NVECS[:3])  # Local: locate the fire head in the shared final layer without a magic index.
    gradient = network.network[-1].bias.grad  # Local: gradient descent subtracts this vector from action logits.
    assert gradient[fire_offset] > 0 and gradient[fire_offset + 1] < 0  # Global: rewarded fire becomes more probable.


def test_dead_slots_receive_delayed_rewards_and_gradients_are_finite():
    """Local: synthetic terminal reward; global: individual death cannot truncate team returns."""
    actor_network, critic_network = ActorNetwork(), CentralCritic()  # Test architecture, not a trained model.
    with torch.no_grad():  # Zero critic makes the expected reward propagation transparent.
        for parameter in critic_network.parameters():
            parameter.zero_()
    actor, critic = make_actor(actor_network), make_critic(critic_network)  # Use production library wrappers.
    loss, estimator = make_objectives(actor, critic, TrainConfig())  # Real library objective and GAE.
    observation = torch.zeros(2, 4, 5, OBS_DIM)  # Arena/time/player layout must survive until GAE.
    alive = torch.ones(2, 4, 5, 1, dtype=torch.bool)  # Alive before the first action.
    alive[:, 1:, 0] = False  # Player zero dies after the first transition.
    reward = torch.zeros(2, 4, 5, 1)  # No early reward is available to the dead player.
    reward[:, -1] = 1  # All retained slots receive the delayed terminal team outcome.
    done = torch.zeros(2, 4, 1, dtype=torch.bool)  # Arena-level terminal flags.
    done[:, -1] = True  # Terminate only at round completion.
    agents = TensorDict({"observation": observation, "alive": alive}, [2, 4, 5])
    next_agents = TensorDict({"observation": observation.clone(), "reward": reward}, [2, 4, 5])
    batch = TensorDict({"agents": agents, "next": TensorDict({"agents": next_agents,
                       "done": done, "terminated": done.clone()}, [2, 4])}, [2, 4])
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):  # Behaviors are fixed PPO targets.
        actor(batch)
    prepare_advantages(batch, estimator)  # Compute temporal recursion before applying any death mask.
    targets = batch["agents", "value_target"]  # Return targets remain unnormalized.
    assert torch.all(targets[:, 0, 0] > 0)  # A later victory still reaches the pre-death action.
    assert torch.allclose(targets[:, :, 0], targets[:, :, 1])  # Death did not truncate the return recursion.
    losses = loss(batch.reshape(-1))  # Shuffle boundary is arena/time only.
    total = masked_total_loss(losses, alive.reshape(8, 5, 1))  # Keep critic learning after death.
    total.backward()  # Synthetic backward verifies the graph without taking an optimizer step.
    assert torch.isfinite(total)  # Reject numerical errors before real training.
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in list(actor_network.parameters()) + list(critic_network.parameters()))


def test_death_mask_excludes_actor_and_entropy_but_not_critic():
    """Local: inspect gradients directly; global: dead-player actions cannot optimize the actor."""
    objective = torch.ones(2, 5, requires_grad=True)  # Independent synthetic player losses.
    entropy = torch.ones(2, 5, requires_grad=True)  # Entropy requires the same survival treatment.
    critic = torch.ones(2, 5, requires_grad=True)  # Critic must retain every player-time slot.
    alive = torch.ones(2, 5, 1, dtype=torch.bool)  # Before-step activity mask.
    alive[:, 0] = False  # Explicitly dead slots.
    total = masked_total_loss({"loss_objective": objective, "loss_entropy": entropy,
                              "loss_critic": critic}, alive)
    total.backward()  # Differentiate just the masked reduction.
    assert torch.all(objective.grad[:, 0] == 0) and torch.all(entropy.grad[:, 0] == 0)
    assert torch.all(critic.grad[:, 0] > 0)  # Delayed terminal supervision remains possible.


def test_safe_checkpoint_roundtrip_and_inference(tmp_path):
    """Local: safe model/NumPy serialization; global: downloadable actors work in gameplay."""
    actor = ActorNetwork()  # Untouched actor is a valid initial policy baseline.
    path = tmp_path / "model.pt"  # Pytest owns the disposable output directory.
    state = {"actor": actor.state_dict(), "observation_dim": OBS_DIM,
             "fixture": np.arange(4, dtype=np.int64), "rng": np.random.get_state()}
    save_checkpoint(path, state)  # Exercise the production atomic serializer.
    restored = load_checkpoint(path)  # Must work with weights_only=True internally.
    assert np.array_equal(restored["fixture"], state["fixture"])  # NumPy arrays retain values/type.
    policy = PolicyCheckpoint(path)  # Load exactly as the renderer/evaluator does.
    controls = policy.act(np.zeros((10, OBS_DIM), dtype=np.float32))  # Arbitrary leading player axes work.
    assert controls.shape == (10, 7) and np.isfinite(controls).all()  # Legal shared physical control shape.
