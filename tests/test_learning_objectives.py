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
from ministrike.sim.constants import OBS_DIM


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
