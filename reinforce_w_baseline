"""
REINFORCE with baseline implementation
Policy network and value network separated; but it uses total loss, which may cause unstable learning 
TODO(1): loss function and compute gradients separately
TODO(2): use get_default_config method (and create algo from config object)
TODO(3): replace build_torch_policy with build_policy_class
TODO(4): gradient clipping
TODO(5): value loss clipping
"""
import ray
from ray import tune
# from ray.rllib.agents import Trainer
# from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
# from ray.rllib.evaluation import PolicyEvaluator
from ray.rllib.policy import Policy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
# from ray.rllib.models.torch.torch_policy_template import build_torch_policy
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.tune.logger import pretty_print


torch, nn = try_import_torch()


# Build models: policy and value networks
# class ReinforceModelOutdated(nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(ReinforceModelOutdated, self).__init__()
#
#         self.fc1 = nn.Linear(obs_space.shape[0], 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, num_outputs)
#
#         self.value1 = nn.Linear(obs_space.shape[0], 64)
#         self.value2 = nn.Linear(64, 64)
#         self.value3 = nn.Linear(64, 1)
#
#         self._value_out = None
#
#     def forward(self, input):
#         x = input.float()
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         logits = self.fc3(x)
#
#         value = torch.relu(self.value1(x))
#         value = torch.relu(self.value2(value))
#         value = self.value3(value)
#
#         self._value_out = value
#         return logits, []
#
#     def value_function(self):
#         return self._value_out.squeeze(1)


class ReinforceModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = obs_space.shape[0]
        self.hidden_size = 64

        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, input_dict: dict, state: list, seq_lens: TensorType) -> (TensorType, list):
        obs = input_dict["obs"].float()
        logits = self.policy_net(obs)
        self._value_out_from_forward = self.value_net(obs)
        return logits, state

    def value_function(self) -> TensorType:
        # return self.value_net(self._last_input_dict["obs"].float()).squeeze(-1)
        return self._value_out_from_forward.squeeze(-1)


# Register the model
ModelCatalog.register_custom_model("reinforce_model", ReinforceModel)


# Define loss function to build policy
def reinforce_loss_outdated(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    log_probs = action_dist.logp(train_batch["actions"])
    values = model.value_function()

    advantages = train_batch["rewards"] + train_batch["dones"].float() * policy.config["gamma"] * values - values
    policy_loss = -torch.mean(log_probs * advantages.detach())

    value_loss = 0.5 * torch.mean(torch.pow(advantages.detach(), 2))
    return policy_loss + value_loss


# def reinforce_loss(policy, model, dist_class, train_batch):
#     logits, _ = model(train_batch)
#     actions = train_batch[SampleBatch.ACTIONS]
#     rewards = train_batch[SampleBatch.REWARDS]
#     dones = train_batch[SampleBatch.DONES]
#     values = model.value_function()
#
#     # Compute returns
#     returns = []
#     current_return = 0
#     for step in reversed(range(len(rewards))):
#         current_return = rewards[step] + policy.config["gamma"] * current_return * (1 - dones[step])
#         returns.insert(0, current_return)
#
#     # returns = torch.tensor(returns, dtype=torch.float32).to(model.device)
#     returns = torch.tensor(returns, dtype=torch.float32)
#     advantages = returns - values.detach()
#
#     # Compute action probabilities
#     dist = dist_class(logits, model)
#     log_probs = dist.logp(actions)
#
#     # Calculate policy loss
#     policy_loss = -torch.mean(log_probs * advantages)
#
#     # Calculate value loss
#     value_loss = 0.5 * torch.mean(torch.pow(values - returns, 2))
#
#     # Combine the losses
#     loss = policy_loss + value_loss
#     return loss


def reinforce_loss(policy, model, dist_class, train_batch):
    logits, _ = model(train_batch)
    values = model.value_function()
    actions = train_batch[SampleBatch.ACTIONS]
    rewards = train_batch[SampleBatch.REWARDS]
    dones = train_batch[SampleBatch.DONES]

    # Compute advantages using rewards and values
    advantages = torch.zeros_like(rewards)
    current_return = torch.zeros_like(rewards[-1])
    for step in reversed(range(len(rewards))):
        # current_return = rewards[step] + policy.config["gamma"] * current_return * (1 - dones[step])
        current_return = rewards[step] + policy.config["gamma"] * current_return * (1 - dones[step].to(torch.float32))
        advantages[step] = current_return - values[step]

    # Policy loss
    # action_distribution = policy.dist_class(logits, model)
    # log_probs = action_distribution.log_prob(actions)
    # policy_loss = -torch.mean(log_probs * advantages.detach())

    dist = dist_class(logits, model)
    log_probs = dist.logp(actions)
    policy_loss = -torch.mean(log_probs * advantages.detach())

    # Value loss
    value_loss = 0.5 * torch.mean((values.squeeze() - advantages.detach()) ** 2)

    # print(f"config = {pretty_print(policy.config)}")

    # return policy_loss + policy.config["vf_coeff"] * value_loss
    vf_coeff = 0.2  # Make it a config of the policy
    return policy_loss + vf_coeff * value_loss  # 


# Build the policy (subclass the policy using an existing policy class if possible cuz it's been deprecated.)
ReinforceTorchPolicy = build_torch_policy(
    name="ReinforceTorchPolicy",
    loss_fn=reinforce_loss,
    get_default_config=lambda: {"model": {"custom_model": "reinforce_model"}},
)

# config = AlgorithmConfig()
# config.training(gamma=0.99,
#                 lr=1e-3,
#                 train_batch_size=1000,
#                 num_sgd_iter=
#                 )
# config.environment(env="CartPole-v0")
# config.framework(framework="torch")
# config.resources(num_gpus=1)
# config.rollouts(num_rollout_workers=1, batch_mode="complete_episodes")


# Create a new Algorithm using the Policy defined above.
class ReinforceBaseline(Algorithm):
    @classmethod
    def get_default_policy_class(cls, config):
        return ReinforceTorchPolicy
    # TODO: get the method 'get_default_config()' overridden to add custom configurations

# ReinforceTrainer = build_trainer(
#     name="REINFORCE",
#     default_policy=ReinforceTorchPolicy,
#     default_config={
#         "framework": "torch",  # Ensure we use PyTorch
#         "env": "CartPole-v0",  # Define the environment
#         "num_workers": 1,  # Number of parallel workers for sampling
#         "lr": 1e-3,  # Learning rate
#         "train_batch_size": 1000,  # Size of the training batch
#         "batch_mode": "complete_episodes",  # Rollouts are processed in whole episodes
#         "num_sgd_iter": 1,  # Number of SGD iterations per training batch
#         "gamma": 0.99,  # Discount factor
#     },
# )


if __name__ == "__main__":
    ray.init()

    tune.run(ReinforceBaseline,
             config={"framework": "torch",  # Ensure we use PyTorch
                     "env": "CartPole-v0",  # Define the environment
                     "num_workers": 0,  # Number of parallel workers for sampling
                     "lr": 1e-3,  # Learning rate
                     "train_batch_size": 1000,  # Size of the training batch
                     # "batch_mode": "complete_episodes",  # Rollouts are processed in whole episodes
                     # "num_sgd_iter": 1,  # Number of SGD iterations per training batch
                     "gamma": 0.99,  # Discount factor
                     # "vf_coeff": 0.1,
                     # "vf_share_layers": False,
                     },
             )

