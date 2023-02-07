import numpy as np
# import gym
# from gym.spaces import Discrete, MultiDiscrete
# from typing import Dict, List, Union
#
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.policy.rnn_sequencing import add_time_dimension
# from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.policy.view_requirement import ViewRequirement
# from ray.rllib.utils.torch_utils import one_hot
# from ray.rllib.utils.typing import ModelConfigDict, TensorType

from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
# from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class MyRNNModel(TorchRNN, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get configuration for this custom model
        '''
        # Config example: 
        model_config = {
            "custom_model_config": {
                "fc_sizes": [128, 64],
                "fc_activation": "relu",
                "lstm_state_size": 64,
                "post_fc_sizes": [],
                "post_fc_activation": "relu",
                "value_fc_sizes": [128, 64],
                "value_fc_activation": "relu",
                "value_lstm_state_size": 64,
                "value_post_fc_sizes": [],
                "value_post_fc_activation": "relu",
                "is_same_shape": False,
            }
        }
        '''
        if "is_same_shape" in model_config["custom_model_config"]:
            self.is_same_shape = model_config["custom_model_config"]["is_same_shape"]
        else:
            self.is_same_shape = False
            print("is_same_shape not received!!")
            print("is_same_shape == False")
        if "fc_sizes" in model_config["custom_model_config"]:
            self.fc_sizes = model_config["custom_model_config"]["fc_sizes"]
        else:
            self.fc_sizes = [256, 256]
            print(f"fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: fc_sizes = {self.fc_sizes}")
        if "fc_activation" in model_config["custom_model_config"]:
            self.fc_activation = model_config["custom_model_config"]["fc_activation"]
        else:
            self.fc_activation = "relu"
        if "lstm_state_size" in model_config["custom_model_config"]:
            self.lstm_state_size = model_config["custom_model_config"]["lstm_state_size"]
        else:
            self.lstm_state_size = 256
            print(f"lstm_state_size has NOT been received!\n So, lstm_state_size = {self.lstm_state_size}")
        if "post_fc_sizes" in model_config["custom_model_config"]:
            self.post_fc_sizes = model_config["custom_model_config"]["post_fc_sizes"]
        else:
            self.post_fc_sizes = [256]
            print(f"post_fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: post_fc_sizes = {self.post_fc_sizes}")
        if "post_fc_activation" in model_config["custom_model_config"]:
            self.post_fc_activation = model_config["custom_model_config"]["post_fc_activation"]
        else:
            self.post_fc_activation = "relu"
        if "value_fc_sizes" in model_config["custom_model_config"]:
            if self.is_same_shape:
                self.value_fc_sizes = self.fc_sizes.copy()
            else:
                self.value_fc_sizes = model_config["custom_model_config"]["value_fc_sizes"]
        else:
            self.value_fc_sizes = [256, 256]
            print(f"value_fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: value_fc_sizes = {self.value_fc_sizes}")
        if "value_fc_activation" in model_config["custom_model_config"]:
            self.value_fc_activation = model_config["custom_model_config"]["value_fc_activation"]
        else:
            self.value_fc_activation = "relu"
        if "value_lstm_state_size" in model_config["custom_model_config"]:
            if self.is_same_shape:
                self.value_lstm_state_size = self.lstm_state_size
            else:
                self.value_lstm_state_size = model_config["custom_model_config"]["value_lstm_state_size"]
        else:
            self.value_lstm_state_size = 256
            print(f"value_lstm_state_size has NOT been received!)")
            print(f"So, value_lstm_state_size = {self.value_lstm_state_size}")
        if "value_post_fc_sizes" in model_config["custom_model_config"]:
            if self.is_same_shape:
                self.value_post_fc_sizes = self.post_fc_sizes.copy()
            else:
                self.value_post_fc_sizes = model_config["custom_model_config"]["value_post_fc_sizes"]
        else:
            self.value_post_fc_sizes = [256]
            print(f"value_post_fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: value_post_fc_sizes = {self.value_post_fc_sizes}")
        if "value_post_fc_activation" in model_config["custom_model_config"]:
            self.value_post_fc_activation = model_config["custom_model_config"]["value_post_fc_activation"]
        else:
            self.value_post_fc_activation = "relu"

        # Define observation size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        # Initialize logits
        self._logits = None
        # Holds the current "base" output (before logits/value_out layer).
        self._features = None
        self._values = None

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        # Create layers and get fc_net
        for size in self.fc_sizes[:]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.fc_activation,
                )
            )
            prev_layer_size = size
        self.fc_net = nn.Sequential(*layers)

        # Create an lstm layer
        self.lstm = nn.LSTM(self.fc_sizes[-1], self.lstm_state_size, batch_first=True)

        # Create post FC layers
        post_layers = []
        prev_post_layer_size = self.lstm_state_size
        for size in self.post_fc_sizes[:]:
            post_layers.append(
                SlimFC(
                    in_size=prev_post_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.post_fc_activation,
                )
            )
            prev_post_layer_size = size
        self.post_fc_net = nn.Sequential(*post_layers)

        # Get VALUE fc layers
        # TODO: (Done-0) Get value layers (1)fc (2) lstm (3)post fc
        #       and connect to the output layers!!
        value_layers = []
        prev_value_layer_size = int(np.product(obs_space.shape))
        # Create layers and get fc_net
        for size in self.value_fc_sizes[:]:
            value_layers.append(
                SlimFC(
                    in_size=prev_value_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.value_fc_activation,
                )
            )
            prev_value_layer_size = size
        self.value_fc_net = nn.Sequential(*value_layers)

        # Create an lstm layer
        self.value_lstm = nn.LSTM(self.value_fc_sizes[-1], self.value_lstm_state_size, batch_first=True)

        # Create post FC layers
        value_post_layers = []
        prev_value_post_layer_size = self.value_lstm_state_size
        for size in self.value_post_fc_sizes[:]:
            value_post_layers.append(
                SlimFC(
                    in_size=prev_value_post_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.value_post_fc_activation,
                )
            )
            prev_value_post_layer_size = size
        self.value_post_fc_net = nn.Sequential(*value_post_layers)

        # Get last layers
        if self.post_fc_sizes:
            self.last_size = self.post_fc_sizes[-1]
        else:  # 만약 post_fc_sizes가 beer있는 list-ramen?
            self.last_size = self.lstm_state_size
        if self.value_post_fc_sizes:
            self.last_value_size = self.value_post_fc_sizes[-1]
        else:
            self.last_value_size = self.value_lstm_state_size
        # Policy network's last layer
        self.action_branch = nn.Linear(self.last_size, num_outputs)
        # Value network's last layer
        self.value_branch = nn.Linear(self.last_value_size, 1)

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc_net[-1]._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc_net[-1]._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.value_fc_net[-1]._model[0].weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
            self.value_fc_net[-1]._model[0].weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
        ]
        # print("INITIALIZATION carried out")
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._values), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # Get an output of the policy network
        x_fc_net = self.fc_net(inputs)  # Note: time-dimension has been added to the inputs
        x_lstm, [h1, c1] = self.lstm(
            x_fc_net, [torch.unsqueeze(state[0], 0),
                       torch.unsqueeze(state[1], 0)
                       ]
        )
        if self.post_fc_sizes:
            self._features = self.post_fc_net(x_lstm)
        else:
            self._features = x_lstm
        action_out = self.action_branch(self._features)
        # print(f'inputs = {inputs.shape}')
        # print(f'seq_len = {seq_lens}')
        # print(f'state[0] = {state[0].shape}')
        # print(f'x_fc_net = {x_fc_net.shape}')
        # print(f'x_lstm = {x_lstm.shape}')

        # Get an output of the value network
        x_value_fc_net = self.value_fc_net(inputs)
        x_value_lstm, [h2, c2] = self.value_lstm(
            x_value_fc_net, [torch.unsqueeze(state[2], 0),
                             torch.unsqueeze(state[3], 0)
                             ]
        )
        if self.value_post_fc_sizes:
            self._values = self.value_post_fc_net(x_value_lstm)
        else:
            self._values = x_value_lstm

        return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]


# class RayTorchRNNModelExample(TorchRNN, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         fc_size=64,
#         lstm_state_size=256,
#     ):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
#
#         self.obs_size = get_preprocessor(obs_space)(obs_space).size
#         self.fc_size = fc_size
#         self.lstm_state_size = lstm_state_size
#
#         # Build the Module from fc + LSTM + 2xfc (action + value outs).
#         self.fc1 = nn.Linear(self.obs_size, self.fc_size)
#         self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
#         self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
#         self.value_branch = nn.Linear(self.lstm_state_size, 1)
#         # Holds the current "base" output (before logits layer).
#         self._features = None
#
#     @override(ModelV2)
#     def get_initial_state(self):
#         # TODO: (sven): Get rid of `get_initial_state` once Trajectory
#         #  View API is supported across all of RLlib.
#         # Place hidden states on same device as model.
#         h = [
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#         ]
#         return h
#
#     @override(ModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         return torch.reshape(self.value_branch(self._features), [-1])
#
#     @override(TorchRNN)
#     def forward_rnn(self, inputs, state, seq_lens):
#         """Feeds `inputs` (B x T x ..) through the Gru Unit.
#         Returns the resulting outputs as a sequence (B x T x ...).
#         Values are stored in self._cur_value in simple (B) shape (where B
#         contains both the B and T dims!).
#         Returns:
#             NN Outputs (B x T x ...) as sequence.
#             The state batches as a List of two items (c- and h-states).
#         """
#         x = nn.functional.relu(self.fc1(inputs))
#         self._features, [h, c] = self.lstm(
#             x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
#         )
#         action_out = self.action_branch(self._features)
#         return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


# """
# Helper class to simplify implementing RNN models with TorchModelV2.
# Instead of implementing forward(), you can implement forward_rnn() which
# takes batches with the time dimension added already.
# Here is an example implementation for a subclass
# """
#
#
# class RayRNNExample2(RecurrentNetwork, nn.Module):
#     def __init__(self, obs_space, num_outputs):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs,
#                          model_config, name)
#         self.obs_size = _get_size(obs_space)
#         self.rnn_hidden_dim = model_config["lstm_cell_size"]
#         self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
#         self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)
#         self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
#         self._cur_value = None
#     @override(ModelV2)
#     def get_initial_state(self):
#         # Place hidden states on same device as model.
#         h = [self.fc1.weight.new(
#             1, self.rnn_hidden_dim).zero_().squeeze(0)]
#         return h
#     @override(ModelV2)
#     def value_function(self):
#         assert self._cur_value is not None, "must call forward() first"
#         return self._cur_value
#
#     @override(RecurrentNetwork)
#     def forward_rnn(self, input_dict, state, seq_lens):
#         x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
#         h_in = state[0].reshape(-1, self.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)
#         self._cur_value = self.value_branch(h).squeeze(1)
#         return q, [h]



# Example from https://discuss.ray.io/t/seperate-networks-for-actor-and-critic-in-the-ppo/5781
# This applies action masking at the forward method, so that it may differ a bit from the original styles.


# class RecurrentTorchModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#                  name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs,model_config, name)
#         nn.Module.__init__(self)
#
#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         if isinstance(seq_lens, np.ndarray):
#             seq_lens = torch.Tensor(seq_lens).int()
#
#         output, new_state = self.forward_rnn(
#             add_time_dimension(
#                 input_dict["obs"]["observation"].float(), seq_lens),
#             state, seq_lens)
#         action_mask = input_dict["obs"]["action_mask"]
#         inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
#         logits = torch.reshape(output, [-1, self.num_outputs])
#         masked_logits = logits + inf_mask
#         return masked_logits, new_state
#
#     def forward_rnn(self, inputs, state, seq_lens):
#         raise NotImplementedError("You must implement this for an RNN model")

# class TorchRNNModel(RecurrentTorchModel, nn.Module):
#     def __init__(self,
#                  obs_space,
#                  action_space,
#                  num_outputs,
#                  model_config,
#                  name,
#                  fc_size=40,
#                  lstm_state_size=80):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config,
#                          name)
#         self.obs_size = 16
#         self.fc_size = fc_size
#         self.lstm_state_size = lstm_state_size
#
#         # Build the Module from fc + LSTM
#             #actor net
#         self.actor_fc1 = nn.Linear(self.obs_size, self.fc_size)
#         self.actor_lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
#         # self.actor_layers = nn.Sequential(nn.Linear(self.obs_size, self.fc_size),
#         #                                   nn.ReLU(),
#         #                                   nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True))
#         self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
#             #value net
#         self.value_fc1 = nn.Linear(self.obs_size, self.fc_size)
#         self.value_lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
#         # self.value_layers = nn.Sequential(nn.Linear(self.obs_size, self.fc_size),
#         #                                   nn.ReLU(),
#         #                                   nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True))
#         self.value_branch = nn.Linear(self.lstm_state_size, 1)
#
#     @override(ModelV2)
#     def value_function(self):
#         assert self._values is not None, "must call forward() first"
#         return torch.reshape(self.value_branch(self._values), [-1])
#
#     @override(ModelV2)
#     def get_initial_state(self):
#         # make hidden states on same device as model
#         h = [self.actor_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.actor_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.value_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.value_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)]
#         return h
#
#     @override(RecurrentTorchModel)
#     def forward_rnn(self, inputs, state, seq_lens):
#         x1 = nn.functional.relu(self.actor_fc1(inputs))
#         self._features, [h1, c1] = self.actor_lstm(
#             x1, [torch.unsqueeze(state[0], 0),
#                 torch.unsqueeze(state[1], 0)])
#         action_out = self.action_branch(self._features)
#
#         x2 = nn.functional.relu(self.value_fc1(inputs))
#         self._values, [h2, c2] = self.value_lstm(
#             x2, [torch.unsqueeze(state[2], 0),
#                 torch.unsqueeze(state[3], 0)])
#
#         return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]
