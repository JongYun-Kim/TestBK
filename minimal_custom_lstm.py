import numpy as np
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class MinimalLSTM(TorchRNN, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_sizes=[128],
            fc_activation="relu",
            lstm_state_size=128,
            post_fc_sizes=[128],
            post_fc_activation="relu",
            value_fc_sizes=[128],
            value_fc_activation="relu",
            value_lstm_state_size=64,
            value_post_fc_sizes=[128],
            value_post_fc_activation="relu",
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get configuration for the custom model
        self.fc_sizes = fc_sizes.copy()
        self.fc_activation = fc_activation
        self.lstm_state_size = lstm_state_size
        self.post_fc_sizes = post_fc_sizes.copy()
        self.post_fc_activation = post_fc_activation
        self.value_fc_sizes = value_fc_sizes.copy()
        self.value_fc_activation = value_fc_activation
        self.value_lstm_state_size = value_lstm_state_size
        self.value_post_fc_sizes = value_post_fc_sizes.copy()
        self.value_post_fc_activation = value_post_fc_activation

        # Define observation size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        # Initialize logits
        # self._logits = None
        # Holds the current 'base' output (before logits/value_out layer).
        self._features = None
        self._values = None

        # Build a Model: fc + LSTM + [fc + fc] -> logits, value
        # Create fc_net (policy network: fc-lstm-fc)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
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

        # Create an lstm layer (policy network)
        self.lstm = nn.LSTM(self.fc_sizes[-1], self.lstm_state_size, batch_first=True)

        # Create post FC layers (policy network)
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

        # Create a fc_net (value network; separated from the policy net; critic: fc-lstm-fc)
        value_layers = []
        prev_value_layer_size = int(np.product(obs_space.shape))
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

        # Create an lstm layer (value network)
        self.value_lstm = nn.LSTM(self.value_fc_sizes[-1], self.value_lstm_state_size, batch_first=True)

        # Create post FC layers (value network)
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
        else:  # post_fc_sizes is empty: then connect the last layer to the lstm layer (policy side; actor)
            self.last_size = self.lstm_state_size
        if self.value_post_fc_sizes:
            self.last_value_size = self.value_post_fc_sizes[-1]
        else:
            self.last_value_size = self.value_lstm_state_size
        # Policy network's last layer (no activation)
        self.action_branch = nn.Linear(self.last_size, num_outputs)
        # Value network's last layer (no activation)
        self.value_branch = nn.Linear(self.last_value_size, 1)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.fc_net[-1]._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc_net[-1]._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.value_fc_net[-1]._model[0].weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
            self.value_fc_net[-1]._model[0].weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._values), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Get an output of the policy network (logits from the actor)
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

        # Get an output of the value network (value from the critic)
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









class MinimalLSTMShorter(TorchRNN, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_sizes=128,
            lstm_state_size=128,
            post_fc_sizes=128,
            value_fc_sizes=128,
            value_lstm_state_size=128,
            value_post_fc_sizes=128,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get configuration for this custom model
        self.fc_sizes = fc_sizes
        self.lstm_state_size = lstm_state_size
        self.post_fc_sizes = post_fc_sizes
        self.value_fc_sizes = value_fc_sizes
        self.value_lstm_state_size = value_lstm_state_size
        self.value_post_fc_sizes = value_post_fc_sizes

        # Define observation size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        # Base outputs before feeding into the last branches
        self._features = None
        self._values = None

        # Actor
        self.actor_fc1 = nn.Linear(self.obs_size, self.fc_sizes)
        self.actor_lstm = nn.LSTM(self.fc_sizes, self.lstm_state_size, batch_first=True)
        self.actor_fc2 = nn.Linear(self.lstm_state_size, self.post_fc_sizes)
        self.action_branch = nn.Linear(self.post_fc_sizes, num_outputs)

        # Critic
        self.value_fc1 = nn.Linear(self.obs_size, self.value_fc_sizes)
        self.value_lstm = nn.LSTM(self.value_fc_sizes, self.value_lstm_state_size, batch_first=True)
        self.value_fc2 = nn.Linear(self.value_lstm_state_size, self.value_post_fc_sizes)
        self.value_branch = nn.Linear(self.value_post_fc_sizes, 1)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.actor_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.actor_fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.value_fc1.weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
            self.value_fc1.weight.new(1, self.value_lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._values), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Compute actor outputs
        x = nn.functional.relu(self.actor_fc1(inputs))
        x, [h1, c1] = self.actor_lstm(x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        self._features = nn.functional.relu(self.actor_fc2(x))
        action_out = self.action_branch(self._features)

        # Compute critic outputs
        x2 = nn.functional.relu(self.value_fc1(inputs))
        x2, [h2, c2] = self.value_lstm(x2, [torch.unsqueeze(state[2], 0), torch.unsqueeze(state[3], 0)])
        self._values = nn.functional.relu(self.value_fc2(x2))

        return action_out, [torch.squeeze(h1, 0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0)]
