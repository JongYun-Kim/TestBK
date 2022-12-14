from gym.spaces import Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()


class FCActionMaskModelJY(TorchModelV2, nn.Module):
    """
    Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    PyTorch version of above ActionMaskingModel.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        # Get the original observation space from the input as it has been preprocessed
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # print(f"action_mask.dtype = {action_mask.dtype}")
        # print(f"inf_mask = {inf_mask.dtype}")
        # print(f"logits = {logits.dtype}")
        # print(f"masked_logits = {masked_logits.dtype}")

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


# class TorchCustomModelSimpleFC(TorchModelV2, nn.Module):
#     """Example of a PyTorch custom model that just delegates to a fc-net."""
#
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name
#         )
#         nn.Module.__init__(self)
#
#         self.torch_sub_model = TorchFC(
#             obs_space, action_space, num_outputs, model_config, name
#         )
#
#     def forward(self, input_dict, state, seq_lens):
#         input_dict["obs"] = input_dict["obs"].float()
#         fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
#         return fc_out, []
#
#     def value_function(self):
#         return torch.reshape(self.torch_sub_model.value_function(), [-1])
