"""
This script defines a custom intrinsic curiosity module (ICM)
and implements reactive exploration (RE) method
by computing extra intrinsic rewards from the extrinsic reward prediction model.
Ray 1.10...!!
"""

from gym.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchMultiCategorical
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder, one_hot as tf_one_hot
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType

from ray.rllib.utils.exploration.curiosity import Curiosity


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
F = None
if nn is not None:
    F = nn.functional


# @override
class ReactiveExploration(Curiosity):
    """
    Inherited from Curiosity, this class implements Reactive Exploration (RE).
    RE builds an extrinsic reward prediction model and computes an extra intrinsic reward on top of ICM
    based on the error of the reward prediction model.
    """
    def __init__(self,
                 action_space: Space,
                 *,
                 framework: str,
                 model: ModelV2,  # the original ICM
                 # model2: ModelV2,  # reward prediction model for Reactive Exploration
                 feature_dim: int = 288,  # Encoder output dimension: Obs -> Context(feature)
                 feature_net_config: Optional[ModelConfigDict] = None,
                 # re_net_config: Optional[ModelConfigDict] = None,  # RE net outputs 1-d, a predicted reward.
                 inverse_net_hiddens: Tuple[int] = (256,),
                 inverse_net_activation: str = "relu",
                 forward_net_hiddens: Tuple[int] = (256,),
                 forward_net_activation: str = "relu",
                 beta: float = 0.2,  # loss weight between forward/inverse models
                 eta: float = 1.0,  # Scaling factor of intrinsic reward
                 eta_re: float = 1.0,  # Scaling factor of RE reward
                 loss_coeff: float = 1.0,  # Scaling factor for loss in RE_net
                 lr: float = 1e-3,
                 lr_re: float = 1e-3,
                 sub_exploration: Optional[FromConfigSpec] = None,
                 re_net_hiddens: Tuple[int] = (256,),
                 re_net_activation: str = "relu",
                 **kwargs):

        for my_i in range(2):
            print("Reactive Exploration has been implemented !!!!")

        if not isinstance(action_space, (Discrete, MultiDiscrete)):
            raise ValueError(
                "Only (Multi)Discrete action spaces supported for Curiosity "
                "so far!")

        # TODO: init에서의 superclass가 무엇인지 체크!! -> 당장은 현재의 parent class 인 것 같은데,, 잘 되는 듯;;
        super().__init__(action_space,
                         model=model,
                         framework=framework,
                         feature_dim=feature_dim,
                         feature_net_config=feature_net_config,
                         inverse_net_activation=inverse_net_activation,
                         inverse_net_hiddens=inverse_net_hiddens,
                         forward_net_hiddens=forward_net_hiddens,
                         forward_net_activation=forward_net_activation,
                         beta=beta,
                         eta=eta,
                         lr=lr,
                         sub_exploration=sub_exploration,
                         **kwargs)

        if self.policy_config["num_workers"] != 0:
            raise ValueError(
                "Curiosity exploration currently does not support parallelism."
                " `num_workers` must be 0!")
        # Coefficients
        self.beta = beta
        self.eta = eta
        self.eta_re = eta_re
        self.loss_coeff = loss_coeff

        self.action_dim = self.action_space.n if isinstance(
            self.action_space, Discrete) else np.sum(self.action_space.nvec)

        # Generate a model for RE net (model2)
        # self.model2 = model2
        # self.device2 = None
        # if isinstance(self.model2, nn.Module):
        #     params2 = list(self.model2.parameters())
        #     if params2:
        #         self.device2 = params2[0].device
        # print(f"self.device2 = {self.device2}")
        # if re_net_config is None:
        #     re_net_config = self.policy_config["model"].copy()
        # self.re_net_config = re_net_config
        self.lr_re = lr_re
        # self._curiosity_re_net = ModelCatalog.get_model_v2(
        #     self.model2.obs_space,
        #     self.action_space,
        #     1,  # Output: a reward (scalar)
        #     model_config=self.re_net_config,
        #     framework=self.framework,
        #     name="re_net",
        # )
        self.re_net_hiddens = re_net_hiddens
        self.re_net_activation = re_net_activation
        self.obs_dim = len(self.model.obs_space.sample())
        print(f"observation_dimension = {self.obs_dim}")
        self._curiosity_re_net = self._create_fc_net(
            [self.obs_dim + self.action_dim] + list(
                self.re_net_hiddens) + [1],
            self.re_net_activation,
            name="re_net")

        # Settings for another model for ICM net (model1: feature net + forward net + inverse net)
        # Note: the model has been generated from the parent class, Exploration.
        self.feature_dim = feature_dim
        if feature_net_config is None:
            feature_net_config = self.policy_config["model"].copy()
        self.feature_net_config = feature_net_config
        self.inverse_net_hiddens = inverse_net_hiddens
        self.inverse_net_activation = inverse_net_activation
        self.forward_net_hiddens = forward_net_hiddens
        self.forward_net_activation = forward_net_activation
        self.lr = lr
        # Creates modules/layers inside the actual ModelV2.
        self._curiosity_feature_net = ModelCatalog.get_model_v2(
            self.model.obs_space,
            self.action_space,
            self.feature_dim,
            model_config=self.feature_net_config,
            framework=self.framework,
            name="feature_net",
        )
        self._curiosity_inverse_fcnet = self._create_fc_net(
            [2 * self.feature_dim] + list(self.inverse_net_hiddens) +
            [self.action_dim],
            self.inverse_net_activation,
            name="inverse_net")
        self._curiosity_forward_fcnet = self._create_fc_net(
            [self.feature_dim + self.action_dim] + list(
                self.forward_net_hiddens) + [self.feature_dim],
            self.forward_net_activation,
            name="forward_net")

        # Print for debugging (This class does not run on debugging mode on PyCharm
        print(f"inverse_net structure = {[2 * self.feature_dim] + list(self.inverse_net_hiddens) + [self.action_dim]}")
        print(f"inverse_net_hiddens = {self.inverse_net_hiddens}")
        print(f"forward_net structure = {[self.feature_dim + self.action_dim] + list(self.forward_net_hiddens) + [self.feature_dim]}")
        print(f"action_dim = {self.action_dim}")
        print(f"observation_space_dim = {self.model.obs_space}")
        print(f"feature_dim = {self.feature_dim}")
        print(f"beta = {self.beta}")
        print(f"eta = {self.eta}")
        print("feature net = (shown below)")
        print(self._curiosity_feature_net)
        print(f"feature_net_config = {self.feature_net_config}")
        print("re_net = (shown below)")
        print(self._curiosity_re_net)
        # print(f"re_net_config = {self.re_net_config}")
        print(f"model1 = {self.model}")
        # print(f"model2 = {self.model2})")

        # self.obs_dim = len(self.model.obs_space.sample())
        # print(f"observation_dimension = {self.obs_dim}")
        # self._curiosity_re_fcnet = self._create_fc_net(
        #     [self.obs_dim + self.action_dim] + list(
        #         self.re_net_hiddens) + [1],
        #     self.re_net_activation,
        #     name="re_net")
        # print(f"re_net = {self._curiosity_re_fcnet}")

        # TODO: (sven) if sub_exploration is None, use Trainer's default
        #  Exploration config.
        if sub_exploration is None:
            raise NotImplementedError
        self.sub_exploration = sub_exploration
        # This is only used to select the correct action

        self.exploration_submodule = from_config(
            cls=Exploration,
            config=self.sub_exploration,
            action_space=self.action_space,
            framework=self.framework,
            policy_config=self.policy_config,
            model=self.model,
            num_workers=self.num_workers,
            worker_index=self.worker_index,
        )

    @override(Curiosity)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        # Simply delegate to sub-Exploration module.
        return self.exploration_submodule.get_exploration_action(
            action_distribution=action_distribution,
            timestep=timestep,
            explore=explore)

    @override(Curiosity)
    def get_exploration_optimizer(self, optimizers):
        # Create, but don't add Adam for curiosity NN updating to the policy.
        # If we added and returned it here, it would be used in the policy's
        # update loop, which we don't want (curiosity updating happens inside
        # `postprocess_trajectory`).
        if self.framework == "torch":
            feature_params = list(self._curiosity_feature_net.parameters())
            inverse_params = list(self._curiosity_inverse_fcnet.parameters())
            forward_params = list(self._curiosity_forward_fcnet.parameters())
            re_params = list(self._curiosity_re_net.parameters())

            # Now that the Policy's own optimizer(s) have been created (from
            # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
            # we can add our curiosity sub-modules to the Policy's Model.
            self.model._curiosity_feature_net = \
                self._curiosity_feature_net.to(self.device)
            self.model._curiosity_inverse_fcnet = \
                self._curiosity_inverse_fcnet.to(self.device)
            self.model._curiosity_forward_fcnet = \
                self._curiosity_forward_fcnet.to(self.device)
            self._optimizer = torch.optim.Adam(forward_params + inverse_params + feature_params, lr=self.lr)

            # self.model2._curiosity_re_net = self._curiosity_re_net.to(self.device2)
            self.model._curiosity_re_net = self._curiosity_re_net.to(self.device)
            self._optimizer2 = torch.optim.Adam(re_params, lr=self.lr_re)
        else:
            self.model._curiosity_feature_net = self._curiosity_feature_net
            self.model._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet
            self.model._curiosity_forward_fcnet = self._curiosity_forward_fcnet
            # Feature net is a RLlib ModelV2, the other 2 are keras Models.
            self._optimizer_var_list = \
                self._curiosity_feature_net.base_model.variables + \
                self._curiosity_inverse_fcnet.variables + \
                self._curiosity_forward_fcnet.variables
            self._optimizer = tf1.train.AdamOptimizer(learning_rate=self.lr)
            # Create placeholders and initialize the loss.
            if self.framework == "tf":
                self._obs_ph = get_placeholder(
                    space=self.model.obs_space, name="_curiosity_obs")
                self._next_obs_ph = get_placeholder(
                    space=self.model.obs_space, name="_curiosity_next_obs")
                self._action_ph = get_placeholder(
                    space=self.model.action_space, name="_curiosity_action")
                self._forward_l2_norm_sqared, self._update_op = \
                    self._postprocess_helper_tf(
                        self._obs_ph, self._next_obs_ph, self._action_ph)

        return optimizers  # the optimizers are pristine because we want the models to be trained separately.

    @override(Curiosity)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        """Calculates phi values (obs, obs', and predicted obs') and ri.
        Also calculates forward and inverse losses and updates the curiosity
        module on the provided batch using our optimizer.
        """
        if self.framework != "torch":
            self._postprocess_tf(policy, sample_batch, tf_sess)
        else:
            self._postprocess_torch(policy, sample_batch)

    def _postprocess_tf(self, policy, sample_batch, tf_sess):
        # tf1 static-graph: Perform session call on our loss and update ops.
        if self.framework == "tf":
            forward_l2_norm_sqared, _ = tf_sess.run(
                [self._forward_l2_norm_sqared, self._update_op],
                feed_dict={
                    self._obs_ph: sample_batch[SampleBatch.OBS],
                    self._next_obs_ph: sample_batch[SampleBatch.NEXT_OBS],
                    self._action_ph: sample_batch[SampleBatch.ACTIONS],
                })
        # tf-eager: Perform model calls, loss calculations, and optimizer
        # stepping on the fly.
        else:
            forward_l2_norm_sqared, _ = self._postprocess_helper_tf(
                sample_batch[SampleBatch.OBS],
                sample_batch[SampleBatch.NEXT_OBS],
                sample_batch[SampleBatch.ACTIONS],
            )
        # Scale intrinsic reward by eta hyper-parameter.
        sample_batch[SampleBatch.REWARDS] = \
            sample_batch[SampleBatch.REWARDS] + \
            self.eta * forward_l2_norm_sqared

        return sample_batch

    def _postprocess_helper_tf(self, obs, next_obs, actions):
        with (tf.GradientTape()
              if self.framework != "tf" else NullContextManager()) as tape:
            # Push both observations through feature net to get both phis.
            phis, _ = self.model._curiosity_feature_net({
                SampleBatch.OBS: tf.concat([obs, next_obs], axis=0)
            })
            phi, next_phi = tf.split(phis, 2)

            # Predict next phi with forward model.
            predicted_next_phi = self.model._curiosity_forward_fcnet(
                tf.concat(
                    [phi, tf_one_hot(actions, self.action_space)], axis=-1))

            # Forward loss term (predicted phi', given phi and action vs
            # actually observed phi').
            forward_l2_norm_sqared = 0.5 * tf.reduce_sum(
                tf.square(predicted_next_phi - next_phi), axis=-1)
            forward_loss = tf.reduce_mean(forward_l2_norm_sqared)

            # Inverse loss term (prediced action that led from phi to phi' vs
            # actual action taken).
            phi_cat_next_phi = tf.concat([phi, next_phi], axis=-1)
            dist_inputs = self.model._curiosity_inverse_fcnet(phi_cat_next_phi)
            action_dist = Categorical(dist_inputs, self.model) if \
                isinstance(self.action_space, Discrete) else \
                MultiCategorical(
                    dist_inputs, self.model, self.action_space.nvec)
            # Neg log(p); p=probability of observed action given the inverse-NN
            # predicted action distribution.
            inverse_loss = -action_dist.logp(tf.convert_to_tensor(actions))
            inverse_loss = tf.reduce_mean(inverse_loss)

            # Calculate the ICM loss.
            loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss

        # Step the optimizer.
        if self.framework != "tf":
            grads = tape.gradient(loss, self._optimizer_var_list)
            grads_and_vars = [(g, v)
                              for g, v in zip(grads, self._optimizer_var_list)
                              if g is not None]
            update_op = self._optimizer.apply_gradients(grads_and_vars)
        else:
            update_op = self._optimizer.minimize(
                loss, var_list=self._optimizer_var_list)

        # Return the squared l2 norm and the optimizer update op.
        return forward_l2_norm_sqared, update_op

    def _postprocess_torch(self, policy, sample_batch):
        # Push both observations through feature net to get both phis.
        phis, _ = self.model._curiosity_feature_net({
            SampleBatch.OBS: torch.cat([
                # torch.from_numpy(sample_batch[SampleBatch.OBS]),
                # torch.from_numpy(sample_batch[SampleBatch.NEXT_OBS])
                torch.from_numpy(sample_batch[SampleBatch.OBS]).to(policy.device),
                torch.from_numpy(sample_batch[SampleBatch.NEXT_OBS]).to(policy.device),
            ])
        })
        phi, next_phi = torch.chunk(phis, 2)  # phi.shape = ([sample_batch_size, context_size])
        actions_tensor = torch.from_numpy(
            sample_batch[SampleBatch.ACTIONS]).long().to(policy.device)

        # Predict next phi with forward model.
        predicted_next_phi = self.model._curiosity_forward_fcnet(
            torch.cat(
                [phi, one_hot(actions_tensor, self.action_space).float()],
                dim=-1
            )
        )
        # Forward loss term (predicted phi', given phi and action vs actually observed phi').
        forward_l2_norm_sqared = 0.5 * torch.sum(  # f_l2_norm.shape = ([sample_batch_size])
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1)  # (1/2)*context differences.^2
        forward_loss = torch.mean(forward_l2_norm_sqared)  # scalar!! as in 0-dimension

        # Inverse loss term (prediced action that led from phi to phi' vs
        # actual action taken).
        phi_cat_next_phi = torch.cat([phi, next_phi], dim=-1)
        dist_inputs = self.model._curiosity_inverse_fcnet(phi_cat_next_phi)
        action_dist = TorchCategorical(dist_inputs, self.model) if \
            isinstance(self.action_space, Discrete) else \
            TorchMultiCategorical(
                dist_inputs, self.model, self.action_space.nvec)
        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        inverse_loss = -action_dist.logp(actions_tensor)
        inverse_loss = torch.mean(inverse_loss)

        # Compute intrinsic reward for Reactive Exploration
        # (1) Get rewards from sample_batch
        # TODO: 그대로 받아도 되는지 COPY 해야하는지 체크!!
        rewards_true = torch.from_numpy(sample_batch[SampleBatch.REWARDS]).long().to(policy.device)
        # (2) Get predicted rewards by forward-passing through re_net
        # TODO: 모델에서 바로뽑아서 값이 제대로 나오는지 체크
        # predicted_reward_signals = self.model2._curiosity_re_net(
        predicted_reward_signals = self.model._curiosity_re_net(
            torch.cat(
                [torch.from_numpy(sample_batch[SampleBatch.OBS]).to(policy.device), one_hot(actions_tensor, self.action_space).float()],
                dim=-1
            )
        )
        # (3) Get RE intrinsic reward
        re_l2_norm_sqared = 0.5 * torch.sum(torch.pow(predicted_reward_signals - rewards_true, 2.0), dim=-1)

        # Scale intrinsic reward by eta hyper-parameter.
        sample_batch[SampleBatch.REWARDS] = \
            sample_batch[SampleBatch.REWARDS] + \
            self.eta * forward_l2_norm_sqared.detach().cpu().numpy() + \
            self.eta_re * re_l2_norm_sqared.detach().cpu().numpy()
        print(f"Sample_batch size = {len(sample_batch[SampleBatch.REWARDS])}")
        print(f"avg reward = {np.mean(sample_batch[SampleBatch.REWARDS])}")

        # Compute RE loss and optimize it
        # TODO: Compute loss_re "before" updating the rewards from the trajectories in the sample batch
        loss_re = self.loss_coeff * torch.mean(re_l2_norm_sqared)
        self._optimizer2.zero_grad()
        loss_re.backward()
        self._optimizer2.step()
        print(f"loss_re = {loss_re}")

        # Calculate the ICM loss.
        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        # Perform an optimizer step.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        print(f"loss_icm = {loss * self.eta}")

        # Return the postprocessed sample batch (with the corrected rewards).
        return sample_batch

    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.
        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation (str): An activation specifier string (e.g. "relu").
        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = [
            tf.keras.layers.Input(
                shape=(layer_dims[0], ), name="{}_in".format(name))
        ] if self.framework != "torch" else []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act))
            else:
                layers.append(
                    tf.keras.layers.Dense(
                        units=layer_dims[i + 1],
                        activation=get_activation_fn(act),
                        name="{}_{}".format(name, i)))

        if self.framework == "torch":
            return nn.Sequential(*layers)
        else:
            return tf.keras.Sequential(layers)
