# The purpose of this script is just to check if it works without raylet die error under high load on it
# Usually raylet dies within 10 iterations
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ICM_org import ReactiveExplorationOrg
from myICM import ReactiveExploration


def main():
    is_your_icm = True
    # is_your_icm = False

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "CartPole-v0"
    config["num_workers"] = 0
    config["framework"] = "torch"
    config["exploration_config"] = {
        "type": ReactiveExploration if is_your_icm else ReactiveExplorationOrg,  # <- Use the Curiosity module for exploring.
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 32,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [64],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [64],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [64],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }
    # Configurations for the RE module
    if is_your_icm:
        config["exploration_config"]["eta_re"] = 1.0
        config["exploration_config"]["lr_re"] = 0.001  # Learning rate of the reward prediction model
        # config["exploration_config"]["re_net_config"] = {
        #         "fcnet_hiddens": [256],
        #         "fcnet_activation": "relu",
        # }
        config["exploration_config"]["re_net_hiddens"] = [16, 16]
        config["exploration_config"]["re_net_activation"] = "relu"

    ray.init()

    tune.run("PPO", config=config)

    print("done!")
    print("See the policy params")


if __name__ == "__main__":
    main()
