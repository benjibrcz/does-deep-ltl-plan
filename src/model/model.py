from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl.ltl_net import LTLNet
from model.mixed_distribution import MixedDistribution
from preprocessing.vocab import VOCAB
from model.policy import ContinuousActor
from model.policy import DiscreteActor
from utils import torch_utils
from utils.torch_utils import get_number_of_params


class ChainedDistanceHead(nn.Module):
    """
    Auxiliary head that predicts chained distances for planning.

    Given the combined embedding, predicts d(agent→intermediate) + d(intermediate→goal).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding):
        return self.net(embedding)


class TransitionHead(nn.Module):
    """
    Auxiliary head that predicts next-state env features given current features and action.

    This forces the network to learn a transition/dynamics model, which is useful for planning.
    The prediction target is env_features[t+1], predicted from env_features[t] + action[t].
    """

    def __init__(self, env_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.env_dim = env_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(env_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_dim)  # Predict next env features
        )

    def forward(self, env_features, action):
        """
        Args:
            env_features: Current environment features [batch, env_dim]
            action: Action taken [batch, action_dim]
        Returns:
            Predicted next env features [batch, env_dim]
        """
        x = torch.cat([env_features, action], dim=1)
        return self.net(x)


class Model(nn.Module):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 ltl_net: nn.Module,
                 env_net: Optional[nn.Module],
                 aux_head: Optional[nn.Module] = None,
                 transition_head: Optional[nn.Module] = None,
                 ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.ltl_net = ltl_net
        self.env_net = env_net
        self.aux_head = aux_head
        self.transition_head = transition_head
        self.recurrent = False

    def compute_env_embedding(self, obs):
        """Compute just the environment embedding (before LTL)."""
        return self.env_net(obs.features) if self.env_net is not None else obs.features

    def compute_embedding(self, obs):
        env_embedding = self.compute_env_embedding(obs)
        ltl_embedding = self.ltl_net(obs.seq)
        return torch.cat([env_embedding, ltl_embedding], dim=1)

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)
        return dist, value

    def forward_with_aux(self, obs):
        """Forward pass that also returns auxiliary prediction."""
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)

        if self.aux_head is not None:
            aux_pred = self.aux_head(embedding).squeeze(1)
        else:
            aux_pred = None

        return dist, value, aux_pred


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
        use_aux_head: bool = False,
        aux_hidden_dim: int = 64,
        use_transition_head: bool = False,
        transition_hidden_dim: int = 64,
) -> Model:
    if len(VOCAB) <= 3:
        raise ValueError('VOCAB not initialized')
    obs_shape = env.observation_space['features'].shape
    print(f'Observation shape: {obs_shape}')
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        assert len(obs_shape) == 1
        env_net = None
        env_embedding_dim = obs_shape[0]

    embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
    ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)
    print(f'Num LTLNet params: {get_number_of_params(ltl_net)}')

    combined_dim = env_embedding_dim + ltl_net.embedding_dim

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[combined_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[combined_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([combined_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    # Optional auxiliary head for chained distance prediction
    aux_head = None
    if use_aux_head:
        aux_head = ChainedDistanceHead(combined_dim, aux_hidden_dim)
        print(f'Num AuxHead params: {get_number_of_params(aux_head)}')

    # Optional transition head for next-state prediction
    # Uses raw observation features (not env_embedding) since that's what we store during collection
    transition_head = None
    if use_transition_head:
        raw_obs_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape
        transition_head = TransitionHead(raw_obs_dim, action_dim, transition_hidden_dim)
        print(f'Num TransitionHead params: {get_number_of_params(transition_head)}')

    model = Model(actor, critic, ltl_net, env_net, aux_head, transition_head)

    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"], strict=False)
    return model
