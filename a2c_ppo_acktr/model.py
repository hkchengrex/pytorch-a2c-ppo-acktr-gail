import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from pysc2.lib import actions
from pysc2.lib import features


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, map_features, flat_features, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        '''
        if base_kwargs is None:
            base_kwargs = {}
            
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        '''
        ###
        image_shape = obs_shape.spaces['feature_screen'].shape
        non_image_shape = obs_shape.spaces['info_discrete'].shape
        self.base = MixBase(image_shape[0], non_image_shape[0], map_features, flat_features, **base_kwargs)
        ###

        '''
        else:
            print(obs_shape, action_space)
            self.base = base(obs_shape[0], **base_kwargs)      
        
        '''

        '''
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        '''

        ###
        num_outputs = action_space.spaces['discrete_output'].shape[0]
        self.dist_dis = Bernoulli(self.base.output_size, num_outputs)

        num_outputs = action_space.spaces['continous_output'].shape[0]
        self.dist_con = DiagGaussian(self.base.output_size, num_outputs)
        ###

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    #######
    def act(self, image_inputs, non_image_inputs, rnn_hxs, masks, deterministic=False):

        value, actor_features, rnn_hxs = self.base(image_inputs, non_image_inputs, rnn_hxs, masks)

        ####

        discrete_dist = self.dist_dis(actor_features)
        con_dist = self.dist_con(actor_features)

        if deterministic:
            discrete_action = discrete_dist.mode()
            continous_action = con_dist.mode()
        else:
            discrete_action = discrete_dist.sample()
            continous_action = con_dist.sample()

        action_log_probs = con_dist.log_probs(continous_action) + discrete_dist.log_probs(discrete_action)

        con_dist_entropy = con_dist.entropy().mean()
        discrete_dist_entropy = discrete_dist.entropy().mean()

        ####

        return value, discrete_action, continous_action, action_log_probs, rnn_hxs

    #######

    #######
    def get_value(self, image_inputs, non_image_inputs, rnn_hxs, masks):
        value, _, _ = self.base(image_inputs, non_image_inputs, rnn_hxs, masks)
        return value

    #######

    #######
    def evaluate_actions(self, image_inputs, non_image_inputs, rnn_hxs, masks, dis_action, con_action):
        value, actor_features, rnn_hxs = self.base(image_inputs, non_image_inputs, rnn_hxs, masks)

        #######
        discrete_dist = self.dist_dis(actor_features)
        con_dist = self.dist_con(actor_features)
        #######

        #######

        action_log_probs = con_dist.log_probs(dis_action) + discrete_dist.log_probs(con_action)
        dist_entropy = discrete_dist.entropy().mean() + con_dist.entropy().mean()
        #######

        return value, action_log_probs, dist_entropy, rnn_hxs
    #######


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MixBase(NNBase):
    def __init__(self, num_image_inputs, num_non_image_inputs, map_features, flat_features, recurrent=False, hidden_size=512):
        super(MixBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.map_features = map_features
        self.flat_features = flat_features
        self.num_image_inputs = num_image_inputs
        self.num_non_image_inputs = num_non_image_inputs
        print(self.num_non_image_inputs)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_image_inputs, 16, 5, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(16, 32, 3, stride=1)), nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        state_channels = 32 + num_non_image_inputs  # stacking screen, info
        self.fc = nn.Sequential(init_(nn.Linear(78 * 78 * state_channels, 256)), nn.ReLU())

        self.actor = nn.Sequential(init_(nn.Linear(256, hidden_size)), nn.Tanh())
        self.critic = nn.Sequential(init_(nn.Linear(256, 1)), nn.Tanh())
        self.embed_screen = self.init_embed_obs(features.SCREEN_FEATURES, self._embed_spatial)

        self.embed_flat = self.init_embed_obs(self.flat_features, self._embed_flat)

        self.train()

    def forward(self, inputs_image, inputs_non_image, rnn_hxs, masks):

        embed_screen = self.embed_obs(inputs_image, self.embed_screen, self.make_one_hot_2d)
        embed_flat = self.embed_obs_flat(inputs_non_image, self.embed_flat, self.make_one_hot_1d)

        # Spatial Convnet
        inputs_image = self.main(embed_screen)

        '''
        x = Variable(
            inputs_non_image.repeat(
                inputs_non_image.shape[0],
                math.ceil(inputs_image.shape[2] * inputs_image.shape[3] / inputs_non_image.shape[0])
            ).resize_(
                inputs_non_image.shape[0], 1, inputs_image.shape[2], inputs_image.shape[3]
            )
        )
        '''
        # Non Spatial Vector BroadCast
        embed_flat = embed_flat.unsqueeze(2).unsqueeze(3).expand(embed_flat.size(0), embed_flat.size(1), 78, 78)

        # Concat Obervation
        x_state = torch.cat((inputs_image, embed_flat), dim=1)  # concat along channel dimension

        # Non Spatial
        non_spatial = x_state.view(x_state.shape[0], -1)
        non_spatial = self.fc(non_spatial)

        # Policy Out
        non_spatial_policy = self.actor(non_spatial)

        # Critic Out
        value = self.critic(non_spatial)

        return value, non_spatial_policy, rnn_hxs

    def _embed_flat(self, in_, out_):
        return self._linear_init(in_, out_)

    def _linear_init(self, in_, out_):
        relu_gain = nn.init.calculate_gain('relu')
        linear = nn.Linear(in_, out_)
        linear.weight.data.mul_(relu_gain)
        return nn.DataParallel(linear)

    def make_one_hot_1d(self, labels, dtype, C=2):
        '''
        Reference: https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
        Parameters
        ----------
        labels : N, where N is batch size.
        dtype: Cuda or not
        C : number of classes in labels.

        Returns
        -------
        target : N x C
        '''
        out = Variable(dtype(labels.size(0), C).zero_())
        index = labels.contiguous().view(-1, 1).long()
        return out.scatter_(1, index, 1)

    def init_embed_obs(self, spec, embed_fn):
        """
            Define network architectures
            Each input channel is processed by a Sequential network
        """
        out_sequence = {}
        for s in spec:
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                ###
                dims = max(dims, 1)
                ###
                sequence = nn.DataParallel(nn.Sequential(
                    embed_fn(s.scale, 1),
                    nn.ReLU(True)))
                out_sequence[s.index] = sequence
        return out_sequence

    def _embed_spatial(self, in_, out_):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        return init_(nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0))

    def make_one_hot_2d(self, labels, dtype, C=2):
        '''
        Reference: http://jacobkimmel.github.io/pytorch_onehot/
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.LongTensor
            N x 1 x H x W, where N is batch size.
        dtype: Cuda or not
        C : number of classes in labels.

        Returns
        -------
        target : N x C x H x W
        '''
        one_hot = Variable(dtype(labels.size(0), C, labels.size(2), labels.size(3)).zero_())
        target = one_hot.scatter_(1, labels.long(), 1)
        return target

    def embed_obs(self, obs, networks, one_hot):
        """
            Embed observation channels
        """
        # Channel dimension is 1
        feats = torch.chunk(obs, self.num_image_inputs, dim=1)
        out_list = []
        i = 0
        for feat, s in zip(feats, self.map_features):
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = one_hot(feat, torch.cuda.FloatTensor, C=s.scale)
                out = networks[s.index](indices.float())
            elif s.type == features.FeatureType.SCALAR:
                out = self._log_transform(feat, s.scale)
            else:
                raise NotImplementedError
            out_list.append(out)
            i += 1
        # Channel dimension is 1
        return torch.cat(out_list, 1)

    def embed_obs_flat(self, obs, networks, one_hot):
        """
            Embed observation channels
        """
        # Channel dimension is 1
        feats = torch.chunk(obs, self.num_non_image_inputs, dim=1)
        out_list = []
        for feat, s in zip(feats, self.flat_features):
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = one_hot(feat, torch.cuda.FloatTensor, C=s.scale)
                out = networks[s.index](indices.float())
            elif s.type == features.FeatureType.SCALAR:
                out = self._log_transform(feat, s.scale)
            else:
                raise NotImplementedError
            out_list.append(out)
        # Channel dimension is 1
        return torch.cat(out_list, 1)

    def _log_transform(self, x, scale):
        return torch.log(8 * x / scale + 1)
