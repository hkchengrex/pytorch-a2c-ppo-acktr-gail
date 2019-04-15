import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):

        ####

        image_shape = obs_shape.spaces['feature_screen'].shape
        non_image_shape = obs_shape.spaces['info_discrete'].shape

        self.image = torch.zeros(num_steps + 1, num_processes, *image_shape)
        self.non_image = torch.zeros(num_steps + 1, num_processes, *non_image_shape)

        ###

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)




        ###

        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        continuous_output_shape = action_space.spaces['continuous_output'].shape[0]
        discrete_output_shape = action_space.spaces['discrete_output'].shape[0]

        self.dis_actions = torch.zeros(num_steps, num_processes, discrete_output_shape)
        self.con_actions = torch.zeros(num_steps, num_processes, continuous_output_shape)

        ###

        '''
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        '''

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        ####

        self.image = self.image.to(device)
        self.non_image = self.non_image.to(device)

        ###

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)

        ####

        self.action_log_probs = self.action_log_probs.to(device)
        self.dis_actions = self.dis_actions.to(device)

        if self.con_actions is not None:
            self.con_actions = self.con_actions.to(device)
        ####


        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, image, non_image, recurrent_hidden_states, dis_actions, con_actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):

        ####
        self.image[self.step + 1].copy_(image)
        self.non_image[self.step + 1].copy_(non_image)

        ####

        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)

        ####

        self.dis_actions[self.step].copy_(dis_actions)
        if con_actions is not None:
            self.con_actions[self.step].copy_(con_actions)
        else:
            self.con_actions = None
        self.action_log_probs[self.step].copy_(action_log_probs)
        ####


        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):

        ####

        self.image[0].copy_(self.image[-1])
        self.non_image[0].copy_(self.non_image[-1])

        ####


        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                                          gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                                         gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
        for indices in sampler:

            ###

            image_batch = self.image[:-1].view(-1, *self.image.size()[2:])[indices]
            non_image_batch = self.non_image[:-1].view(-1, *self.non_image.size()[2:])[indices]


            ###


            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]


            ###

            dis_actions_batch = self.dis_actions.view(-1,
                                              self.dis_actions.size(-1))[indices]

            if self.con_actions is not None:
                con_actions_batch = self.con_actions.view(-1,
                                                  self.con_actions.size(-1))[indices]
            else:
                con_actions_batch = None

            ###



            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield image_batch, non_image_batch, recurrent_hidden_states_batch, dis_actions_batch, con_actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):

            ###

            image_batch = []
            non_image_batch = []

            ###

            recurrent_hidden_states_batch = []

            dis_actions_batch = []
            con_actions_batch = []

            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                ###
                image_batch.append(self.image[:-1, ind])
                non_image_batch.append(self.non_image[:-1, ind])
                ###

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])


                ###
                dis_actions_batch.append(self.dis_actions[:, ind])
                if self.con_actions is not None:
                    con_actions_batch.append(self.con_actions[:, ind])
                else:
                    con_actions_batch = None
                ###

                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)

            ###


            image_batch = torch.stack(image_batch, 1)
            non_image_batch = torch.stack(non_image_batch, 1)

            ###

            dis_actions_batch = torch.stack(dis_actions_batch, 1)

            if con_actions_batch is not None:
                con_actions_batch = torch.stack(con_actions_batch, 1)
            else:
                con_actions_batch = None

            ###



            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)

            image_batch = _flatten_helper(T, N, image_batch)
            non_image_batch = _flatten_helper(T, N, non_image_batch)

            ####
            dis_actions_batch = _flatten_helper(T, N, dis_actions_batch)

            if con_actions_batch is not None:
                con_actions_batch = _flatten_helper(T, N, con_actions_batch)
            else:
                con_actions_batch = None
            ####

            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield image_batch, non_image_batch, recurrent_hidden_states_batch, dis_actions_batch, con_actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
