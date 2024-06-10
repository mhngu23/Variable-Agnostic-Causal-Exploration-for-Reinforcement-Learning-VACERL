import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import math
import src.Causal_Discovery_Algorithm.plot_heatmap as plot_heatmap
import src.Causal_Discovery_Algorithm.utils as utils
import src.Causal_Discovery_Algorithm.intrinsic_reward as IR
import src.Causal_Discovery_Algorithm.Attention.transformer_with_attention as transformer_with_attention
import src.Causal_Discovery_Algorithm.AutoEncoder.image_encoder as image_encoder
from src.Causal_Discovery_Algorithm.Causality.structure_causal_graph import SCM

import copy


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

MH_ACTION = {
        0:[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1:[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2:[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3:[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4:[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5:[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6:[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7:[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8:[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9:[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        10:[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        11:[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        }

ACTION = {
        0:[1, 0, 0, 0, 0, 0, 0],
        1:[0, 1, 0, 0, 0, 0, 0],
        2:[0, 0, 1, 0, 0, 0, 0],
        3:[0, 0, 0, 1, 0, 0, 0],
        4:[0, 0, 0, 0, 1, 0, 0],
        5:[0, 0, 0, 0, 0, 1, 0],
        6:[0, 0, 0, 0, 0, 0, 1],
        }


def get_action(env, configuration, policy=None, ob=None):
    if policy is None:
        if configuration["env_name"] in FROZEN_LAKE_ENV or configuration["env_name"] in MH_ENV:
            return torch.randint(0, env.action_space.n, (1,)).item()
        else:
            return torch.randint(0, env.action_space.n-1, (1,))

def intervention_sampling(env, configuration, obs_pos_mapping, episode_number = 0, previous_record_episode = None, policy=None):
    """
    This function returns an episode from intervention sampling. 
    If an episode is completed successfully (assuming we know where is the goal) -> return episode for further analysis.
    Can test if the episode is not terminate successfully as well.
    """
    episode = []
    image_mapping = {}
    obs_pos_mapping = {}
    extend_image_buffer = []

    # Reset environment
    if configuration["env_name"] not in MH_ENV:
        ob, _ = env.reset(seed=1)
    else:
        ob = env.reset()
        ob = ob["colors"]

    done = False
    count = 0
    truncated = False

    while done is False:
        action = get_action(env, configuration)
        if configuration["env_name"] in FROZEN_LAKE_ENV:
            next_ob, rew, done, _, _ = env.step(action)
        elif configuration["env_name"] in MH_ENV:
            ob_before_flatten = copy.deepcopy(ob)
            extend_image_buffer.append(ob_before_flatten)
            next_ob, rew, done, _ = env.step(action)
            if count == 500:
                truncated == True
            next_ob = next_ob["colors"]
        else:
            agent_pos = env.get_agent_position()
            agent_frame = env.get_frame()
            ob_before_flatten = copy.deepcopy(ob)
            extend_image_buffer.append(ob_before_flatten)
            next_ob, rew, done, truncated, _ = env.step(action)
        
        count += 1
     
        if configuration["env_name"] in FROZEN_LAKE_ENV:
            step = (ob, action)
        elif configuration["env_name"] in MH_ENV:
            ob = ob.flatten()
            action = MH_ACTION.get(int(action))
            step = np.concatenate((ob, action)).tolist()
            if rew == 1 and previous_record_episode is not None:
                step = previous_record_episode[-1]
        else:
            ob = ob.flatten()
            original_action = action
            action = ACTION.get(int(action))
            step = np.concatenate((ob[:147], action)).tolist()
            
            if env.is_goal and previous_record_episode is not None:
                step = previous_record_episode[-1]

            # Only activate if we need to plot attention -> Comment out otherwise
            obs_pos_mapping[tuple(step)] = [agent_pos, original_action]

            # Only activate if we need to record image dataframe
            image_mapping[tuple(step)] = [agent_frame, count, episode_number, ob_before_flatten]
        
        episode.append(step)

        # if done or truncated:
        #     if rew == 1:
        #         if previous_record_episode is None:
        #             previous_record_episode = episode
        #             return episode, image_mapping, previous_record_episode, extend_image_buffer
        #             # return episode, image_mapping, previous_record_episode, count
        #         else:
        #             if episode[-1] == previous_record_episode[-1]:      
        #                 return episode, image_mapping, previous_record_episode, extend_image_buffer
        #                 # return episode, image_mapping, previous_record_episode, count
        #             else:
        #                 return None, None, previous_record_episode, extend_image_buffer
        #                 # return None, None, previous_record_episode, count
        #     else:
        #         if previous_record_episode is None:          
        #             return None, None, None, extend_image_buffer
        #             # return None, None, None, count
        #         else:
        #             return None, None, previous_record_episode, extend_image_buffer   
        #             # return None, None, previous_record_episode, count

        if done or truncated:
            if hasattr(env, "is_goal"):
                if env.is_goal:
                    if previous_record_episode is None:
                        previous_record_episode = episode
                        return episode, image_mapping, obs_pos_mapping, previous_record_episode, extend_image_buffer, count
                    else:
                        if episode[-1] == previous_record_episode[-1]:      
                            return episode, image_mapping, obs_pos_mapping, previous_record_episode, extend_image_buffer, count
                        else:
                            return None, None, None, previous_record_episode, None, count
                else:
                    if previous_record_episode is None:          
                        return None, None, None, None, None, count
                    else:
                        return None, None, None, previous_record_episode, None, count      
            else:
                if rew == 1:
                    if previous_record_episode is None:
                        previous_record_episode = episode
                        return episode, image_mapping, obs_pos_mapping, previous_record_episode, extend_image_buffer, count
                    else:
                        if episode[-1] == previous_record_episode[-1]:      
                            return episode, image_mapping, obs_pos_mapping, previous_record_episode, extend_image_buffer, count
                        else:
                            return None, None, None, previous_record_episode, None, count
                else:
                    if previous_record_episode is None:          
                        return None, None, None, None, None, count
                    else:
                        return None, None, None, previous_record_episode, None, count
        else:
            ob = next_ob

class ProbabilisticModel(nn.Module):
    def __init__(self, input_size):
        super(ProbabilisticModel, self).__init__()
        self.linear_1 = nn.Linear(input_size, 1000)
        self.linear_2 = nn.Linear(1000, 100)
        self.linear_3 = nn.Linear(100, 50)
        self.fc_mean = nn.Linear(50, 1)
        self.fc_std = nn.Linear(50, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        mean = self.fc_mean(x)
        var = torch.exp(self.fc_std(x))  # Ensure std is positive
        return mean, var

_LOG_2PI = math.log(2 * math.pi)

# Define gaussian negative log likelihood
def gaussian_log_likelihood_loss(pred, target, with_logvar=False,
                                 fixed_variance=None, detach_mean=False,
                                 detach_var=False):
    mean = pred[0]
    if detach_mean:
        mean = mean.detach()

    if with_logvar:
        logvar = pred[1]
        if detach_var:
            logvar = logvar.detach()

        if fixed_variance is not None:
            logvar = torch.ones_like(mean) * math.log(fixed_variance)
        ll = -0.5 * ((target - mean)**2 * (-logvar).exp() + logvar + _LOG_2PI)
    else:
        var = pred[1]
        if detach_var:
            var = var.detach()

        if fixed_variance is not None:
            var = torch.ones_like(mean) * fixed_variance
        ll = -0.5 * ((target - mean)**2 / var + torch.log(var) + _LOG_2PI)

    return -torch.sum(ll, axis=-1)


def train_mean_std_model(buffer, env_name):
    X = []
    y = []
    for episode in buffer:
        for i in range(len(episode) - 1):
            inp_seq = episode[i]
            out_seq = episode[-1]
            X.append(torch.tensor(inp_seq))
            y.append(torch.tensor(out_seq))
    
    if env_name in FROZEN_LAKE_ENV:
        input_size = 2
    elif env_name in MH_ENV:
        input_size = 1669
    else:
        input_size = 154
    model = ProbabilisticModel(input_size=input_size)

    criterion = gaussian_log_likelihood_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5000):
        print("Epoch", epoch)
        loss = 0
        for input_sequence, target_output in zip(X, y):
            input_sequence_t = torch.tensor(input_sequence, dtype=torch.float32)
            target_output_t = torch.tensor(target_output, dtype=torch.float32)

            optimizer.zero_grad()
            mean, var = model(input_sequence_t)
            predict = [mean, var]
            loss += criterion(predict, target_output_t)
            # print(loss)
        # print(loss)
        loss.backward()
        optimizer.step()
    
    return model


def train_cid(index, env, data_storage, configuration, model_dict, wand_reward_run = None, policy=None):
    """
    This function performs the causal discovery algorithm.
    """
    # SAMPLING BLOCK
    print("---------------------------Begin Sampling-----------------------------------------------")
    previous_record_episode = None
    step = 0
    episode_number = 0
    if index == 0:
        while step < int(configuration["head_timestep"]):
            episode, image_mapping, obs_pos_mapping, previous_record_episode, extend_image_buffer, count = intervention_sampling(env, configuration, data_storage["obs_pos_mapping"], episode_number, previous_record_episode)            
            step += count
            episode_number += 1
            print(step)
            if episode is not None:
                data_storage = utils.update_buffer(configuration, data_storage, episode, extend_image_buffer, image_mapping, obs_pos_mapping)
                print("Length of replay buffer", len(data_storage["replay_buffer"]))
            else: 
                print("Length of replay buffer", len(data_storage["replay_buffer"]))
                continue
    
    print("length replay buffer", len(data_storage["replay_buffer"]))

    data_storage["image_mapping"] = {}

    print(f"\n--------------------------Begin training CID-----------------------------------------------")
    cid_model = train_mean_std_model(data_storage["replay_buffer"], configuration["env_name"])
    
    return cid_model





    
    