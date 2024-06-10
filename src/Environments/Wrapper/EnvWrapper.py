import math
import gym as gym2
import torch
import numpy as np

import copy
from gymnasium import logger, spaces
from gymnasium.core import Wrapper, ActionWrapper, ObservationWrapper, ObsType, Wrapper
from numpy import dot
from numpy.linalg import norm
from simhash import Simhash
import operator
from functools import reduce
from typing import Any

import src.Causal_Discovery_Algorithm.AutoEncoder.image_encoder as image_encoder
from src.Causal_Discovery_Algorithm.AutoEncoder.image_encoder import VAE_MG, VAE_MH, image_normalize


ACTION = {
        0:[1, 0, 0, 0, 0, 0, 0],
        1:[0, 1, 0, 0, 0, 0, 0],
        2:[0, 0, 1, 0, 0, 0, 0],
        3:[0, 0, 0, 1, 0, 0, 0],
        4:[0, 0, 0, 0, 1, 0, 0],
        5:[0, 0, 0, 0, 0, 1, 0],
        6:[0, 0, 0, 0, 0, 0, 1],
        }

ACTION_NOISY = {
        0:[1, 0, 0, 0, 0, 0, 0, 0],
        1:[0, 1, 0, 0, 0, 0, 0, 0],
        2:[0, 0, 1, 0, 0, 0, 0, 0],
        3:[0, 0, 0, 1, 0, 0, 0, 0],
        4:[0, 0, 0, 0, 1, 0, 0, 0],
        5:[0, 0, 0, 0, 0, 1, 0, 0],
        6:[0, 0, 0, 0, 0, 0, 1, 0],
        7:[0, 0, 0, 0, 0, 0, 0, 1],
        }

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

FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

def measure_cosine_distance(ob1, ob2):
        cosine_diff = dot(ob1, ob2.T)/(norm(ob1) * norm(ob2))    
        return cosine_diff

class CausalBonus(Wrapper):
    """
    Adds an exploration bonus based on causality
    """

    def __init__(self, env, intrinsic_reward, ob_diff_thres, configuration, model_dict, data_storage):
        """A wrapper that adds an exploration bonus to crucial causality state.

        Args:
            env: The environment to apply the wrapper
            intrinsic_reward: The dictionary of intrinsic reward <- result of causal discovery step
            ob_dff_thres: Threshold to compare the cosine distance between observation

        """
        super().__init__(env)
        self.intrinsic_reward = intrinsic_reward
        self.previous_obs = None
        self.previous_obs_flatten = None
        self.ob_diff_thres = ob_diff_thres
        self.configuration = configuration
        
        # Record data
        self.data_storage = data_storage
        self.episode = []
        self.image_mapping = {}
        self.extend_image_buffer = []
        self.obs_pos_mapping = {}
        
        # Encoding model
        self.VAE = VAE_MG()
        if model_dict["encoding_model"] is True:
            checkpoint = torch.load(f'model/model_VAE_{configuration["env_name"]}_{configuration["algorithm"]}_{configuration["train_datetime"]}.pth')
            self.VAE.load_state_dict(checkpoint['model_state_dict']) 

        # Count reward parameters
        self.ob_count = {}
    
        # Normalize intrinsic reward parameters
        self.estimate_return = 0
        self.intrinsic_return_buffer = []
        self.delta = 0
        self.mean = 0
        self.var = 1
        self.count = 0

    def updated_normalize_reward_parameters(self):
        if len(self.intrinsic_return_buffer) < 5:
            self.intrinsic_return_buffer.append(self.estimate_return)
        else:
            print("return buffer", self.intrinsic_return_buffer)
            batch_mean, batch_var, batch_count = np.mean(self.intrinsic_return_buffer), (np.std(self.intrinsic_return_buffer)+0.000001)**2, len(self.intrinsic_return_buffer)

            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / total_count
            ma = self.var * self.count
            mb = batch_var * batch_count 
            M2 = ma + mb + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            self.count = self.count + batch_count
            self.var = new_var
            self.mean = new_mean
            print("new var", self.var)
            print("new_mean", self.mean)

            self.intrinsic_return_buffer = []
            
        self.estimate_return = 0

    def record_extend_image_buffer(self):
        self.extend_image_buffer.append(copy.deepcopy(self.previous_obs))
    
    def record_episode(self, step, terminated, truncated, reward):
        if (terminated or truncated) and reward == 1 and self.data_storage["replay_buffer"] != []:
            step = self.data_storage["replay_buffer"][0][-1]
        self.episode.append(step)
    
    def record_obs_pos_mapping(self, step, agent_pos, original_action):
        self.obs_pos_mapping[tuple(step)] = [agent_pos, original_action]

    def record_image_mapping(self, step, agent_frame):
        self.image_mapping[tuple(step)] = agent_frame
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.previous_obs, info = super().reset(seed=seed, options=options)
        return self.previous_obs, info

    def step(self, action):
        if self.previous_obs is not None:
            self.previous_obs_flatten = self.previous_obs.flatten()
            temp_reward = []
            
            new_ob = np.array(self.previous_obs_flatten).reshape(7,7,3) 
            new_ob = self.VAE.encoder(image_normalize(new_ob, self.configuration["env_name"]))

            items = list(self.intrinsic_reward.items())

            for key, reward in items[1:]:

                current_ob = np.array(key[:147]).reshape(7,7,3)   
                current_ob = self.VAE.encoder(image_normalize(current_ob, self.configuration["env_name"]))

                try:
                    ob_diff = measure_cosine_distance(new_ob.detach().numpy(), current_ob.detach().numpy())
                except Exception as e:
                    print("Error:", e)
                    print(new_ob)
                    print(current_ob)
                    continue
            
                # Check with state diff threshold
                if tuple(key[-7:]) == tuple(ACTION.get(int(action))) and ob_diff > self.ob_diff_thres:
                    if not temp_reward:
                        temp_reward.append([key, reward, ob_diff])
                    else:
                        if ob_diff >= temp_reward[0][2]: # check with the last state diff that has been recorded in temp record
                            temp_reward = []
                            temp_reward.append([key, reward, ob_diff])
                        else:
                            continue

            # If temp is empty then intrinsic_reward = 0
            if not temp_reward:
                intrinsic_reward = 0
            else:
                ob, intrinsic_reward = temp_reward[0][0], temp_reward[0][1] # this is the value of the reward that has been recorded
                
                # Update counting dictionary
                pre_count = 0
                if ob in self.ob_count:
                    pre_count = self.ob_count[ob]
                new_count = pre_count + 1
                self.ob_count[ob] = new_count

                if intrinsic_reward == 1: 
                    intrinsic_reward = 0 
                else:
                    intrinsic_reward = 1 / math.sqrt(new_count) * intrinsic_reward
                    # intrinsic_reward = intrinsic_reward
                
        else:
            intrinsic_reward = 0
        
        self.record_extend_image_buffer()
        agent_pos = self.env.get_agent_position()
        agent_frame = self.env.get_frame()


        obs, reward, terminated, truncated, info = self.env.step(action)

        action = ACTION.get(int(action))
        step = np.concatenate((self.previous_obs_flatten, action)).tolist()
        self.record_episode(step, terminated, truncated, reward)
        self.record_obs_pos_mapping(step, agent_pos, action)
        self.record_image_mapping(step, agent_frame)
        
        # Normalize reward
        intrinsic_reward = intrinsic_reward/(np.sqrt(self.var))

        if terminated or truncated:
            self.updated_normalize_reward_parameters()
            if reward == 1:
                # Update data record
                if len(self.data_storage["replay_buffer"]) <= self.configuration["buffer_size"]: 
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_mapping"].update(self.image_mapping)
                    self.data_storage["obs_pos_mapping"].update(self.obs_pos_mapping)
                    self.data_storage["image_encoding_buffer"] += self.extend_image_buffer
                else:
                    remove_ep = self.data_storage["replay_buffer"].pop(0)
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_mapping"].update(self.image_mapping)
                    self.data_storage["obs_pos_mapping"].update(self.obs_pos_mapping)
                    self.data_storage["image_encoding_buffer"] = self.data_storage["image_encoding_buffer"][len(remove_ep):] + self.extend_image_buffer

            self.episode = []
            self.image_mapping = {}
            self.extend_image_buffer = []
            self.obs_pos_mapping = {}

        final_reward = reward + intrinsic_reward

        if self.estimate_return == 0:
            self.estimate_return = final_reward
        else:
            self.estimate_return = self.estimate_return * 0.99 + final_reward 

        self.previous_obs = obs

        return obs, final_reward, terminated, truncated, info

class CausalBonusMH(gym2.Wrapper):
    """
    Adds an exploration bonus based on causality
    """

    def __init__(self, env, intrinsic_reward, ob_diff_thres, configuration, model_dict, data_storage):
        """A wrapper that adds an exploration bonus to crucial causality state.

        Args:
            env: The environment to apply the wrapper
            intrinsic_reward: The dictionary of intrinsic reward <- result of causal discovery step
            ob_dff_thres: Threshold to compare the cosine distance between observation

        """
        super().__init__(env)
        self.intrinsic_reward = intrinsic_reward
        self.previous_obs = None
        self.previous_obs_flatten = None
        self.ob_diff_thres = ob_diff_thres
        self.configuration = configuration
        
        # Record data
        self.data_storage = data_storage
        self.episode = []
        self.image_mapping = {}
        self.extend_image_buffer = []
        self.obs_pos_mapping = {}
        self.ob_count = {}
        self.estimate_return = 0

        # Encoding Model
        self.VAE = VAE_MH()
        if model_dict["encoding_model"] is True:
            checkpoint = torch.load(f'model/model_VAE_{configuration["env_name"]}_{configuration["algorithm"]}_{configuration["train_datetime"]}.pth')
            self.VAE.load_state_dict(checkpoint['model_state_dict']) 

        # Count reward parameters
        self.ob_count = {}
    
        # Normalize intrinsic reward parameters
        self.estimate_return = 0
        self.intrinsic_return_buffer = []
        self.delta = 0
        self.mean = 0
        self.var = 1
        self.count = 0

    
    def record_extend_image_buffer(self):
        self.extend_image_buffer.append(copy.deepcopy(self.previous_obs))
    
    def record_episode(self, step, terminated, reward):
        if terminated and reward == 1 and self.data_storage["replay_buffer"] != []:
            step = self.data_storage["replay_buffer"][0][-1]
        self.episode.append(step)
    
    def record_obs_pos_mapping(self, step, agent_pos, original_action):
        self.obs_pos_mapping[tuple(step)] = [agent_pos, original_action]

    def record_image_mapping(self, step, agent_frame):
        self.image_mapping[tuple(step)] = agent_frame

    def updated_normalize_reward_parameters(self):
        if len(self.intrinsic_return_buffer) < 5:
            self.intrinsic_return_buffer.append(self.estimate_return)
        else:
            print("return buffer", self.intrinsic_return_buffer)
            batch_mean, batch_var, batch_count = np.mean(self.intrinsic_return_buffer), (np.std(self.intrinsic_return_buffer)+0.000001)**2, len(self.intrinsic_return_buffer)

            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / total_count
            ma = self.var * self.count
            mb = batch_var * batch_count 
            M2 = ma + mb + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            self.count = self.count + batch_count
            self.var = new_var
            self.mean = new_mean
            print("new var", self.var)
            print("new_mean", self.mean)

            self.intrinsic_return_buffer = []
            
        self.estimate_return = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.previous_obs = obs["colors"]
        return obs

    def step(self, action):
    
        if self.previous_obs is not None:
            self.previous_obs_flatten = self.previous_obs.flatten()
            temp_reward = []
            
            new_ob = np.array(self.previous_obs).reshape(21,79)
            new_ob = torch.mean(self.VAE.encoder(image_normalize(new_ob, self.configuration["env_name"])))

            items = list(self.intrinsic_reward.items())

            for key, reward in items[1:]:

                current_ob = np.array(key[:1659]).reshape(21, 79)  
                current_ob = torch.mean(self.VAE.encoder(image_normalize(current_ob, self.configuration["env_name"])))

                try:
                    ob_diff = measure_cosine_distance(new_ob.detach().numpy(), current_ob.detach().numpy())
                except Exception as e:
                    print("Error:", e)
                    print(new_ob)
                    print(current_ob)
                    continue
            
                # Check with state diff threshold
                if tuple(key[-10:]) == tuple(MH_ACTION.get(int(action))) and ob_diff > self.ob_diff_thres:
                    if not temp_reward:
                        temp_reward.append([key, reward, ob_diff])
                    else:
                        if ob_diff >= temp_reward[0][2]: # check with the last state diff that has been recorded in temp record
                            temp_reward = []
                            temp_reward.append([key, reward, ob_diff])
                        else:
                            continue

            # If temp is empty then intrinsic_reward = 0
            if not temp_reward:
                intrinsic_reward = 0
            else:
                ob, intrinsic_reward = temp_reward[0][0], temp_reward[0][1] # this is the value of the reward that has been recorded
                
                # Update counting dictionary
                pre_count = 0
                if ob in self.ob_count:
                    pre_count = self.ob_count[ob]
                new_count = pre_count + 1
                self.ob_count[ob] = new_count

                if intrinsic_reward == 1: 
                    intrinsic_reward = 0 
                else:
                    intrinsic_reward = 1 / math.sqrt(new_count) * intrinsic_reward
                
        else:
            intrinsic_reward = 0
        
        self.record_extend_image_buffer()

        obs, reward, terminated, info = self.env.step(action)

        action = MH_ACTION.get(int(action))
        step = np.concatenate((self.previous_obs_flatten, action)).tolist()

        self.record_episode(step, terminated, reward)
        
        # Normalize reward
        intrinsic_reward = intrinsic_reward/(np.sqrt(self.var))

        if terminated:
            self.updated_normalize_reward_parameters()
            # Update data record
            if reward == 1:
                if len(self.data_storage["replay_buffer"]) <= self.configuration["buffer_size"]: 
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_encoding_buffer"] += self.extend_image_buffer
                else:
                    remove_ep = self.data_storage["replay_buffer"].pop(0)
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_encoding_buffer"] = self.data_storage["image_encoding_buffer"][len(remove_ep):] + self.extend_image_buffer

            self.episode = []
            self.extend_image_buffer = []

        final_reward = reward + intrinsic_reward

        if self.estimate_return == 0:
            self.estimate_return = final_reward
        else:
            self.estimate_return = self.estimate_return * 0.99 + final_reward 


        self.previous_obs = obs["colors"]

        return obs, final_reward, terminated, info

class CausalBonusR(Wrapper):
    """
    Adds an exploration bonus based on causality
    """

    def __init__(self, env, intrinsic_reward, ob_diff_thres, configuration, model_dict, data_storage):
        """A wrapper that adds an exploration bonus to crucial causality state.

        Args:
            env: The environment to apply the wrapper
            intrinsic_reward: The dictionary of intrinsic reward <- result of causal discovery step
            ob_dff_thres: Threshold to compare the cosine distance between observation

        """
        super().__init__(env)
        self.intrinsic_reward = intrinsic_reward
        self.causal_subgoals = self.create_causal_subgoal_dict(self.intrinsic_reward)
        self.previous_obs = None
        self.previous_obs_flatten = None
        self.ob_diff_thres = ob_diff_thres
        self.configuration = configuration
        
        # Record data
        self.data_storage = data_storage
        self.episode = []
        self.image_mapping = {}
        self.extend_image_buffer = []
        self.obs_pos_mapping = {}

        # Count reward parameters
        self.ob_count = {}
    
        # Normalize intrinsic reward parameters
        self.estimate_return = 0
        self.intrinsic_return_buffer = []
        self.delta = 0
        self.mean = 0
        self.var = 1
        self.count = 0
    
    def updated_normalize_reward_parameters(self):
        if len(self.intrinsic_return_buffer) < 5:
            self.intrinsic_return_buffer.append(self.estimate_return)
        else:
            print("return buffer", self.intrinsic_return_buffer)
            batch_mean, batch_var, batch_count = np.mean(self.intrinsic_return_buffer), (np.std(self.intrinsic_return_buffer)+0.000001)**2, len(self.intrinsic_return_buffer)

            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / total_count
            ma = self.var * self.count
            mb = batch_var * batch_count 
            M2 = ma + mb + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            self.count = self.count + batch_count
            self.var = new_var
            self.mean = new_mean

            self.intrinsic_return_buffer = []
            
        self.estimate_return = 0
    
    def create_causal_subgoal_dict(self, intrinsic_reward):
        keys = list(intrinsic_reward.keys())[1:]
        count_depth = {}
        subgoal_sample = []
        for i in range(len(keys)):
            if i == 0:
                count_depth[keys[i]] = 1
            if i > 0:
                if intrinsic_reward.get(keys[i]) == intrinsic_reward.get(keys[i-1]):
                    count_depth[keys[i]] = count_depth.get(keys[i-1])
                else:
                    count_depth[keys[i]] = count_depth.get(keys[i-1]) + 1
        causal_subgoals_probability = {key: 1 / value for key, value in count_depth.items()}
        coefficient = {key: value/sum(causal_subgoals_probability.values())*10 for key, value in causal_subgoals_probability.items()}
        for key in coefficient:
            counts = coefficient.get(key)
            for _ in range(int(counts)):
                subgoal_sample.append(key)
        return subgoal_sample

    
    def record_episode(self, step, terminated, truncated, reward):
        if (terminated or truncated) and reward == 0 and self.data_storage["replay_buffer"] != []:
            step = self.data_storage["replay_buffer"][0][-1]
        self.episode.append(step)

    def record_obs_pos_mapping(self, step, achieved_goal):
        self.obs_pos_mapping[tuple(step)] = achieved_goal

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.previous_obs, info = super().reset(seed=seed, options=options)
        return self.previous_obs, info

    def step(self, action):
        if self.previous_obs is not None:
            self.previous_obs_extract = self.previous_obs["observation"]

            temp_reward = []
            
            new_ob = np.array(self.previous_obs_extract)

            items = list(self.intrinsic_reward.items())

            for key, reward in items[1:]:

                current_ob = np.array(key)[:-4]
                current_action = np.array(key)[-4:]

                try:
                    ob_diff = measure_cosine_distance(new_ob, current_ob)
                    action_diff =  measure_cosine_distance(np.array(action), current_action)
                except Exception as e:
                    print("Error:", e)
                    print(new_ob)
                    print(current_ob)
                    continue
            
                # Check with state diff threshold
                if action_diff > self.ob_diff_thres and ob_diff > self.ob_diff_thres:
                    if not temp_reward:
                        temp_reward.append([key, reward, ob_diff])
                    else:
                        if ob_diff >= temp_reward[0][2]: # check with the last state diff that has been recorded in temp record
                            temp_reward = []
                            temp_reward.append([key, reward, ob_diff])
                        else:
                            continue

            # If temp is empty then intrinsic_reward = 0
            if not temp_reward:
                intrinsic_reward = 0
            else:
                ob, intrinsic_reward = temp_reward[0][0], temp_reward[0][1] # this is the value of the reward that has been recorded
                
                # Update counting dictionary
                pre_count = 0
                if ob in self.ob_count:
                    pre_count = self.ob_count[ob]
                new_count = pre_count + 1
                self.ob_count[ob] = new_count

                if intrinsic_reward == 1: 
                    intrinsic_reward = 0 
                else:
                    intrinsic_reward = 1 / math.sqrt(new_count) * intrinsic_reward                
        else:
            intrinsic_reward = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        achieved_goal = obs["achieved_goal"]

        step = np.concatenate((self.previous_obs_extract, action)).tolist()
        self.record_episode(step, terminated, truncated, reward)
        
        self.record_obs_pos_mapping(step, achieved_goal)
        # Normalize reward
        intrinsic_reward = intrinsic_reward/(np.sqrt(self.var))

        if terminated or truncated:
            self.updated_normalize_reward_parameters()

            self.episode = []
            self.obs_pos_mapping = {}

        final_reward = reward + intrinsic_reward

        if self.estimate_return == 0:
            self.estimate_return = final_reward
        else:
            self.estimate_return = self.estimate_return * 0.99 + final_reward 

        self.previous_obs = obs

        return obs, final_reward, terminated, truncated, info
        

class FlatObsWrapper_ImageOnly(ObservationWrapper):
    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )
        
    def observation(self, obs):
        return obs["image"].flatten()