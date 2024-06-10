import os
import numpy as np
import gym
import torch 

import torch.nn as nn

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

MED_ENV = ["MH_3", "MH_5", "MH_7", "MH_8", "MH_9", "MH_10", "R_1"]

HARD_ENV = ["R_3"]

EASY_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV", "MG_1",  "MH_1", "MG_5"]


ACTION = {  0:[1, 0, 0, 0, 0, 0, 0],
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

# Get the current directory
current_dir = os.getcwd()

# Get the parent directory (one level up)
parent_dir = os.path.join(current_dir)
results_file_path = os.path.join(parent_dir, "Results")

import pandas as pd
import matplotlib.pyplot as plt


def log_result_train_task(change_in_training = None, algo_name = None, train_datetime = None):
    if change_in_training is not None:
        min_length_item = min(len(item) for item in change_in_training)    

        dict_random = {}
        for item in range(len(change_in_training)):
             dict_random.update({"change_in_training_" + str(item):change_in_training[item][0:min_length_item]})

        df = pd.DataFrame(dict_random)
        df.to_csv(f"{results_file_path}\\{algo_name}_{str(train_datetime.date())}_{str(train_datetime.hour)}_{str(train_datetime.minute)}.csv", index=False)    
    plt.show()
    return

class UpdateEvalCallback(EvalCallback):
    def __init__(self, eval_env, env_name, algo_name, train_datetime,
                  callback_on_new_best: BaseCallback | None = None, callback_after_eval: BaseCallback | None = None, 
                  n_eval_episodes: int = 5, eval_freq: int = 10000, log_path: str | None = None, best_model_save_path: str | None = None, 
                  deterministic: bool = True, render: bool = False, verbose: int = 1, warn: bool = True):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, 
                         n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.env_name = env_name
        self.algo_name = algo_name
        self.train_datetime = train_datetime
        # with open(f"Results/Task_Training_result/{train_datetime}_{self.env_name}_{self.algo_name}.txt", "a") as output_file:
        #     output_file.write(f"Algorithm number: {algorithm_number}\n")

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            with open(f"Results/Task_Training_result/{self.train_datetime}_{self.env_name}_{self.algo_name}.txt", "a") as output_file:
                output_file.write(f"Eval num_timesteps={self.num_timesteps}, " f"average_episode_reward={mean_reward:.2f} +/- {std_reward:.2f}\n")
                # output_file.write(f"Eval num_timesteps={self.num_timesteps}, " f"sum_episode_reward={np.sum(episode_rewards):.2f}\n")
                output_file.write(f"Episode length: {mean_ep_length} +/- {std_ep_length:.2f}\n")

            if self.verbose >= 1:
                # print(episode_rewards)
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class SuccessEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0, env_name = "", data_storage = {}, configuration = None, algo_name =""):
        super(SuccessEpisodeCallback, self).__init__(verbose)
        self.replay_buffer = []
        self.episode = []
        self.old_obs = None 
        self.env_name = env_name
        self.data_storage = data_storage
        self.configuration = configuration
        self.algo_name = algo_name
        self.extend_image_buffer = []
        self.previous_record_episode = self.data_storage["replay_buffer"][0]

    def _on_step(self) -> bool:
        if self.algo_name == "PPO": 
            if self.locals['n_steps'] == 0 and self.configuration["env_name"] not in MH_ENV:
                self.old_obs = self.locals['obs_tensor'].to('cpu').detach().numpy()[0]
            elif self.locals['n_steps'] == 0 and self.configuration["env_name"] in MH_ENV:
                self.old_obs = self.locals['obs_tensor']["colors"].to('cpu').detach().numpy()[0]
        else:
            self.old_obs = self.locals['new_obs']

        action = self.locals['actions'][0]

        if self.env_name in FROZEN_LAKE_ENV:
            step = (self.old_obs, action)
        elif self.env_name in MH_ENV:
            self.extend_image_buffer.append(self.old_obs)
            self.old_obs_flatten = self.old_obs.flatten()
            action = MH_ACTION.get(int(action))
            
            step = np.concatenate((self.old_obs_flatten, action)).tolist()

            if float(self.locals['rewards']) == 1.0:
                step  = self.previous_record_episode[-1]
        else:
            self.extend_image_buffer.append(self.old_obs)
            self.old_obs_flatten = self.old_obs.flatten()
            action = ACTION.get(int(action))

            step = np.concatenate((self.old_obs_flatten[:147], action)).tolist()

            if float(self.locals['rewards']) == 1.0:
                step  = self.previous_record_episode[-1]

        self.episode.append(step)
          
        # Check if the current episode is successful
        episode_done = self.locals['dones'][-1]
        if episode_done:
            if self.locals['rewards'] == 1:

                if len(self.data_storage["replay_buffer"]) <= self.configuration["buffer_size"]: 
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_encoding_buffer"] += self.extend_image_buffer
                else:
                    # If successfully trained add to buffer and remove the oldest experience
                    self.data_storage["replay_buffer"].pop(0)
                    self.data_storage["replay_buffer"].append(self.episode)
                    self.data_storage["image_encoding_buffer"] += self.extend_image_buffer
            
            self.episode = []
            self.extend_image_buffer = []

        if self.configuration["env_name"] not in MH_ENV:
            self.old_obs = self.locals['new_obs'][0]
        else:
            self.old_obs = self.locals['new_obs']["colors"][0]
        return True
        
def get_callback(configuration, evaluate_env, algo_name, data_storage):
	callback1 = WandbCallback(
		gradient_save_freq=100,
		verbose=1,)
    
	if configuration["env_name"] in EASY_ENV:
		callback2 = UpdateEvalCallback(evaluate_env, env_name = configuration["env_name"], 
                            algo_name = configuration["algorithm"], train_datetime = configuration["train_datetime"],
                            eval_freq=100, n_eval_episodes=50)
	elif configuration["env_name"] in MED_ENV:
		callback2 = UpdateEvalCallback(evaluate_env, env_name = configuration["env_name"], 
                            algo_name = configuration["algorithm"], train_datetime = configuration["train_datetime"],
                            eval_freq=10000, n_eval_episodes=50)
	elif configuration["env_name"] in HARD_ENV:
		callback2 = UpdateEvalCallback(evaluate_env, env_name = configuration["env_name"], 
                            algo_name = configuration["algorithm"], train_datetime = configuration["train_datetime"],
                            eval_freq=200000, n_eval_episodes=50)
	else:
		callback2 = UpdateEvalCallback(evaluate_env, env_name = configuration["env_name"], 
                                    algo_name = configuration["algorithm"], train_datetime = configuration["train_datetime"],
                                    eval_freq=1000000, n_eval_episodes=50)
        
	return callback1, callback2

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def get_policy_kwargs():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return policy_kwargs