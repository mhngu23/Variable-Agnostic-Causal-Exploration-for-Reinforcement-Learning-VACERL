import os
import argparse
import warnings
import datetime

import wandb

import main_utils

import src.Task_Training_Algorithm.PPOAlgorithm as PPOAlgorithm
import src.Task_Training_Algorithm.HERAlgorithm as HERAlgorithm
import src.Task_Training_Algorithm.HACAlgorithm as HACAlgorithm
from src.Causal_Discovery_Algorithm.discover_subgoal_hierarchy import *
from src.Causal_Discovery_Algorithm.cid_in_rl_adaptation import *
from src.Environments.Wrapper.EnvWrapper import *
from minigrid.wrappers import ReseedWrapper


FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

current_time = datetime.datetime.now()
train_datetime = f"{current_time.year}{current_time.month}{current_time.day}_{current_time.hour}_{current_time.minute}"

# os.environ["WANDB_MODE"] = "dryrun"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="A program to run causal RL algorithm.", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

parser = main_utils.add_argument(parser)

def reseed_wrapper(train_env, evaluate_env, seed_type):
    if seed_type == "multi":
        evaluate_env = ReseedWrapper(evaluate_env, seeds=[1, 5, 9, 11, 14, 19, 21, 25, 30, 34], seed_idx=0)
        train_env = ReseedWrapper(train_env, seeds=[1, 5, 9, 11, 14, 19, 21, 25, 30, 34], seed_idx=0)
    elif seed_type == "one":
        evaluate_env = ReseedWrapper(evaluate_env, seeds=[1], seed_idx=0)
        train_env = ReseedWrapper(train_env, seeds=[1], seed_idx=0)    
    else:
        return train_env, evaluate_env
    return train_env, evaluate_env

def train_function(env, args):
    configuration = main_utils.get_main_configuration(args, train_datetime)
    
    data_storage = {
        "replay_buffer": [],
        "image_encoding_buffer": [],
        "obs_pos_mapping": {},
        "image_mapping": {},
    }

    model_dict = {"attention_model" : False,
                  "encoding_model": False}

    total_timesteps = args.total_timestep
    loop = args.loop
    seed_type = args.seed_type

    if configuration["algorithm"] == "Based":
        training_config, run = main_utils.get_training_config(configuration, total_timesteps=total_timesteps)
        if configuration["env_name"] not in MH_ENV:
            train_env, eval_env = reseed_wrapper(env, env, seed_type)
            policy, _ = PPOAlgorithm.train_and_evaluate(configuration, train_env, eval_env, training_config, run,  data_storage=data_storage)
        else:
            policy, _ = PPOAlgorithm.train_and_evaluate(configuration, env, env, training_config, run,  data_storage=data_storage)
        run.finish()
            
    elif configuration["algorithm"] == "Updated":
        run = None
        policy = None
        wandb_reward_run = None
        intrinsic_reward = {}
        for j in range(1, loop):
            # Train and Evaluate Task 
            training_config, run = main_utils.get_training_config(configuration, run=run, total_timesteps=total_timesteps/loop, policy=policy)
            update_training_env = main_utils.get_update_training_environment(args, configuration, intrinsic_reward, data_storage, model_dict)
            
            if configuration["env_name"] not in MH_ENV:
                train_env, eval_env = reseed_wrapper(update_training_env, env, seed_type)
                policy, data_storage = PPOAlgorithm.train_and_evaluate(configuration, update_training_env, eval_env, training_config, run,  policy, data_storage)
            else:
                policy, data_storage = PPOAlgorithm.train_and_evaluate(configuration, update_training_env, env, training_config, run,  policy, data_storage)
            
            run.finish()

            # Train Causal Discovery
            if data_storage["replay_buffer"] != []:
                wandb_reward_run = main_utils.get_wandb_reward_config(wandb_reward_run)    
                intrinsic_reward, model_dict = discover_subgoal_hierarchy(j, env, data_storage, configuration, model_dict, wandb_reward_run) 


    elif configuration["algorithm"] == "HER":
        assert args.env in R_ENV, "Only test for R environments."
        training_config, run = main_utils.get_training_config(configuration, total_timesteps=total_timesteps)
        policy, _ = HERAlgorithm.train_and_evaluate(configuration, env, env, training_config, run,  data_storage=data_storage)
        run.finish()

    elif configuration["algorithm"] == "HAC":
        assert args.env in R_ENV, "Only test for R environments."
        run = None
        policy = None
        wandb_reward_run = None
        intrinsic_reward = {}
        training_config, run = main_utils.get_training_config(configuration, run=run, total_timesteps=total_timesteps/loop, policy=policy)
        training_config["policy_type"] = "MultiInputPolicy"
        training_env = main_utils.get_update_training_environment(args, configuration, intrinsic_reward, data_storage, model_dict)
    
        policy, data_storage = HACAlgorithm.train_and_evaluate(configuration, env, env, training_config, run,  policy, data_storage)   
        run.finish()
    
    elif configuration["algorithm"] == "HAC_U":
        assert args.env in R_ENV, "Only test for R environments."
        run = None
        policy = None
        wandb_reward_run = None
        intrinsic_reward = {}
        for j in range(0, loop):
            # Train Causal Discovery
            wandb_reward_run = main_utils.get_wandb_reward_config(wandb_reward_run)
            intrinsic_reward, model_dict = discover_subgoal_hierarchy(j, env, data_storage, configuration, model_dict, wandb_reward_run) 
            
            # Train and Evaluate Task 
            training_config, run = main_utils.get_training_config(configuration, run=run, total_timesteps=total_timesteps/loop, policy=policy)
            training_config["policy_type"] = "MultiInputPolicy"
            training_env = main_utils.get_update_training_environment(args, configuration, intrinsic_reward, data_storage, model_dict)
    
            policy, data_storage = HACAlgorithm.train_and_evaluate(configuration, training_env, env, training_config, run,  policy, data_storage)
            
            run.finish()
            exit()
    
    elif configuration["algorithm"] == "HAC_U_loop":
        assert args.env in R_ENV, "Only test for R environments."
        run = None
        policy = None
        wandb_reward_run = None
        intrinsic_reward = {}
        for j in range(1, loop):
            # Train and Evaluate Task 
            training_config, run = main_utils.get_training_config(configuration, run=run, total_timesteps=total_timesteps/loop, policy=policy)
            training_config["policy_type"] = "MultiInputPolicy"
            training_env = main_utils.get_update_training_environment(args, configuration, intrinsic_reward, data_storage, model_dict)
    
            policy, data_storage = HACAlgorithm.train_and_evaluate_loop(j, configuration, training_env, env, training_config, run,  policy, data_storage)

            # Train Causal Discovery
            if data_storage["replay_buffer"] != []:
                wandb_reward_run = main_utils.get_wandb_reward_config(wandb_reward_run)
                intrinsic_reward, model_dict = discover_subgoal_hierarchy(j, env, data_storage, configuration, model_dict, wandb_reward_run) 
            
            run.finish()
            

if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    wandb.tensorboard.patch(root_logdir=f"runs/{train_datetime}/{args.env}")

    env = main_utils.get_based_environment(args=args)
    
    train_function(env=env, args=args)