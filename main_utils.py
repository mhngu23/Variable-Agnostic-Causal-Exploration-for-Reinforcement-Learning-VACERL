import wandb
import gymnasium as gymnasium
import gym as gym2

from src.Environments.Based.MG_1 import *
from src.Environments.Based.MG_2 import *
from src.Environments.Based.MG_3 import *
from src.Environments.Based.MG_4 import *
from src.Environments.Based.MG_5 import *
from src.Environments.Wrapper.EnvWrapper import *
from src.Environments.Wrapper.NewFrozenLakeEnv import *
from minigrid.wrappers import ImgObsWrapper

VECTORIZED_ENV = ["4x4FL", "8x8FL", "MG_1_flatten", "MG_1_flatten_noisy_TV", "MG_3_flatten"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

R_ENV = ["R_1", "R_2", "R_3", "R_4", "R_5", "R_6"]

def add_argument(parser):
    parser.add_argument("--env", type=str, help="The name of the environment to run your algorithm on.",
    choices=["4x4FL", "8x8FL", 
             "MG_1", "MG_1_noisy_TV", "MG_1_flatten", "MG_1_flatten_noisy_TV", "MG_2", "MG_2_noisy_TV", "MG_2_noisy_TV_v2", "MG_3", "MG_3_flatten", "MG_4", "MG_4_noisyTV", "MG_5", 
             "MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10",
             "R_1", "R_2", "R_3", "R_4", "R_5", "R_6"],

    default="R_1",
    )

    parser.add_argument(
        "--render-mode", "-r", type=str,
        help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
        choices=["human", "ansi", "rgb_array"],
        default="rgb_array",
    )

    parser.add_argument("--algorithm", "-a", type=str,
        help="The type of algorithm that you will be using.",
        choices=["Based", 
                 "Count_Position", "Count_Based_Position_Action", "Count_Based_Observation_Action", "Count_Based_Observation_Action_Hash", 
                 "cid_in_rl", "ATTENTION", "Updated", "HER", "HER_Updated", "HAC", "HAC_U", "HAC_U_loop", "HAC_U_CG", "HAC_U_loop_CG"],

        default="HAC_U",
    )

    parser.add_argument("--seed_type", "-s", type=str,
        help="The seed type of the environment.",
        choices=["multi", "one", "unlimited"],
        default="multi",
    )

    parser.add_argument("--buffer-size-causal-discovery", "-bs", type=int,
        help="The size of the buffer, storing episode that will be used for all training causal discovery.",
        default=100,
    )

    parser.add_argument("--number_attention", "-na", type=int,
        help="The top number of attended items that we will keep.",
        default=8,
    )

    parser.add_argument("--state_diff_attention", type=float,
        help="The measurement threshold of diff between two states",
        default=0.9,
    )
    
    parser.add_argument("--state_diff_policy", type=float,
        help="The measurement threshold of diff between two states",
        default=0.95,
    )

    parser.add_argument("--attention-lr", "-attention_lr", type=float,
        help="The learning rate for attention training.",
        default=0.001,
    )

    parser.add_argument("--f-lr", "-f_lr", type=float,
        help="The learning rate for training functional parameter",
        default=0.0005,
    )

    parser.add_argument("--s-lr", "-s_lr", type=float,
        help="The learning rate for training structural parameter.",
        default=0.0005,
    )

    parser.add_argument("--batch-size", "-batch_size", type=int,
        help="The batch size used for training f and s",
        default=256,
    )

    parser.add_argument("--alteration-index", type=int,
        help="The number of time switching between training f and s",
        default=2,
    )

    parser.add_argument("--training_f", type=int,
        help="The number of time training f",
        default=60,
    )

    parser.add_argument("--training_s", type=int,
        help="The number of time training s",
        default=60,
    )

    parser.add_argument("--reward", "-reward", type=float,
        help="The reward disminishing parameter.",
        default=0.0001,
    )

    parser.add_argument("--total_timestep", type=int,
        help="The number of time_step for training task algortihm",
        default=100000,
    )

    parser.add_argument("--head_timestep", type=int,
    help="The number of headstart timestep",
        default=5000,
    )

    parser.add_argument("--loop", type=int,
        help="The number of loop for the updated between causal discovery and downstream task",
        default=50,
    )

    parser.add_argument("--tuning", type=bool,
    help="If tuning is true program will enter grid search hyperparameter tuning mode",
        default=False,
    )

    return parser

def get_main_configuration(args, train_datetime):
    # configuraiton from arguement
    configuration = {
            "algorithm" : args.algorithm,
            "env_name" : args.env,
            "buffer_size" : args.buffer_size_causal_discovery,
            "number_attended_item" : args.number_attention,
            "attention_lr" : args.attention_lr,
            "state_diff_attention": args.state_diff_attention,
            "state_diff_policy": args.state_diff_policy,
            "f_lr" : args.f_lr,
            "s_lr" : args.s_lr,
            "batch_size" : args.batch_size,
            "alteration_index" : args.alteration_index,
            "training_f" : args.training_f,
            "training_s" : args.training_s,
            "train_datetime": train_datetime,
            "root_logdir" : f"runs/{train_datetime}/{args.env}",
            "reward" : args.reward,
            "total_timestep": args.total_timestep,
            "head_timestep": args.head_timestep,
            "loop": args.loop
    }

    return configuration


def get_wandb_reward_config(run):
    if run is not None:
        run = wandb.init(
        project="Causal-discovery", 
        id = run.id,
        resume="must",
        )
            
    else:   
        run = wandb.init(
        project="Causal-discovery" 
        )
    return run 

def get_training_config(configuration, run=None, total_timesteps=10000, policy=None, tb_log_name = "algo_1"):
    total_timesteps = int(total_timesteps)
    reset_num_timesteps = policy is None

    if configuration["env_name"] in VECTORIZED_ENV: 
        config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "env_name": configuration["env_name"],
        "reset_num_timesteps":reset_num_timesteps,
        "tb_log_name":tb_log_name
        }

    elif configuration["env_name"] in MH_ENV or configuration["env_name"] in R_ENV:
        config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": total_timesteps,
        "env_name": configuration["env_name"],
        "reset_num_timesteps":reset_num_timesteps,
        "tb_log_name":tb_log_name
        }

    else:
        config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": total_timesteps,
        "env_name": configuration["env_name"],
        "reset_num_timesteps":reset_num_timesteps,
        "tb_log_name":tb_log_name
        }   
    
    
    if run is not None:
        run = wandb.init(
        project="Model-Training",
        id = run.id,
        resume="must",
        config=config,
        sync_tensorboard=True, 
        save_code=True, 
        )
        
    else:   
        run = wandb.init(
        project="Model-Training",
        config=config,
        sync_tensorboard=True, 
        save_code=True, 
        )

    return config, run

def get_based_environment(args):
    if args.env == "4x4FL":
        env = gymnasium.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode = args.render_mode)

    elif args.env == "8x8FL":
        env = gymnasium.make('FrozenLake-v1', max_episode_steps=2000, desc=None, map_name="8x8", is_slippery=False, render_mode = args.render_mode)
    
    elif args.env == "MG_1":
        env = ImgObsWrapper(MG_1(render_mode = args.render_mode, max_steps=500))

    elif args.env == "MG_1_flatten":
        env = FlatObsWrapper_ImageOnly(MG_1(render_mode = args.render_mode, max_steps=500))

    elif args.env == "MG_1_noisy_TV":
        env = ImgObsWrapper(MG_1_noisy_TV(render_mode = args.render_mode, max_steps=500))
    
    elif args.env == "MG_1_flatten_noisy_TV":
        env = FlatObsWrapper_ImageOnly(MG_1_noisy_TV(render_mode = args.render_mode, max_steps=500))  
    
    elif args.env == "MG_2":
        env = ImgObsWrapper(MG_2(render_mode = args.render_mode, max_steps=1000))
    
    elif args.env == "MG_2_noisy_TV":
        env = ImgObsWrapper(MG_2_noisy_TV(render_mode = args.render_mode, max_steps=2500))

    elif args.env == "MG_2_noisy_TV_v2":
        env = ImgObsWrapper(MG_2_noisy_TV_v2(render_mode = args.render_mode, max_steps=2500))

    elif args.env == "MG_3":
        env = ImgObsWrapper(MG_3(render_mode = args.render_mode, max_steps=5000))

    elif args.env == "MG_3_flatten":
        env = FlatObsWrapper_ImageOnly(MG_3(render_mode = args.render_mode, max_steps=5000))    

    elif args.env == "MG_4":
        env = ImgObsWrapper(MG_4(render_mode = args.render_mode))
    
    elif args.env == "MG_4_noisyTV":
        env = ImgObsWrapper(MG_4_NoisyTV(render_mode = args.render_mode))
    
    elif args.env == "MG_5":
        env = ImgObsWrapper(MG_5(render_mode = args.render_mode, num_rows=1, room_size=3))
    
    elif args.env == "MH_1":
        env = gym2.make("MiniHack-Room-5x5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)
    
    elif args.env == "MH_2": 
        # Not Yet Learnable
        env = gym2.make("MiniHack-MazeWalk-9x9-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)

    elif args.env == "MH_3":
        env = gym2.make("MiniHack-Room-Ultimate-5x5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)

    elif args.env == "MH_4":
        env = gym2.make("MiniHack-Corridor-R3-v0", observation_keys=("colors",), penalty_step=0)
    
    elif args.env == "MH_5":
        env = gym2.make("MiniHack-Room-Ultimate-15x15-v0", observation_keys=("colors",), penalty_step=0)
    
    elif args.env == "MH_6":
        env = gym2.make("MiniHack-Corridor-R5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)
    
    elif args.env == "MH_7":
        env = gym2.make("MiniHack-Corridor-R2-v0", observation_keys=("colors",), penalty_step=0)
    
    elif args.env == "MH_8":
        env = gym2.make("MiniHack-Room-Monster-5x5-v0", observation_keys=("colors",), penalty_step=0)
    
    elif args.env == "MH_9":
        env = gym2.make("MiniHack-Room-Monster-15x15-v0", observation_keys=("colors",), penalty_step=0)
    
    elif args.env == "MH_10":
        env = gym2.make("MiniHack-River-Narrow-v0", observation_keys=("colors",), penalty_step=0)

    elif args.env == "R_1":
        env = gymnasium.make('FetchReach-v2', max_episode_steps=100)
    
    elif args.env == "R_2":
        env = gymnasium.make('FetchReachDense-v2', max_episode_steps=100)

    elif args.env == "R_3":
        env = gymnasium.make('FetchPickAndPlace-v2', max_episode_steps=100)
    
    return env


def get_update_training_environment(args, configuration, intrinsic_reward, data_storage, model_dict):
    env_name = configuration["env_name"]
    state_diff = configuration["state_diff_policy"]
    if env_name == "4x4FL":
        update_training_env = UpdatedFrozenLakeEnv(desc=None, map_name="4x4", is_slippery=False, render_mode = args.render_mode, intrinsic_reward=intrinsic_reward)
        update_training_env = gymnasium.wrappers.TimeLimit(update_training_env, 100)

    elif env_name == "8x8FL":
        update_training_env = UpdatedFrozenLakeEnv(desc=None, map_name="8x8", is_slippery=False, render_mode = args.render_mode, intrinsic_reward=intrinsic_reward)
        update_training_env = gymnasium.wrappers.TimeLimit(update_training_env, 2000)

    elif env_name == "MG_1":
        update_training_env = ImgObsWrapper((MG_1(render_mode = args.render_mode, max_steps=500)))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MG_1_flatten":
        update_training_env = FlatObsWrapper_ImageOnly(MG_1(render_mode = args.render_mode, max_steps=500))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif env_name == "MG_1_noisy_TV":
        update_training_env = ImgObsWrapper((MG_1_noisy_TV(render_mode = args.render_mode, max_steps=500)))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MG_1_flatten_noisy_TV":
        update_training_env = FlatObsWrapper_ImageOnly(MG_1_noisy_TV(render_mode = args.render_mode, max_steps=500)) 
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif env_name == "MG_2":
        update_training_env = ImgObsWrapper((MG_2(render_mode = args.render_mode, max_steps=1000)))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MG_2_noisy_TV":
        update_training_env = ImgObsWrapper(MG_2_noisy_TV(render_mode = args.render_mode, max_steps=2500))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MG_2_noisy_TV_v2":
        update_training_env = ImgObsWrapper(MG_2_noisy_TV_v2(render_mode = args.render_mode, max_steps=2500))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif env_name == "MG_3":
        update_training_env = ImgObsWrapper((MG_3(render_mode = args.render_mode, max_steps=5000)))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MG_3_flatten":
        update_training_env = FlatObsWrapper_ImageOnly(MG_3(render_mode = args.render_mode, max_steps=5000))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MG_4":
        update_training_env = ImgObsWrapper(MG_4(render_mode = args.render_mode))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MG_4_noisyTV":
        update_training_env = ImgObsWrapper(MG_4_NoisyTV(render_mode = args.render_mode))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif env_name == "MG_5":
        update_training_env = ImgObsWrapper((MG_5(render_mode = args.render_mode, num_rows=1, room_size=3)))
        update_training_env = CausalBonus(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MH_1":
        update_training_env = gym2.make("MiniHack-Room-5x5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_2":
        update_training_env = gym2.make("MiniHack-MazeWalk-9x9-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_3":
        update_training_env = gym2.make("MiniHack-Room-Ultimate-5x5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MH_4":
        update_training_env = gym2.make("MiniHack-Corridor-R3-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MH_5":
        update_training_env = gym2.make("MiniHack-Room-Ultimate-15x15-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_6":
        update_training_env = gym2.make("MiniHack-Corridor-R5-v0", observation_keys=("colors",), max_episode_steps=500, penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_7":
        update_training_env = gym2.make("MiniHack-Corridor-R2-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_8":
        update_training_env = gym2.make("MiniHack-Room-Monster-5x5-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "MH_9":
        update_training_env = gym2.make("MiniHack-Room-Monster-5x5-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "MH_10":
        update_training_env = gym2.make("MiniHack-River-Narrow-v0", observation_keys=("colors",), penalty_step=0)
        update_training_env = CausalBonusMH(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)
    
    elif args.env == "R_1":
        update_training_env = gymnasium.make('FetchReach-v2', max_episode_steps=100)
        update_training_env = CausalBonusR(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "R_2":
        update_training_env = gymnasium.make('FetchReachDense-v2', max_episode_steps=100)
        update_training_env = CausalBonusR(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    elif args.env == "R_3":
        update_training_env = gymnasium.make('FetchPickAndPlace-v2', max_episode_steps=100)
        update_training_env = CausalBonusR(update_training_env, intrinsic_reward, state_diff, configuration, model_dict, data_storage)

    return update_training_env
    