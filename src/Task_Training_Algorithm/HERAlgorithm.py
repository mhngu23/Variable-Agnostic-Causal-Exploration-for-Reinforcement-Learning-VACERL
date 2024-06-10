from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from src.Task_Training_Algorithm.policy_utils import get_callback, get_policy_kwargs

VECTORIZED_ENV = ["4x4FL", "8x8FL","4x4FL_noisy_TV", "8x8FL_noisy_TV", "MG_1_flatten", "MG_1_flatten_noisy_TV", "MG_3_flatten"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

R_ENV = ["R_1", "R_2", "R_3", "R_4", "R_5", "R_6"]


def train_and_evaluate(configuration, train_env, evaluate_env, training_config, run, policy = None, data_storage = {}):
	algo_name = "HER"
	print(training_config["policy_type"])
	
	model_class = DDPG  # works also with SAC, DDPG and TD3

	goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

	Wandb_Callback, Eval_Callback = get_callback(configuration, evaluate_env, algo_name, data_storage)

	policy = model_class(training_config["policy_type"], env=train_env, tau=0.05, batch_size=1024, learning_rate=0.001, gamma=0.95,   policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]), replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,),verbose=1, tensorboard_log=configuration["root_logdir"])

	if configuration["algorithm"] == "Updated" or configuration["algorithm"] == "ATTENTION":
		policy.learn(
			total_timesteps=training_config["total_timesteps"],
			callback=[Wandb_Callback, Eval_Callback],
			tb_log_name=training_config["tb_log_name"],
			reset_num_timesteps=training_config["reset_num_timesteps"]
		)
	else:
		policy.learn(
			total_timesteps=training_config["total_timesteps"],
			callback=[Wandb_Callback, Eval_Callback],
			tb_log_name=training_config["tb_log_name"],
			reset_num_timesteps=training_config["reset_num_timesteps"]
		)
		
	
	return policy, train_env.data_storage 
	