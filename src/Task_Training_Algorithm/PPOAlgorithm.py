from stable_baselines3 import PPO
from src.Task_Training_Algorithm.policy_utils import get_callback, get_policy_kwargs

VECTORIZED_ENV = ["4x4FL", "8x8FL","4x4FL_noisy_TV", "8x8FL_noisy_TV", "MG_1_flatten", "MG_1_flatten_noisy_TV", "MG_3_flatten"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

R_ENV = ["R_1", "R_2", "R_3", "R_4", "R_5", "R_6"]


def train_and_evaluate(configuration, train_env, evaluate_env, training_config, run, policy = None, data_storage = {}):
	algo_name = "PPO"
	print(training_config["policy_type"])

	Wandb_Callback, Eval_Callback = get_callback(configuration, evaluate_env, algo_name, data_storage)

	if policy is None and (configuration["env_name"] in VECTORIZED_ENV or configuration["env_name"] in MH_ENV or configuration["env_name"] in R_ENV):
		policy = PPO(policy=training_config["policy_type"], env=train_env, verbose=1, tensorboard_log=configuration["root_logdir"], ent_coef=0.005)
	if policy is None and configuration["env_name"] not in VECTORIZED_ENV:
		policy = PPO(policy=training_config["policy_type"], env=train_env, verbose=1, policy_kwargs=get_policy_kwargs(), tensorboard_log=configuration["root_logdir"], ent_coef=0.005)
	
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
	