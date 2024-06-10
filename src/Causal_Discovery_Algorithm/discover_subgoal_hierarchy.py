import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

import math
import src.Causal_Discovery_Algorithm.plot_heatmap as plot_heatmap
import src.Causal_Discovery_Algorithm.utils as utils
import src.Causal_Discovery_Algorithm.intrinsic_reward as IR
import src.Causal_Discovery_Algorithm.Attention.transformer_with_attention as transformer_with_attention
import src.Causal_Discovery_Algorithm.AutoEncoder.image_encoder as image_encoder
from src.Causal_Discovery_Algorithm.AutoEncoder.image_encoder import VAE_MG, VAE_MH, image_normalize
from src.Causal_Discovery_Algorithm.Causality.structure_causal_graph import SCM

import copy

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

R_ENV = ["R_1", "R_2", "R_3", "R_4", "R_5", "R_6"]

ACTION = {
        0:[1, 0, 0, 0, 0, 0, 0],
        1:[0, 1, 0, 0, 0, 0, 0],
        2:[0, 0, 1, 0, 0, 0, 0],
        3:[0, 0, 0, 1, 0, 0, 0],
        4:[0, 0, 0, 0, 1, 0, 0],
        5:[0, 0, 0, 0, 0, 1, 0],
        6:[0, 0, 0, 0, 0, 0, 1],
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

def get_action(env, configuration, policy=None, ob=None):
    if policy is None:
        if configuration["env_name"] in FROZEN_LAKE_ENV or configuration["env_name"] in MH_ENV:
            return torch.randint(0, env.action_space.n, (1,)).item()
        elif configuration["env_name"] in R_ENV:
            return np.random.uniform(low=-1.0, high=1.0, size=(4,))
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
    if configuration["env_name"] not in MH_ENV and configuration["env_name"] not in R_ENV:
        ob, _ = env.reset(seed=1)
    else:
        if configuration["env_name"] in MH_ENV:
            ob = env.reset()
            ob = ob["colors"]
        elif configuration["env_name"] in R_ENV:
            ob = env.reset(seed=0)
            achieved_goal = ob[0]["achieved_goal"]
            ob = ob[0]["observation"]

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
        elif configuration["env_name"] in R_ENV:
            ob_before_flatten = copy.deepcopy(ob)
            extend_image_buffer.append(ob_before_flatten)
            next_ob, rew, done, truncated, _ = env.step(action)
            next_achieved_goal = copy.deepcopy(next_ob["achieved_goal"])
            next_ob = next_ob["observation"]

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
        elif configuration["env_name"] in R_ENV:
            step = np.concatenate((ob, action)).tolist()
            key = tuple(torch.tensor(step).detach().numpy())
            # tuple(step.detach().numpy())
            obs_pos_mapping[key] = next_achieved_goal
            if rew == 0 and previous_record_episode is not None:
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
                if configuration["env_name"] not in R_ENV:
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
                    print(rew)
                    if rew == 0:
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

def check_in_S_COAS(S_COAS, data_storage, VAE = None, new_step = None, state_diff_thres = 0.9, env_name = None):
    if env_name in MH_ENV:
        new_ob = np.array(new_step[:1659]).reshape(21,79) 
        new_ob = torch.mean(VAE.encoder(image_normalize(new_ob, env_name)), axis=0)
        for old_step in S_COAS:
            old_ob = np.array(old_step[:1659]).reshape(21,79)
            old_ob = torch.mean(VAE.encoder(image_normalize(old_ob, env_name)), axis=0)
            state_diff = utils.measure_cosine_distance(new_ob.detach().numpy(), old_ob.detach().numpy())  
            if tuple(old_step[-10:]) == tuple(new_step[-10:]) and (state_diff >= state_diff_thres):
                return True, old_step
    elif env_name in R_ENV:
        new_ob = np.array(new_step[:-4])
        for old_step in S_COAS:
            old_ob = np.array(old_step[:-4])
            state_diff = utils.measure_cosine_distance(new_ob, old_ob) 
            action_diff =  utils.measure_cosine_distance(np.array(old_step[-4:]), np.array(new_step[-4:]))
            if (action_diff >= state_diff_thres) and (state_diff >= state_diff_thres):
                return True, old_step
    else:
        new_ob = np.array(new_step[:147]).reshape(7,7,3) 
        new_ob = VAE.encoder(image_normalize(new_ob, env_name))
        for old_step in S_COAS:
            old_ob = np.array(old_step[:147]).reshape(7,7,3)
            old_ob = VAE.encoder(image_normalize(old_ob, env_name))
            state_diff = utils.measure_cosine_distance(new_ob.detach().numpy(), old_ob.detach().numpy())  
            if tuple(old_step[-7:]) == tuple(new_step[-7:]) and (state_diff >= state_diff_thres):
                return True, old_step
    return False, None

def update_buffer_with_attention(index, env, configuration, data_storage, model_dict):
    """
    This function performs ATTENTION mechanism of the method.
    """
    buffer = data_storage["replay_buffer"]

    S_COAS = []
    dict_attention, model_dict = transformer_with_attention.train_attention(buffer, model_dict, configuration["env_name"], configuration["attention_lr"], configuration["train_datetime"], configuration["algorithm"], configuration["number_attended_item"])

    # Sort the dictionary from lowest to highest value according to their value
    dict_attention = dict(sorted(dict_attention.items(), key=lambda item: item[1][0], reverse=True))

    if configuration["env_name"] not in FROZEN_LAKE_ENV and configuration["env_name"] not in R_ENV:
        if configuration["env_name"] in MH_ENV:
            model_VAE = VAE_MH()
        else:
            model_VAE = VAE_MG()
        if model_dict["encoding_model"] is True:
            checkpoint = torch.load(f'model/model_VAE_{configuration["env_name"]}_{configuration["algorithm"]}_{configuration["train_datetime"]}.pth')
            model_VAE.load_state_dict(checkpoint['model_state_dict']) 
    
    # initiate S_COAS
    for step in (dict_attention.keys()):
        if len(S_COAS) < configuration["number_attended_item"]:
            if configuration["env_name"] in FROZEN_LAKE_ENV: 
                S_COAS.insert(0, step) # The lower attention step will be added to the front of S_COAS
            else:
                if configuration["env_name"] not in R_ENV:
                    in_S_COAS, _ = check_in_S_COAS(S_COAS, data_storage, model_VAE, step, state_diff_thres = configuration["state_diff_attention"], env_name = configuration["env_name"])
                else:
                    in_S_COAS, _ = check_in_S_COAS(S_COAS, data_storage, None, step, state_diff_thres = configuration["state_diff_attention"], env_name = configuration["env_name"])
                if in_S_COAS is True:
                    continue
                else:
                    S_COAS.insert(0, step) # The lower attention step will be added to the front of S_COAS
        else:
            break
    
    dict_attention = {key: value for key, value in dict_attention.items() if key in S_COAS}

    filtered_buffer = []

    for episode in buffer:
        new_episode = []
        for step in episode:
            if configuration["env_name"] in FROZEN_LAKE_ENV:
                if  step in S_COAS:
                    new_episode.append(step)
            elif configuration["env_name"] in R_ENV:
                in_SCOAS, rep_step = check_in_S_COAS(S_COAS, data_storage, None, step, state_diff_thres = configuration["state_diff_attention"], env_name = configuration["env_name"])
                if in_SCOAS:
                    new_episode.append(rep_step)
            else:
                if  tuple(step) in S_COAS:   
                    new_episode.append(step)             
        new_episode.append(episode[-1]) 

        # Add the goal state as the last state in the ranking, if it has not already been in the ranking
        if step not in S_COAS:
            S_COAS.append(step) 

        filtered_buffer.append(new_episode)
    
    utils.write_output_attention(index, configuration, S_COAS, data_storage, dict_attention)

    return filtered_buffer, S_COAS, model_dict, dict_attention

def create_dataset_causal_discovery(filtered_buffer, S_COAS):
    """
    This function returns the dataset that will be used for causal discovery
    """
    dataset = [[], []]
    
    S_COAS = [tuple(step) for step in S_COAS]
    
    for episode in filtered_buffer:
        i = 1
        while i < len(episode):
            X = []
            for step in episode[:i]:
                X.append(step)
            dataset[0].append(X)
            dataset[1].append(episode[i])
            i += 1

    return dataset, S_COAS

def discover_causality(configuration, dataset, S_COAS, wand_reward_run):
    """
    This function performs causal discovery technique of the method, including update f and s parameters alternatively
    """
    train_set, test_set = utils.split_dataset(dataset, ratio = 0.9)

    number_of_objects = len(S_COAS)

    scm = SCM(configuration, number_of_objects, S_COAS, DEVICE)

    loss_list_f, accuracy_f, loss_list_s, accuracy_s = [], [], [], []

    for i in range(configuration["alteration_index"]):
        print(f"This is the {i} iteration")
        scm.train_f(configuration["env_name"], train_set[0], train_set[1], loss_list_f, accuracy_f, configuration["batch_size"], wand_reward_run) 
        scm.train_s(configuration["env_name"], train_set[0], train_set[1], loss_list_s, accuracy_s, configuration["batch_size"],  wand_reward_run) 

    wand_reward_run.finish()

    edge_params_sigmoid = torch.sigmoid(scm.best_s_param.edge_params).detach()
    # Ensure edge can only go in one direction
    for i in range(len(edge_params_sigmoid)):
        for j in range(len(edge_params_sigmoid)):
            if edge_params_sigmoid[i][j] > edge_params_sigmoid[j][i]:
                edge_params_sigmoid[j][i] = 0
            else:
                edge_params_sigmoid[i][j] = 0

    if configuration["env_name"] in MH_ENV:
        causality_threshold = 0.5
    if configuration["env_name"] in R_ENV:
        causality_threshold = 0.6
    else:
        causality_threshold = 0.7

    edge_params_sigmoid_after = torch.where(edge_params_sigmoid > causality_threshold, torch.tensor(1), torch.tensor(0))
    
    return edge_params_sigmoid.detach().tolist(), edge_params_sigmoid_after.detach().tolist()

def graph_pruning(edge_params, attention_list):
    causal_graph = {}

    for i in range(len(attention_list)):
        node = attention_list[i]
        
        connected_vertices = []
        for j in range(len(edge_params[i])):
            if edge_params[i][j] == 1:
                connected_vertices.append(attention_list[j])

        causal_graph[node] = connected_vertices
      
    return causal_graph

def discover_subgoal_hierarchy(index, env, data_storage, configuration, model_dict, wand_reward_run = None, policy=None):
    """
    This function performs the causal discovery algorithm.
    """
    # SAMPLING BLOCK
    previous_record_episode = None
    step = 0
    episode_number = 0
    if index == 0:
        while step < int(configuration["head_timestep"]):
            # if configuration["env_name"] not in R_ENV:
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
    
    # VAE BLOCK
    if configuration["env_name"] not in FROZEN_LAKE_ENV and configuration["env_name"] not in R_ENV:
        print(f"\n--------------------------Begin training VAE-----------------------------------------------")
        model_dict = image_encoder.train_encoder(model_dict, data_storage["image_encoding_buffer"], configuration["env_name"],  configuration["algorithm"], configuration["train_datetime"])

    # ATTENTION BLOCK
    print(f"\n--------------------------Begin training ATTENTION-----------------------------------------------")
    filtered_buffer, S_COAS, model_dict, dict_attention = update_buffer_with_attention(index, env, configuration, data_storage, model_dict)
    print(f"This is the length of replay buffer {len(filtered_buffer)}")

    if configuration["algorithm"] == "ATTENTION":
        intrinsic_reward = IR.create_intrinsic_reward_attention_only(configuration, dict_attention)
        wand_reward_run.finish()
        return intrinsic_reward, model_dict
    
    # CAUSAL DISCOVERY BLOCK
    print(f"\n--------------------------Begin training causality discovery-----------------------------------------------")
    dataset_causal_discovery, S_COAS = create_dataset_causal_discovery(filtered_buffer, S_COAS)
    edge_params_before, edge_params_after = discover_causality(configuration, dataset_causal_discovery, S_COAS, wand_reward_run)
    
    causal_graph = graph_pruning(edge_params_after, S_COAS)
    intrinsic_reward = IR.create_intrinsic_reward(configuration, causal_graph)

    if configuration["algorithm"] == "HAC_U_loop_CG" or configuration["algorithm"] == "HAC_U_CG":
        causal_graph_obs_pos = {}
        keys = list(causal_graph.keys())
        for key in keys:
            connected_vertices = []
            node = data_storage["obs_pos_mapping"].get(key)
            items = causal_graph.get(key) 
            for item in items:
                connected_vertices.append(data_storage["obs_pos_mapping"].get(item))
            try:
                causal_graph_obs_pos[tuple(node)] = connected_vertices 
            except:
                continue

    utils.write_output_causal_discovery(index, configuration, edge_params_before, edge_params_after, causal_graph, intrinsic_reward, data_storage)

    if configuration["env_name"] not in R_ENV:
        c = {}
    data_storage["image_mapping"] = {}

    if configuration["algorithm"] == "HAC_U_loop_CG" or configuration["algorithm"] == "HAC_U_CG":
        return intrinsic_reward, model_dict, causal_graph_obs_pos
    else:
        return intrinsic_reward, model_dict





    
    