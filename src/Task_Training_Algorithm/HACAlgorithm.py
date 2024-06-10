import os

import torch
import gym
import numpy as np
from src.Task_Training_Algorithm.Based_Algorithm.HAC import HAC, HAC_Loop
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(configuration, train_env, evaluate_env, training_config, run, policy = None, data_storage = {}):
    # env_name = "MountainCarContinuous-v0"
    
    save_episode = 10               # keep saving every n episodes
    # 20,000,000
    max_episodes = 100000             # max num of training episodes
    random_seed = 0
    render = False
    
    # env = gym.make(env_name)
    env = train_env
    state_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = np.array([env.action_space.high[0], env.action_space.high[0], env.action_space.high[0], env.action_space.high[0]])
    action_bounds = torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = np.array([0.0, 0.0, 0.0, 0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([-1.0, -1.0, -1.0, -1.0])
    action_clip_high = np.array([1.0, 1.0, 1.0, 1.0])
    
    # state bounds and offset
    state_bounds_np = np.array([2, 2, 2])
    # state_bounds_np = np.array([1.5, 0.9, 0.7])

    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  np.array([0.0, 0.0, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    # state_clip_low = np.array([1.2, 0.5, 0.3])
    state_clip_low = np.array([-2, -2, -2])
    state_clip_high = np.array([2, 2, 2])
    # state_clip_high = np.array([1.5, 0.9, 0.7])
    
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.002, 0.002, 0.002, 0.002])        
    exploration_state_noise = np.array([0.0001, 0.0001, 0.0001]) 
    
    threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 3                # num of levels in hierarchy
    H = 20                      # time horizon to achieve subgoal
    lamda = 0.3                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    filename_subgoal = "log_subgoal_{}_{}_{}.csv".format(configuration['env_name'], configuration["algorithm"], configuration['train_datetime'])
   
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr, filename_subgoal)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    # logging file:
    log_f = open("log.txt","w+")

    directory = "./preTrained/{}/{}/{}level/".format(configuration["env_name"], configuration["algorithm"], k_level) 
    name = "HAC_{}_{}".format(configuration["algorithm"], configuration["train_datetime"])

    filename = "log_{}_{}_{}.csv".format(configuration['env_name'], configuration["algorithm"], configuration['train_datetime'])
    headers = ['Step', 'Success_Ratio'] 
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    
    # training procedure 
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        
        state = env.reset(seed=0)[0]
        goal_state = state["desired_goal"].copy()
        achieved_goal = state["achieved_goal"].copy()
        state = state["observation"].copy()

        # collecting experience in environment
        next_state, achieved_goal, done, truncated, info = agent.run_HAC(env, k_level-1, state, goal_state, False, configuration, data_storage, training_mode=True, i_episode=i_episode)
        
        # print(env.compute_reward(achieved_goal, goal_state, info))
        if abs(env.compute_reward(achieved_goal, goal_state, info)) < 0.05:
            print("################ Solved! ################ ")
            agent.save(directory, name)
        
        # update all levels
        agent.update(n_iter, batch_size)
        
        if i_episode % 2000 == 0:  
            testing_agent = HAC(k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr, filename_subgoal)
            
            testing_agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                        state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
            
            try:
                testing_agent.load(directory, name)
            except:
                testing_agent = agent

            agent.evaluate(testing_agent, i_episode, evaluate_env, k_level, filename, configuration, data_storage, directory, name)

        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))

def train_and_evaluate_loop(j, configuration, train_env, evaluate_env, training_config, run, policy = None, data_storage = {}):
    # env_name = "MountainCarContinuous-v0"
    
    save_episode = 10               # keep saving every n episodes
    max_episodes = training_config["total_timesteps"]             # max num of training episodes
    random_seed = 0
    render = False
    
    # env = gym.make(env_name)
    env = train_env
    state_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = np.array([env.action_space.high[0], env.action_space.high[0], env.action_space.high[0], env.action_space.high[0]])
    action_bounds = torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = np.array([0.0, 0.0, 0.0, 0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([-1.0, -1.0, -1.0, -1.0])
    action_clip_high = np.array([1.0, 1.0, 1.0, 1.0])
    
    # state bounds and offset
    state_bounds_np = np.array([2, 2, 2])
    # state_bounds_np = np.array([1.5, 0.9, 0.7])

    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  np.array([0.0, 0.0, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    # state_clip_low = np.array([1.2, 0.5, 0.3])
    state_clip_low = np.array([-2, -2, -2])
    state_clip_high = np.array([2, 2, 2])
    # state_clip_high = np.array([1.5, 0.9, 0.7])
    
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.002, 0.002, 0.002, 0.002])        
    exploration_state_noise = np.array([0.0001, 0.0001, 0.0001]) 
    
    threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 3                # num of levels in hierarchy
    H = 20                      # time horizon to achieve subgoal
    lamda = 0.3                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    
    filename_subgoal = "log_subgoal_{}_{}_{}.csv".format(configuration['env_name'], configuration["algorithm"], configuration['train_datetime'])
 
    # creating HAC agent and setting parameters
    if policy is None:
        agent = HAC_Loop(k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                    action_bounds, action_offset, state_bounds, state_offset, lr, filename_subgoal)
        
        agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                        state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    else:
        agent = policy
    
    # logging file:
    log_f = open("log.txt","w+")

    filename = "log_{}_{}_{}.csv".format(configuration['env_name'], configuration["algorithm"], configuration['train_datetime'])
    headers = ['Step', 'Success_Ratio'] 
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    t = int(max_episodes * (j-1) + j)
    end = int(max_episodes + t) 

    directory = "./preTrained/{}/{}/{}level/".format(configuration["env_name"], configuration["algorithm"], k_level) 
    name = "HAC_{}_{}".format(configuration["algorithm"], configuration["train_datetime"])
    
    # training procedure 
    for i_episode in range(t, end):
        agent.reward = 0
        agent.timestep = 0
        
        state = train_env.reset(seed=0)[0]
        goal_state = state["desired_goal"].copy()
        achieved_goal = state["achieved_goal"].copy()
        state = state["observation"].copy()

        # collecting experience in environment
        next_state, achieved_goal, done, truncated, info, train_env = agent.run_HAC_loop(train_env, k_level-1, state, goal_state, False, configuration, data_storage, training_mode=True, i_episode=i_episode)
        
        # print(env.compute_reward(achieved_goal, goal_state, info))
        if abs(env.compute_reward(achieved_goal, goal_state, info)) < 0.05:
            print("################ Solved! ################ ")
            agent.save(directory, name)
        
        # agent.save(directory, name)
        
        # update all levels
        agent.update(n_iter, batch_size)
        
        if i_episode % 2000 ==0:
            testing_agent = HAC_Loop(k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr, filename_subgoal)

            testing_agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                        state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
            
            try:
                testing_agent.load(directory, name)
            except:
                testing_agent = agent

            agent.evaluate(testing_agent, i_episode, evaluate_env, k_level, filename, configuration, data_storage, directory, name)
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))

    return agent, train_env.data_storage




        