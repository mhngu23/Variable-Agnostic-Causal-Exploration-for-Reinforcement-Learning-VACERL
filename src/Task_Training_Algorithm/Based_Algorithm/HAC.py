import torch
import numpy as np
from src.Task_Training_Algorithm.Based_Algorithm.DDPG import DDPG
from src.Task_Training_Algorithm.Based_Algorithm.utils import ReplayBuffer
import copy
import csv
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                 action_bounds, action_offset, state_bounds, state_offset, lr, file_name):
        
        # adding lowest level
        self.HAC = [DDPG(state_dim, action_dim, goal_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ReplayBuffer()]
        
        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(DDPG(state_dim, goal_dim, goal_dim, state_bounds, state_offset, lr=lr, H=H))
            self.replay_buffer.append(ReplayBuffer())

        # self.HAC.append(DDPG(state_dim, goal_dim, goal_dim, lr=lr, H=H))
        # self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

        self.file_name_csv = file_name

        
    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise
    
    
    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        return True
    
    
    def run_HAC(self, env, i_level, state, goal, is_subgoal_test, configuration, data_storage, training_mode=True, i_episode=None):
        next_state = None
        done = None
        goal_transitions = []
        
        # logging updates
        self.goals[i_level] = goal
        
        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            
            #   <================ high level policy ================>
            if i_level > 0:
                action = self.HAC[i_level].select_action(state, goal)
                # print(action)

                # add noise or take random action if not subgoal testing
                if training_mode == True:
                    if not is_subgoal_test:
                        if np.random.random_sample() > 0.3:
                            action = action + np.random.normal(0, self.exploration_state_noise)
                            action = action.clip(self.state_clip_low, self.state_clip_high)
                        else:
                            # Testing using HAC_U
                            if configuration["algorithm"] == "HAC_U":
                                # if np.random.random_sample() > 0.3:
                                if np.random.random_sample() > 0.5:
                                # if np.random.random_sample() > 0.7:
                                # if np.random.random_sample() > 0.9:
                                    action = np.random.uniform(self.state_clip_low, self.state_clip_high) 
                                else:
                                    if len(env.causal_subgoals) > 0:
                                        # intrinsic_reward_key = list(env.intrinsic_reward.keys())[1:]
                                        # key = random.choice(intrinsic_reward_key)
                                        key = random.choice(env.causal_subgoals)
                                        action = data_storage["obs_pos_mapping"].get(key) 
                                        action = action.clip(self.state_clip_low, self.state_clip_high)
                                    else:
                                        action = np.random.uniform(self.state_clip_low, self.state_clip_high)        
                            # Sample randomly epsilon
                            else:
                                action = np.random.uniform(self.state_clip_low, self.state_clip_high)

                if i_level == 2:
                    with open(self.file_name_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        
                        # If the file is empty, write the header row
                        if file.tell() == 0:
                            writer.writerow(["episode", "x", "y", "z"])
                        
                        # Write the values from the tuple
                        writer.writerow([i_episode] + list(action))
                    
                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True
                
                # Pass subgoal to lower level 
                next_state, achieved_goal, done, truncated, info = self.run_HAC(env, i_level-1, state, action, is_next_subgoal_test, configuration, data_storage, training_mode, i_episode=i_episode)
                
                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and abs(env.compute_reward(action, achieved_goal, info)) > 0.05:
                    self.replay_buffer[i_level].add((state, action, -self.H, next_state, goal, 0.0, float(done)))
                
                # for hindsight action transition
                action = achieved_goal
                
                print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))

                if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                    self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
                else:
                    self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                
                # copy for goal transition
                goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])
                
                state = next_state
                
            #   <================ low level policy ================>
            else:
                action = self.HAC[i_level].select_action(state, goal)
    
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if training_mode == True:
                        if np.random.random_sample() > 0.2:
                            action = action + np.random.normal(0, self.exploration_action_noise)
                            action = action.clip(self.action_clip_low, self.action_clip_high)
                        else:
                            action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                
               
                # take primitive action
                next_state, rew, done, truncated, info = env.step(action)
                achieved_goal = next_state["achieved_goal"].copy()
                next_state = next_state["observation"].copy()
                
                # if self.render:
                #     # env.render() ##########
                #     if self.k_level == 2:
                #         env.unwrapped.render_goal(self.goals[0], self.goals[1])
                #     elif self.k_level == 3:
                #         env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])
                    
                    
                # this is for logging
                self.reward += rew
                self.timestep += 1
            
            #   <================ finish one step/transition ================>
                print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))
                if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                    self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
                else:
                    self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                # self.replay_buffer[i_level].add((state, action, env.compute_reward(achieved_goal, goal, info), next_state, goal, self.gamma, float(done)))
                
                # copy for goal transition
                goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])

                state = next_state
            
            if done or truncated or abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                break
        
        
        #   <================ finish H attempts ================>
        
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = achieved_goal
            self.replay_buffer[i_level].add(tuple(transition))
            
        return next_state, achieved_goal, done, truncated, info
    
    
    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
    def add_data_to_csv(self,filename, data):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

    def evaluate(self, testing_agent, i_episode, evaluate_env, k_level, filename, configuration, data_storage, directory, name):
        success_ratio_list = []
        for _ in range(50):
            state = evaluate_env.reset(seed=0)[0]
            goal_state = state["desired_goal"].copy()
            achieved_goal = state["achieved_goal"].copy()
            state = state["observation"].copy()            
            next_state, achieved_goal, done, truncated, info = testing_agent.run_HAC(evaluate_env, k_level-1, state, goal_state, False, configuration, data_storage, training_mode=False, i_episode=i_episode) 
            success_ratio_list.append(evaluate_env.compute_reward(achieved_goal, goal_state, info))
        mean_ratio = np.mean(success_ratio_list)
        self.add_data_to_csv(filename, [i_episode, mean_ratio] )

# class HAC_CG:
#     def __init__(self, k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
#                  action_bounds, action_offset, state_bounds, state_offset, lr):
        
#         # adding lowest level
#         self.HAC = [DDPG(state_dim, action_dim, goal_dim, action_bounds, action_offset, lr, H)]
#         self.replay_buffer = [ReplayBuffer()]
        
#         # adding remaining levels
#         for _ in range(k_level-1):
#             self.HAC.append(DDPG(state_dim, goal_dim, goal_dim, state_bounds, state_offset, lr=lr, H=H))
#             self.replay_buffer.append(ReplayBuffer())

#         # self.HAC.append(DDPG(state_dim, goal_dim, goal_dim, lr=lr, H=H))
#         # self.replay_buffer.append(ReplayBuffer())

#         # set some parameters
#         self.k_level = k_level
#         self.H = H
#         self.action_dim = action_dim
#         self.state_dim = state_dim
#         self.threshold = threshold
#         self.render = render
        
#         # logging parameters
#         self.goals = [None]*self.k_level
#         self.reward = 0
#         self.timestep = 0
        
#     def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
#                        state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
#         self.lamda = lamda
#         self.gamma = gamma
#         self.action_clip_low = action_clip_low
#         self.action_clip_high = action_clip_high
#         self.state_clip_low = state_clip_low
#         self.state_clip_high = state_clip_high
#         self.exploration_action_noise = exploration_action_noise
#         self.exploration_state_noise = exploration_state_noise
    
    
#     def check_goal(self, state, goal, threshold):
#         for i in range(self.state_dim):
#             if abs(state[i]-goal[i]) > threshold[i]:
#                 return False
#         return True
    
    
#     def run_HAC_CG(self, env, i_level, state, goal, is_subgoal_test, configuration, data_storage, training_mode=True, causal_graph={}):
#         next_state = None
#         done = None
#         goal_transitions = []
        
#         # logging updates
#         self.goals[i_level] = goal
        
#         # H attempts
#         for _ in range(self.H):
#             # if this is a subgoal test, then next/lower level goal has to be a subgoal test
#             is_next_subgoal_test = is_subgoal_test
            
#             #   <================ high level policy ================>
#             if i_level > 0:
#                 action = self.HAC[i_level].select_action(state, goal)
#                 # print(action)

#                 # add noise or take random action if not subgoal testing
#                 if training_mode == True:
#                     if not is_subgoal_test:
#                         if np.random.random_sample() > 0.3:
#                             action = action + np.random.normal(0, self.exploration_state_noise)
#                             action = action.clip(self.state_clip_low, self.state_clip_high)
#                         else:
#                             # Testing using HAC_U
#                             if configuration["algorithm"] == "HAC_U_CG":
#                                 # if np.random.random_sample() > 0.3:
#                                 if np.random.random_sample() > 0.5:
#                                 # if np.random.random_sample() > 0.7:
#                                 # if np.random.random_sample() > 0.9:
#                                     if tuple(goal) in causal_graph:
#                                         list_of_actions = causal_graph.get(tuple(goal))
#                                         if list_of_actions != []:
#                                             action = random.choice(list_of_actions)
#                                             print("Select based on causal graph", action)
#                                             action = action.clip(self.state_clip_low, self.state_clip_high)
#                                         else:
#                                             action = np.random.uniform(self.state_clip_low, self.state_clip_high)
#                                     else:
#                                         action = np.random.uniform(self.state_clip_low, self.state_clip_high) 
#                                 else:
#                                     action = np.random.uniform(self.state_clip_low, self.state_clip_high) 

#                             # Sample randomly epsilon
#                             else:
#                                 action = np.random.uniform(self.state_clip_low, self.state_clip_high)
       
                    
#                 # Determine whether to test subgoal (action)
#                 if np.random.random_sample() < self.lamda:
#                     is_next_subgoal_test = True
                
#                 # Pass subgoal to lower level 
#                 next_state, achieved_goal, done, truncated, info = self.run_HAC_CG(env, i_level-1, state, action, is_next_subgoal_test, configuration, data_storage, training_mode, causal_graph)
                
#                 # if subgoal was tested but not achieved, add subgoal testing transition
#                 if is_next_subgoal_test and abs(env.compute_reward(action, achieved_goal, info)) > 0.05:
#                     self.replay_buffer[i_level].add((state, action, -self.H, next_state, goal, 0.0, float(done)))
                
#                 # for hindsight action transition
#                 action = achieved_goal
                
#                 print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))

#                 if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
#                     self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
#                 else:
#                     self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                
#                 # copy for goal transition
#                 goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])
                
#                 state = next_state
                
#             #   <================ low level policy ================>
#             else:
#                 action = self.HAC[i_level].select_action(state, goal)
    
#                 # add noise or take random action if not subgoal testing
#                 if not is_subgoal_test:
#                     if training_mode == True:
#                         if np.random.random_sample() > 0.2:
#                             action = action + np.random.normal(0, self.exploration_action_noise)
#                             action = action.clip(self.action_clip_low, self.action_clip_high)
#                         else:
#                             action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                
               
#                 # take primitive action
#                 next_state, rew, done, truncated, info = env.step(action)
#                 achieved_goal = next_state["achieved_goal"].copy()
#                 next_state = next_state["observation"].copy()
                
#                 # if self.render:
#                 #     # env.render() ##########
#                 #     if self.k_level == 2:
#                 #         env.unwrapped.render_goal(self.goals[0], self.goals[1])
#                 #     elif self.k_level == 3:
#                 #         env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])
                    
                    
#                 # this is for logging
#                 self.reward += rew
#                 self.timestep += 1
            
#             #   <================ finish one step/transition ================>
#                 print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))
#                 if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
#                     self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
#                 else:
#                     self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
#                 # self.replay_buffer[i_level].add((state, action, env.compute_reward(achieved_goal, goal, info), next_state, goal, self.gamma, float(done)))
                
#                 # copy for goal transition
#                 goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])

#                 state = next_state
            
#             if done or truncated or abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
#                 break
        
        
#         #   <================ finish H attempts ================>
        
#         # hindsight goal transition
#         # last transition reward and discount is 0
#         goal_transitions[-1][2] = 0.0
#         goal_transitions[-1][5] = 0.0
#         for transition in goal_transitions:
#             # last state is goal for all transitions
#             transition[4] = achieved_goal
#             self.replay_buffer[i_level].add(tuple(transition))
            
#         return next_state, achieved_goal, done, truncated, info
    
    
#     def update(self, n_iter, batch_size):
#         for i in range(self.k_level):
#             self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    
#     def save(self, directory, name):
#         for i in range(self.k_level):
#             self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
#     def load(self, directory, name):
#         for i in range(self.k_level):
#             self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
#     def add_data_to_csv(self,filename, data):
#         with open(filename, 'a', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(data)

#     def evaluate(self, testing_agent, i_episode, evaluate_env, k_level, filename, configuration, data_storage, directory, name, causal_graph):
#         success_ratio_list = []
#         for _ in range(50):
#             state = evaluate_env.reset(seed=0)[0]
#             goal_state = state["desired_goal"].copy()
#             achieved_goal = state["achieved_goal"].copy()
#             state = state["observation"].copy()            
#             next_state, achieved_goal, done, truncated, info = testing_agent.run_HAC_CG(evaluate_env, k_level-1, state, goal_state, False, configuration, data_storage, training_mode=False, causal_graph=causal_graph)  # User-defined policy function
#             success_ratio_list.append(evaluate_env.compute_reward(achieved_goal, goal_state, info))
#         mean_ratio = np.mean(success_ratio_list)
#         self.add_data_to_csv(filename, [i_episode, mean_ratio] )

class HAC_Loop:
    def __init__(self, k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                 action_bounds, action_offset, state_bounds, state_offset, lr, file_name):
        
        # adding lowest level
        self.HAC = [DDPG(state_dim, action_dim, goal_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ReplayBuffer()]
        
        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(DDPG(state_dim, goal_dim, goal_dim, state_bounds, state_offset, lr=lr, H=H))
            self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render

        self.episode = []
        self.obs_pos_mapping = {}

        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

        self.file_name_csv = file_name
        
    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise
    
    def record_episode(self, step):
        self.episode.append(step)
    
    def record_obs_pos_mapping(self, step, achieved_goal):
        key = tuple(torch.tensor(step).detach().numpy())
        self.obs_pos_mapping[key] = achieved_goal
    
    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        return True
    
    def run_HAC_loop(self, env, i_level, state, goal, is_subgoal_test, configuration, data_storage, training_mode=True, i_episode=None):
        next_state = None
        done = None
        goal_transitions = []
        
        # logging updates
        self.goals[i_level] = goal
        
        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            
            #   <================ high level policy ================>
            if i_level > 0:
                action = self.HAC[i_level].select_action(state, goal)
                # print(action)

                # add noise or take random action if not subgoal testing
                if training_mode == True:
                    if not is_subgoal_test:
                        if np.random.random_sample() > 0.3:
                            action = action + np.random.normal(0, self.exploration_state_noise)
                            action = action.clip(self.state_clip_low, self.state_clip_high)
                        else:
                            # Testing using HAC_U
                            if configuration["algorithm"] == "HAC_U_loop":
                                if np.random.random_sample() > 0.5:
                                    action = np.random.uniform(self.state_clip_low, self.state_clip_high) 
                                else:
                                    # if len(env.intrinsic_reward) > 1:
                                    #     intrinsic_reward_key = list(env.intrinsic_reward.keys())[1:]
                                    #     key = random.choice(intrinsic_reward_key)
                                    if len(env.causal_subgoals) > 0:
                                        # intrinsic_reward_key = list(env.intrinsic_reward.keys())[1:]
                                        # key = random.choice(intrinsic_reward_key)
                                        key = random.choice(env.causal_subgoals)
                                        action = data_storage["obs_pos_mapping"].get(key) 
                                        action = action.clip(self.state_clip_low, self.state_clip_high)
                                    else:
                                        action = np.random.uniform(self.state_clip_low, self.state_clip_high)        
                            # Sample randomly epsilon
                            else:
                                action = np.random.uniform(self.state_clip_low, self.state_clip_high)
                    
                if i_level == 2:
                    with open(self.file_name_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        
                        # If the file is empty, write the header row
                        if file.tell() == 0:
                            writer.writerow(["episode", "x", "y", "z"])
                        
                        # Write the values from the tuple
                        writer.writerow([i_episode] + list(action))

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True
                
                # Pass subgoal to lower level 
                next_state, achieved_goal, done, truncated, info, env = self.run_HAC_loop(env, i_level-1, state, action, is_next_subgoal_test, configuration, data_storage, training_mode, i_episode=i_episode)
                
                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and abs(env.compute_reward(action, achieved_goal, info)) > 0.05:
                    self.replay_buffer[i_level].add((state, action, -self.H, next_state, goal, 0.0, float(done)))
                
                # for hindsight action transition
                action = achieved_goal
                
                print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))
                if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                    self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
                else:
                    self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                
                # copy for goal transition
                goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])
                
                state = next_state
                
            #   <================ low level policy ================>
            else:
                action = self.HAC[i_level].select_action(state, goal)
    
                # add noise or take random action if not subgoal testing
                if training_mode == True:
                    if not is_subgoal_test:
                        if np.random.random_sample() > 0.2:
                            action = action + np.random.normal(0, self.exploration_action_noise)
                            action = action.clip(self.action_clip_low, self.action_clip_high)
                        else:
                            action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                
                # take primitive action
                step = np.concatenate((state, action)).tolist()
                if training_mode:
                    self.record_episode(step)

                next_state, rew, done, truncated, info = env.step(action)


                achieved_goal = next_state["achieved_goal"].copy()
                next_state = next_state["observation"].copy()

                if training_mode:
                    self.record_obs_pos_mapping(step, achieved_goal)
                
                # if self.render:
                #     # env.render() ##########
                #     if self.k_level == 2:
                #         env.unwrapped.render_goal(self.goals[0], self.goals[1])
                #     elif self.k_level == 3:
                #         env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])
                    
                    
                # this is for logging
                self.reward += rew
                self.timestep += 1
            
            #   <================ finish one step/transition ================>
                        
            # # hindsight action transition
                print(i_level, abs(env.compute_reward(achieved_goal, goal, info)))
                if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                    self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
                else:
                    self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                # self.replay_buffer[i_level].add((state, action, env.compute_reward(achieved_goal, goal, info), next_state, goal, self.gamma, float(done)))
                
                # copy for goal transition
                goal_transitions.append([state, action, -1.0, next_state, None, self.gamma, float(done)])

                state = next_state
            
            if done or truncated or abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                if training_mode == True:
                    if i_level == 2:
                        if abs(env.compute_reward(achieved_goal, goal, info)) < 0.05:
                            if env.data_storage["replay_buffer"] != []:
                                self.episode[-1] = env.data_storage["replay_buffer"][0][-1]
                            if len(env.data_storage["replay_buffer"]) <= configuration["buffer_size"]: 
                                env.data_storage["replay_buffer"].append(self.episode)
                                env.data_storage["obs_pos_mapping"].update(self.obs_pos_mapping)
                            else:
                                remove_ep = env.data_storage["replay_buffer"].pop(0)
                                env.data_storage["replay_buffer"].append(self.episode)
                                env.data_storage["obs_pos_mapping"].update(self.obs_pos_mapping)
                            self.episode = []
                            self.obs_pos_mapping = {}
                        else:
                            self.episode = []
                            self.obs_pos_mapping = {}
                break
        
        
        #   <================ finish H attempts ================>
        
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = achieved_goal
            self.replay_buffer[i_level].add(tuple(transition))
        
            
        return next_state, achieved_goal, done, truncated, info, env
    
    
    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
    def add_data_to_csv(self,filename, data):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

    def evaluate(self, testing_agent, i_episode, env, k_level, filename, configuration, data_storage, directory, name):
        success_ratio_list = []
        for _ in range(50):
            state = env.reset(seed=0)[0]
            goal_state = state["desired_goal"].copy()
            achieved_goal = state["achieved_goal"].copy()
            state = state["observation"].copy()            
            next_state, achieved_goal, done, truncated, info, _ = testing_agent.run_HAC_loop(env, k_level-1, state, goal_state, False, configuration, data_storage, training_mode=False, i_episode=i_episode)  # User-defined policy function
            success_ratio_list.append(env.compute_reward(achieved_goal, goal_state, info))
        mean_ratio = np.mean(success_ratio_list)
        self.add_data_to_csv(filename, [i_episode, mean_ratio])

