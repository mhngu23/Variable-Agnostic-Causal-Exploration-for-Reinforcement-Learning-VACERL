import pandas as pd
import numpy as np 
import seaborn as sns

import matplotlib.pylab as plt

REVERSE_ACTION = {
        (1, 0, 0, 0, 0, 0, 0):0,
        (0, 1, 0, 0, 0, 0, 0):1,
        (0, 0, 1, 0, 0, 0, 0):2,
        (0, 0, 0, 1, 0, 0, 0):3,
        (0, 0, 0, 0, 1, 0, 0):4,
        (0, 0, 0, 0, 0, 1, 0):5,
        (0, 0, 0, 0, 0, 0, 1):6,
        }

def create_grid_dict(env_name):
    if env_name == "4x4FL":
        lake_dict = {}

        # Define the ranges for each component of the key
        range_x = range(16)
        range_y = range(4)

        # Initialize a counter for the values
        counter = [0, 1, 8, 9]

        # Iterate through all combinations of (x, y)
        for x in range_x:
            if x > 0:
                if x % 4 == 0:
                    counter = [x + 10 for x in counter]
                else:
                    counter = [x + 2 for x in counter]
            for y in range_y:
                # Create the key as a formatted string
                key = f"({x}, {y})"
                
                # Calculate the corresponding value based on the provided pattern
                value = counter[y]
                
                # Add the key-value pair to the dictionary
                lake_dict[key] = value
        return lake_dict
    
    elif env_name == "TaxiEnv-v3":
        # Initialize an empty dictionary
        taxi_dict = {}

        # Define the ranges for each component of the key
        range_x = range(5)
        range_y = range(5)
        range_z = range(6)

        # Initialize a counter for the values
        value_counter = [0, 1, 10, 11, 20, 21]

        # Iterate through all combinations of (x, y, z)
        for x in range_x:
            if x > 0:
                value_counter =  [x + 22 for x in value_counter]
            for y in range_y:
                if y > 0:
                    value_counter =  [x + 2 for x in value_counter]
                for z in range_z:
                    # Create the key as a formatted string
                    key = f"({x}, {y}, {z})"
                    
                    value = value_counter[z]
                    
                    # Add the key-value pair to the dictionary
                    taxi_dict[key] = value             

        return taxi_dict
    
    elif env_name == 'MG_1':
        minigrid_dict = {}

        # Define the ranges for each component of the key
        range_x = range(1,6)
        range_y = range(1,3)
        range_z = range(6)

        # Initialize a counter for the values
        value_counter = [0, 1, 2, 3, 4, 5]

        # Iterate through all combinations of (x, y, z)
        for y in range_y:
            if y > 1:
                value_counter =  [x + 6 for x in value_counter]
            for x in range_x:
                if x > 1:
                    value_counter =  [x + 6 for x in value_counter]
                for z in range_z:
                    # Create the key as a formatted string
                    key = f"({x}, {y}, {z})"
                    
                    value = value_counter[z]
                    
                    # Add the key-value pair to the dictionary
                    minigrid_dict[key] = value   
        return minigrid_dict


def create_heatmap_maze(env, env_name, sorted_dict_attention_weight, number_of_attended_item=20, obs_pos_dictionary = {}, train_datetime = None):
    heatmap_dict = {}

    for item in sorted_dict_attention_weight.keys():
        value = sorted_dict_attention_weight[item][0]
        state = sorted_dict_attention_weight[item][1]
        action =  sorted_dict_attention_weight[item][2]

        grid_dict = create_grid_dict(env_name)

        if env_name == "4x4FL":
            state = grid_dict.get(str((state, action)), 9999)
        
        if env_name == "TaxiEnv-v3":
            taxi_row, taxi_col, _, _ = env.decode(state)
            state = grid_dict.get(str((taxi_row, taxi_col, action)), 9999)

        if env_name == 'MG_1':
            # action = REVERSE_ACTION.get(action)
            [(agent_row, agent_col), action] = obs_pos_dictionary[tuple(np.concatenate((state, action)).tolist())] 
            state = grid_dict.get(str((agent_row, agent_col, action.item())), 9999)

        if state in heatmap_dict.keys():
            heatmap_dict[state] += value
        else:
            heatmap_dict[state] = value

    heatmap_dict[9999] = 0

    heatmap_dict = dict(sorted(heatmap_dict.items(), key=lambda item: item[1], reverse=True)[:8])
    # heatmap_dict = dict(sorted(heatmap_dict.items(), key=lambda item: item[1], reverse=True))
    
    total = np.sum(list(heatmap_dict.values()))

    if env_name == "4x4FL":

        list_value = [[] for _ in range(8)]
        list_index = [[0, 8, 16, 24, 32, 40, 48, 56]]

        for i in range(1, 8):
            indexes = [x + 1 * i for x in list_index[0]]
            list_index.append(indexes)

        for i in range(len(list_value)):
            for item in list_index[i]:
                if item in heatmap_dict.keys():
                    list_value[i].append(heatmap_dict[item]/total)
                else:
                    list_value[i].append(0)

        heatmap_df = pd.DataFrame(data=zip(*list_value), columns=['0', '1', '2', '3', '4', '5', '6', '7'])
        
        heatmap = sns.heatmap(heatmap_df, linewidths=.5, annot=False, fmt='.0%', cmap='RdBu')
        line = [2, 4, 6, 8]
        for value in line:
            heatmap.axhline(y=value, color='black', linewidth=3)   
            heatmap.axvline(x=value, color='black', linewidth=3)
    
    elif env_name == "TaxiEnv-v3":
        list_value = [[] for _ in range(10)]
        list_index = [
           [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
        ]

        for i in range(1, 10):
            indexes = [x + 1 * i for x in list_index[0]]
            list_index.append(indexes)

        for i in range(len(list_value)):
            for item in list_index[i]:
                if item in heatmap_dict.keys():
                    list_value[i].append(heatmap_dict[item]/total)
                else:
                    list_value[i].append(0)
        
        heatmap_df = pd.DataFrame(data=zip(*list_value), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        heatmap = sns.heatmap(heatmap_df, linewidths=.5, annot=True, fmt='.0%', cmap='RdBu')
        line = [3, 6, 9, 12]
        for value in line:
            heatmap.axhline(y=value, color='black', linewidth=3)  
        line = [2, 4, 6, 8]
        for value in line:
            heatmap.axvline(x=value, color='black', linewidth=3)


    elif env_name == 'MG_1':
        list_value = [[] for _ in range(10)]
        list_index = [
           [0, 2, 4, 30, 32, 34],
           [1, 3, 5, 31, 33, 35],
           [6, 8, 10, 36, 38, 40],
           [7, 9, 11, 37, 39, 41],
           [12, 14, 16, 42, 44, 46],
           [13, 15, 17, 43, 45, 47],
           [19, 21, 23, 48, 50, 52],
           [20, 22, 24, 49, 51, 53],
           [25, 27, 29, 54, 56, 58],
           [26, 28, 30, 55, 57, 59],
        ]


        for i in range(len(list_value)):
            for item in list_index[i]:
                if item in heatmap_dict.keys():
                    list_value[i].append(heatmap_dict[item]/total)
                else:
                    list_value[i].append(0)

        heatmap_df = pd.DataFrame(data=zip(*list_value), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


        heatmap = sns.heatmap(heatmap_df, linewidths=.5, annot=False, fmt='.3f', cmap='RdBu')
        line = [3, 6]
        for value in line:
            heatmap.axhline(y=value, color='black', linewidth=3)  
        line = [2, 4, 6, 8]
        for value in line:
            heatmap.axvline(x=value, color='black', linewidth=3)

    plt.savefig(f"Results/ATTENTION_result/{env_name}/heatmap_attention_{train_datetime}.png")
    plt.clf()


