import pandas as pd
import pickle
import cv2
import matplotlib.pylab as plt

from numpy import dot
from numpy.linalg import norm

ACTION = {  (1, 0, 0, 0, 0, 0, 0):"L",
            (0, 1, 0, 0, 0, 0, 0):"R",
            (0, 0, 1, 0, 0, 0, 0):"F",
            (0, 0, 0, 1, 0, 0, 0):"PU",
            (0, 0, 0, 0, 1, 0, 0):"Drop",
            (0, 0, 0, 0, 0, 1, 0):"Toogle",
            (0, 0, 0, 0, 0, 0, 1):"Done",
          }

BLOCK_UNLOCK_ENV = ["MG_1", "MG_2", "MG_3", "MG_5"]

R_ENV = ["R_1", "R_2", "R_3"]

def show_result(change_in_loss_function = None, change_in_accuracy = None, file_name = None):
    """
    change_in_loss_function (list): a list of loss value
    """
    if change_in_loss_function is not None and change_in_accuracy is not None:
        df = pd.DataFrame({'step': [i for i in range(len(change_in_loss_function))],
                    'change_in_loss_function': change_in_loss_function})
        df_2 = pd.DataFrame({'step': [i for i in range(len(change_in_accuracy))],
                'change_in_accuracy': change_in_accuracy})
            
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(df["step"].iloc[0:], df["change_in_loss_function"].iloc[0:], c = "orange", label='Change in loss function after each step', linewidth=3)
        plt.title('Loss Graph')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(df_2["step"].iloc[0:], df_2["change_in_accuracy"].iloc[0:], c = "blue", label='Change in accuracy after each step', linewidth=3)
        plt.title('Accuracy Graph')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        #add title and axis labels
    return

def save_pickle(file_path, file_to_save):
    with open(file_path, 'wb') as f:
        pickle.dump(file_to_save, f)

def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read the data from the pickle file
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def split_dataset(dataset, ratio = 0.7):
    """
    This functions is used to split a dataset into training and testing 
    """
    # Divide dataset into training and testing
    # Get the length of the sublists
    total_length = len(dataset[0])  

    # Calculate the split points
    split_point = int(ratio * total_length)  # 70% of the total length

    # Split the train and test
    train_set = [dataset[0][:split_point], dataset[1][:split_point]]
    test_set = [dataset[0][split_point:], dataset[1][split_point:]]
    return train_set, test_set
 

def write_output_attention(index, configuration, attention_ranking, data_storage, dict_attention):
    env_name = configuration["env_name"]
    train_datetime = configuration["train_datetime"]
    if env_name not in R_ENV:
        filename = f"Results/ATTENTION_result/{env_name}/{train_datetime}.txt" 
    else:
        filename = f"Results/ATTENTION_result/{env_name}/{index}_{train_datetime}.txt"

    with open(filename, "w") as output_file:
        if env_name not in R_ENV:
            output_file.write(f"This is attention ranking: \n{attention_ranking}\n")
        else:
            output_file.write(f"This is attention ranking: \n{attention_ranking}\n")
            for item in attention_ranking:
                pos = data_storage["obs_pos_mapping"].get(tuple(item))
                output_file.write(f"\n{pos}\n")

    if env_name in BLOCK_UNLOCK_ENV:
        # get the key list from the dictionary
        for i in range(len(attention_ranking)):

            try:
                [image, timestep, episode_number, _] = data_storage["image_mapping"].get(tuple(attention_ranking[i]))
            except:
                continue
            try:
                [value, _, _] = dict_attention.get(tuple(attention_ranking[i]))
            except:
                value = 0

            action = ACTION.get(tuple(attention_ranking[i][-7:]))
            # it has to have the same train datetime and index and episode_number then can compare timestep
            image_path = f"Results/ATTENTION_result/{env_name}/img/{train_datetime}_{index}_ranking{i}_{timestep}_{action}_{value}.jpg"
            cv2.imwrite(image_path, image)

def write_output_causal_discovery(index, configuration, edge_params_before, edge_params_after, causal_graph, intrinsic_reward, data_storage):
    """
    This function is used to log experiment output of causal discovery algorithm
    """

    env_name = configuration["env_name"]
    train_datetime = configuration["train_datetime"]

    if env_name not in R_ENV:
        filename = f"Results/Causal_Discovery_result/{env_name}/{train_datetime}.txt"
    else:
        filename = f"Results/Causal_Discovery_result/{env_name}/{index}_{train_datetime}.txt"
    
    with open(filename, "w") as output_file:
        # output_file.write(f"This is attention ranking: \n{attention_ranking}\n")
        output_file.write(f"Structure params after remove low params: \n")
        for item in edge_params_before:
            output_file.write(f"{item}\n")
        for item in edge_params_after:
            output_file.write(f"{item}\n")
        output_file.write(f"This is causal_graph: \n{causal_graph}\n")
        output_file.write(f"This is the intrinsic reward: \n{intrinsic_reward}\n")
    
    if env_name in BLOCK_UNLOCK_ENV:
        keys_list = list(intrinsic_reward.keys())
        # get the key list from the dictionary
        for i in range(len(keys_list)):
            try:
                [image, timestep, episode_number, _] = data_storage["image_mapping"].get(keys_list[i])
            except:
                continue
     
            reward = intrinsic_reward.get(keys_list[i])
            action = ACTION.get(tuple(keys_list[i][-7:]))
            image_path = f"Results/Causal_Discovery_result/{env_name}/img/{train_datetime}_{index}_{reward:.5f}_{timestep}_{action}.jpg"
            cv2.imwrite(image_path, image)

def measure_cosine_distance(ob1, ob2):
    cosine_diff = dot(ob1, ob2.T)/(norm(ob1) * norm(ob2))    
    return cosine_diff

def update_buffer(configuration, data_storage, episode, extend_image_buffer, image_mapping = {}, obs_pos_mapping = {}):
    if len(data_storage["replay_buffer"]) <= configuration["buffer_size"]: 
        data_storage["replay_buffer"].append(episode)
        data_storage["image_mapping"].update(image_mapping)
        data_storage["obs_pos_mapping"].update(obs_pos_mapping)
        data_storage["image_encoding_buffer"] += extend_image_buffer
    else:
        # If successfully trained add to buffer and remove the oldest experience
        remove_ep = data_storage["replay_buffer"].pop(0)
        data_storage["replay_buffer"].append(episode)
        data_storage["image_mapping"].update(image_mapping)
        data_storage["obs_pos_mapping"].update(obs_pos_mapping)
        data_storage["image_encoding_buffer"] = data_storage["image_encoding_buffer"][len(remove_ep):] + extend_image_buffer
    return data_storage




