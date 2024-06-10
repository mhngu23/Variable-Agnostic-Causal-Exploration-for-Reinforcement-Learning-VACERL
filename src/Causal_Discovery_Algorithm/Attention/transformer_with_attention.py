from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
                      
FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

R_ENV_1 = ["R_1"]

R_ENV_2 = ["R_2", "R_3", "R_4", "R_5", "R_6"]


def update_attention_dictionary(dict_attention_state_action, input_sequence, attention_weights, env_name):
    attention_weights = tuple(attention_weights.detach().numpy())
    if env_name in FROZEN_LAKE_ENV:
        for i in range(len(input_sequence)): 
            key = tuple(input_sequence[i].detach().numpy())
            if key in dict_attention_state_action.keys():
                [current_value, current_count, _, _] = dict_attention_state_action[key]
                new_value = float(attention_weights[i])
                current_count += 1
                dict_attention_state_action[key] = [max(current_value, new_value), current_count, int(key[0]), int(key[1])] 
            else:
                dict_attention_state_action[key] = [float(attention_weights[i]), 1, int(key[0]), int(key[1])]
    elif env_name in MH_ENV:
        for i in range(len(input_sequence)): 
            key = tuple(input_sequence[i].detach().numpy())
            if key in dict_attention_state_action.keys():
                [current_value, current_count, _, _] = dict_attention_state_action[key]
                new_value = float(attention_weights[i])
                current_count += 1
                dict_attention_state_action[key] = [max(current_value, new_value), current_count, key[0:1659], key[1659:1669]] 
            else:
                dict_attention_state_action[key] = [float(attention_weights[i]), 1, key[0:1659], key[1659: 1669]]
    elif env_name in R_ENV_1:
        for i in range(len(input_sequence)): 
            key = tuple(input_sequence[i].detach().numpy())
            if key in dict_attention_state_action.keys():
                [current_value, current_count, _, _] = dict_attention_state_action[key]
                new_value = float(attention_weights[i])
                current_count += 1
                dict_attention_state_action[key] = [max(current_value, new_value), current_count, key[0:10], key[10:14]] 
            else:
                dict_attention_state_action[key] = [float(attention_weights[i]), 1, key[0:10], key[10:14]]

    elif env_name in R_ENV_2:
        for i in range(len(input_sequence)): 
            key = tuple(input_sequence[i].detach().numpy())
            if key in dict_attention_state_action.keys():
                [current_value, current_count, _, _] = dict_attention_state_action[key]
                new_value = float(attention_weights[i])
                current_count += 1
                dict_attention_state_action[key] = [max(current_value, new_value), current_count, key[0:25], key[25:29]] 
            else:
                dict_attention_state_action[key] = [float(attention_weights[i]), 1, key[0:25], key[25: 29]]
    else: 
        for i in range(len(input_sequence)): 
            key = tuple(input_sequence[i].detach().numpy())
            if key in dict_attention_state_action.keys():
                [current_value, current_count, _, _] = dict_attention_state_action[key]
                new_value = float(attention_weights[i])
                current_count += 1
                dict_attention_state_action[key] = [max(current_value, new_value), current_count, key[0:147], key[147:154]] 
            else:
                dict_attention_state_action[key] = [float(attention_weights[i]), 1, key[0:147], key[147:154]]
    
    return dict_attention_state_action

def patch_attention(m):
    forward_orig = m.forward
    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)
    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, src, tgt):
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer(src, tgt)
        return output

def train_attention(buffer, model_dict, env_name, lr, train_datetime, algorithm, num_attention): 
    X = []
    y = []

    # Define your transformer model
    
    input_dim = 154
    if env_name in FROZEN_LAKE_ENV:
        input_dim = 2
    elif env_name in MH_ENV:
        input_dim = 1669
    elif env_name in R_ENV_1:
        input_dim = 14
    elif env_name in R_ENV_2:
        input_dim = 29

    nhead = 2  # Number of attention heads
    if env_name in MH_ENV:
        nhead = 1
    if env_name in R_ENV_2:
        nhead = 1
        
    num_encoder_layers = 6  # Number of encoder layers
    num_decoder_layers = 6  # Number of decoder layers
    dim_feedforward = 2048  # Dimension of the feedforward layer
    dropout = 0.1  # Dropout rate

    for episode in buffer:
        inp_seq = episode[:-1]
        out_seq = episode[-1]
        X.append(torch.tensor(inp_seq))
        y.append(torch.tensor(out_seq))

    # if model_dict["attention_model"] is False:
    model = TransformerModel(input_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model_dict["attention_model"] is True:
        checkpoint = torch.load(f'model/model_Transformer_{env_name}_{algorithm}_{num_attention}_{train_datetime}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.MSELoss() 

    save_output = SaveOutput()

    # Training loop
    correct = 0
    dict_attention_state_action = {}
    if env_name in MH_ENV:
        index = 3
    elif env_name in R_ENV_1 or env_name in R_ENV_2:
        index = 3
    else:
        index = 10000//len(X)
        
    for i in tqdm(range(index)):
        for input_sequence, target_output in zip(X, y):

            input_sequence_t = torch.tensor(input_sequence, dtype=torch.float32)
            target_output_t = torch.tensor(target_output, dtype=torch.float32)

            optimizer.zero_grad()  

            if i > index-2:

                patch_attention(model.transformer.encoder.layers[-1].self_attn)

                hook_handle = model.transformer.encoder.layers[-1].self_attn.register_forward_hook(save_output)  

            # Forward pass
            try:
                predict_output = model(input_sequence_t, target_output_t)
            except Exception as e:
                print("Error", e)
                src = input_sequence_t.unsqueeze(0)
                tgt = target_output_t.unsqueeze(0)
                print(src.shape)
                print(tgt.shape)
                continue

            if i > index-2:
                attention_weights = save_output.outputs[0][0].mean(dim=0)[-1]
                update_attention_dictionary(dict_attention_state_action, input_sequence, attention_weights, env_name)
 
            save_output.clear()

            # Compute loss
            loss = criterion(predict_output, target_output_t)
            print(loss)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            target_output = np.round(target_output_t.detach().numpy()).astype(int)
            predict_output = np.round(predict_output.detach().numpy()).astype(int)
            
            # element wise comparison
            if np.array_equal(target_output, predict_output):
                correct += 1
            
            wandb.log({"loss_attention": loss})

    for item in dict_attention_state_action.keys():
        [current_value, current_count, state, action] = dict_attention_state_action[item]
        dict_attention_state_action[item] = [current_value, state, action]
    
    # Sort the dictionary from lowest to highest value according to their value
    sorted_dict_attention_state_action = dict(sorted(dict_attention_state_action.items(), key=lambda item: item[1][0]))
    print(f"Number of analyse steps: {len(sorted_dict_attention_state_action)}")

    model_dict["attention_model"] = True
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
	        f'model/model_Transformer_{env_name}_{algorithm}_{num_attention}_{train_datetime}.pth')

    return sorted_dict_attention_state_action, model_dict


