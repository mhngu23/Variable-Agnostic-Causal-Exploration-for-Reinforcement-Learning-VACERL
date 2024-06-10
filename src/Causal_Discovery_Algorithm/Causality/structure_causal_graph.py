import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

FROZEN_LAKE_ENV = ["4x4FL", "8x8FL", "4x4FL_noisy_TV", "8x8FL_noisy_TV"]

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def normalize_vector(vector):
    # Convert the vector to a NumPy array for easy mathematical operations
    vector_array = np.array(vector)

    # Find the minimum and maximum values in the vector
    min_val = np.min(vector_array)
    max_val = np.max(vector_array)

    # Normalize the vector to be between 0 and 1
    normalized_vector = (vector_array - min_val) / (max_val - min_val)

    return normalized_vector.tolist()

class StructureParams(nn.Module):
    def __init__(self, numbers_of_object, attention_ranking, device):
        super(StructureParams, self).__init__() 
        self.device = device
        self.attention_ranking = attention_ranking
        # Init the self.numbers_of_variable * self.numbers_of_variable matrix with 0 value for update
        edge_params = torch.nn.Parameter(torch.rand((numbers_of_object, numbers_of_object)))
        # Register the original list of params
        self.register_parameter('edge_params', edge_params)

class FunctionalNet(nn.Module):
    def __init__(self, numbers_of_object, attention_ranking, device):
        super(FunctionalNet, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.numbers_of_object = numbers_of_object
        self.attention_ranking = attention_ranking
        self.device = device

        input_size = len(attention_ranking[0]) 
        hidden_size = 512
        output_size = len(attention_ranking[0]) 
        self.fs = nn.Sequential(
                init_(nn.Linear(input_size, hidden_size)),
                nn.LeakyReLU(negative_slope = 0.1),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(negative_slope = 0.1),
                init_(nn.Linear(hidden_size, hidden_size)),
                nn.LeakyReLU(negative_slope = 0.1),
                init_(nn.Linear(hidden_size, output_size)))       
                
       
    def forward(self, inputs, structure_params, output_index, input_index):
        """
        inputs (list(tuple)): sequence of (state, action) pairs
        """
        inputs_t = torch.tensor(inputs).to(self.device)

        inputs_t = inputs_t * structure_params[output_index][input_index].to(self.device).unsqueeze(dim=-1)

        output = self.fs(inputs_t)

        return output

class SCM:
    def __init__(self, configuration, number_of_objects, attention_ranking, device):
        """
        configuration: (dict) The dictionary of args configuration.
        number_of_objects: (int) The number of state that will be consider as object for causal discovery.
        attention_ranking: (list) A list of attention ranking the contribution of each object to the goal in order from low to high.
        
        """

        self.device = device
        self.attention_ranking = attention_ranking
        
        # Init the two networks for Structure and Functional Parameter
        self.f_nets = FunctionalNet(number_of_objects, self.attention_ranking, device).to(device)
        self.s_params = StructureParams(number_of_objects, self.attention_ranking, device).to(device)
        
        self.criterion = nn.MSELoss() 

        # Init the two optimizers
        self.f_optimizer = torch.optim.Adam(self.f_nets.parameters(), lr=configuration["f_lr"]) 
        self.s_optimizer = torch.optim.Adam(self.s_params.parameters(), lr= configuration["s_lr"]) 
        
        self.training_f =  configuration["training_f"]
        self.training_s = configuration["training_s"]

        # Best Model
        self.best_s_param = None
        self.best_f_param = None
        self.best_loss = float('inf')

    def sample_configuration(self):
        # Apply sigmoid and bernoulli
        structure_params = torch.bernoulli(torch.sigmoid(self.s_params.edge_params))
        e = torch.eye(structure_params.shape[0], device = structure_params.device).bool()
        structure_params = structure_params.masked_fill(e, 1)
        return structure_params.to(self.device)
    

    def train_f(self, env_name, X, y, loss_list_f, accuracy_f, batch_size, wandb):
        for param in self.f_nets.parameters():
            param.requires_grad = True

        for _ in tqdm(range(self.training_f)):
            batch_index = [random.randint(0, len(X)-1) for _ in range(batch_size)]
            sample_structure_params = self.sample_configuration().detach()

            data_x = [X[index] for index in batch_index]
            data_y = [y[index] for index in batch_index]

            sample_structure_params = (1 - sample_structure_params)

            # Training loop
            correct = 0
            loss = 0
            self.f_optimizer.zero_grad()
            for input_sequence, target_output in zip(data_x, data_y):
                # Get the index of the output in the target ranking
                if env_name in FROZEN_LAKE_ENV:
                    input_index = [self.attention_ranking.index(input) for input in input_sequence]
                    output_index = self.attention_ranking.index(target_output)
                else:          
                    input_index = [self.attention_ranking.index(tuple(input)) for input in input_sequence]
                    output_index = self.attention_ranking.index(tuple(target_output))

                target_output_t = torch.tensor(target_output).float().to(self.device)

                predict_output = self.f_nets.forward(input_sequence, sample_structure_params, output_index, input_index)
                
                # Calculate cross-entropy loss
                loss +=  self.criterion(predict_output, target_output_t)

                predict_output = torch.mean(predict_output, dim=0)   

                target_output = np.round(target_output_t.cpu().detach().numpy()).astype(int)
                predict_output = np.round(predict_output.cpu().detach().numpy()).astype(int)

                # element wise comparison between 2 vectors
                if np.array_equal(target_output, predict_output):
                    correct += 1

            
            loss.backward()
            self.f_optimizer.step()

            # calculate the average_loss per batch
            loss_list_f.append(loss.item()/batch_size)
            
            # calculate the accuracy per batch
            accuracy = correct / batch_size * 100
            accuracy_f.append(accuracy)
            correct = 0 

            wandb.log({"loss_f": loss.item()/batch_size})

    
    def train_s(self, env_name, X, y, loss_list_s, accuracy_s, batch_size, wandb):

        # Freezing update of f_nets
        for param in self.f_nets.parameters():
            param.requires_grad = False
        
        for _ in tqdm(range(self.training_s)):                                             
            batch_index = [random.randint(0, len(X)-1) for _ in range(batch_size)]

            data_x = [X[index] for index in batch_index]
            data_y = [y[index] for index in batch_index]

            # Training loop
            correct = 0
            loss = 0 
            for input_sequence, target_output in zip(data_x, data_y):

                # Clear the gradient at all edges
                self.s_params.edge_params.grad = torch.zeros_like(self.s_params.edge_params)
                
                # Get the index of the output in the target ranking
                if env_name in FROZEN_LAKE_ENV:
                    input_index = [self.attention_ranking.index(input) for input in input_sequence]
                    output_index = self.attention_ranking.index(target_output)
                else:
                    input_index = [self.attention_ranking.index(tuple(input)) for input in input_sequence]
                    output_index = self.attention_ranking.index(tuple(target_output))

                target_output_t = torch.tensor(target_output).float().to(self.device)

                predict_output = self.f_nets.forward(input_sequence, self.s_params.edge_params, output_index, input_index)
            
                loss += self.criterion(predict_output, target_output_t) 
                
                predict_output = torch.mean(predict_output, dim=0)                
                target_output = np.round(target_output_t.cpu().detach().numpy()).astype(int)
                predict_output = np.round(predict_output.cpu().detach().numpy()).astype(int)

                # Element wise comparison between 2 vectors
                if np.array_equal(target_output, predict_output):
                    correct += 1
                    
            # Calculate regularize loss 
            siggamma = self.s_params.edge_params.sigmoid()
            # Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-0.05)
            Lsparse  = siggamma.sum().mul(0.05) 
            loss = loss + Lsparse
            loss = loss

            if loss <= self.best_loss:
                self.best_f_param = self.f_nets
                self.best_s_param = self.s_params
                
            loss.backward()
            self.s_optimizer.step()

            # Taking the average loss of one batch
            loss_list_s.append(loss.item()/batch_size)

            # Taking the accuracy of one batch
            accuracy = correct / batch_size * 100
            accuracy_s.append(accuracy)
            correct = 0 
            
            # wandb.log({"loss_s": loss/batch_size, "accuracy_s": accuracy})
            wandb.log({"loss_s": loss/batch_size})
    
    def test(self, test_set, env_name):

        edge_params_sigmoid = torch.sigmoid(self.best_s_param.edge_params).detach()
        for i in range(len(edge_params_sigmoid)):   
            for j in range(len(edge_params_sigmoid)):
                if edge_params_sigmoid[i][j] > edge_params_sigmoid[j][i]:
                    edge_params_sigmoid[j][i] = 0
                else:
                    edge_params_sigmoid[i][j] = 0
        
        thresholds = [0.55, 0.6, 0.65, 0.7]
        highest_accuracy_threshold = 0.5
        highest_accuracy = 0
        
        for threshold in thresholds:
            edge_params_sigmoid_after = torch.where(edge_params_sigmoid > threshold, torch.tensor(1), torch.tensor(0))
            edge_params_sigmoid_after = torch.tensor(edge_params_sigmoid_after).float().to(self.device)
            count = 0
            correct = 0
            for input_sequence, target_output in zip(test_set[0], test_set[1]):
                count += 1

                if env_name in FROZEN_LAKE_ENV:
                    input_index = [self.attention_ranking.index(input) for input in input_sequence]
                    output_index = self.attention_ranking.index(target_output)
                else:
                    input_index = [self.attention_ranking.index(tuple(input)) for input in input_sequence]
                    output_index = self.attention_ranking.index(tuple(target_output))            
                
                target_output_t = torch.tensor(target_output).float().to(self.device)
                predict_output = self.best_f_param.forward(input_sequence, edge_params_sigmoid_after, output_index, input_index)
                
                predict_output = torch.mean(predict_output, dim=0)   

                target_output = np.round(target_output_t.cpu().detach().numpy()).astype(int)
                predict_output = np.round(predict_output.cpu().detach().numpy()).astype(int)

                if np.array_equal(target_output, predict_output):
                    correct += 1

            # Calculate and compare accuracy
            accuracy = correct / count * 100
            print(f"The accuracy of {threshold} on the test_set is {accuracy}")
            if accuracy > highest_accuracy:
                highest_accuracy_threshold = threshold
                highest_accuracy = accuracy
            else:
                continue

        return highest_accuracy_threshold

            
