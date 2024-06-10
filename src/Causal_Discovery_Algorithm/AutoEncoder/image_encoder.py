from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

MH_ENV = ["MH_1", "MH_2", "MH_3", "MH_4", "MH_5", "MH_6", "MH_7", "MH_8", "MH_9", "MH_10"]

def image_normalize(image, env_name):
    if env_name in MH_ENV:
        normalized_image = image/np.max(image)
        image = torch.tensor(normalized_image, dtype=torch.float32)
        image = image.unsqueeze(0) 
    else:
        normalized_image = image/np.max(image)
        image = torch.tensor(normalized_image, dtype=torch.float32)
        # image = image.permute(2, 0, 1) # Shape (7,7,3)
        image = image.reshape(3, 7, 7)
        image = image.unsqueeze(0) # Shape (1, 3, 7, 7)
    return image

class VAE_MG(nn.Module):
    def __init__(self, latent_size = 64):
        super(VAE_MG, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logvar = nn.Linear(64, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (64, 2, 2)),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
class VAE_MH(nn.Module):
    def __init__(self, latent_size=64):
        super(VAE_MH, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1095, 64),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logvar = nn.Linear(64, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 19 * 77),
            nn.ReLU(),
            nn.Unflatten(1, (128, 19, 77)),
            nn.ConvTranspose2d(128, 64, kernel_size=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        # print("encoder shape", z.shape)
        x = self.decoder(z)
        # print("decoder shape", x.shape)
        # x = x.view(-1, 1, 19, 77)  # Reshape for ConvTranspose2d layers
        # reconstruction = self.conv_transpose(x)
        x = torch.mean(x, dim=0)
        return x, mu, logvar
    
    
def loss_function(reconstruction, x, mean, log_var):
    recon_loss = nn.BCELoss(reduction='sum')(reconstruction, x)
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + KLD


def train_encoder(model_dict, image_encoding_buffer, env_name, algorithm, train_datetime):
    if env_name in MH_ENV:
        model = VAE_MH()
    else:
        model = VAE_MG()
        
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    if model_dict["encoding_model"] is True:
        checkpoint = torch.load(f'model/model_VAE_{env_name}_{algorithm}_{train_datetime}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
    best_loss = float('inf')
    if env_name in MH_ENV:
        # index = 20000//len(image_encoding_buffer)
        index = 3
    else:
        index = 20000//len(image_encoding_buffer)
        # index = 1

    for i in tqdm(range(index)):
        for image in image_encoding_buffer:

            image = image_normalize(image, env_name)

            reconstruction, mu, logvar = model(image)

            # loss = loss_function(reconstruction, image, mu, logvar)
            # print(loss)

            try:
                loss = loss_function(reconstruction, image, mu, logvar)
            except Exception as error:
                print(error)
                print(image.shape)
                continue
                # model_dict["encoding_model"] = True
                # torch.save({'model_state_dict': model.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict()}, 
	            #             f'model/model_VAE_{env_name}_{algorithm}_{train_datetime}.pth')
                # return model_dict
            
            # if loss < best_loss:
            #     best_loss = loss
            #     model_dict["encoding_model"] = model

            print(loss)
            loss.backward()
            optimizer.step()        
            wandb.log({"loss_encoding": loss})

    model_dict["encoding_model"] = True
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
	        f'model/model_VAE_{env_name}_{algorithm}_{train_datetime}.pth')
    
    return model_dict