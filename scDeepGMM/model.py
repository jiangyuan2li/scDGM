import torch 
import torch.nn as nn
from torch.distributions import Categorical, Normal
from .loss import *
import math 
from sklearn.mixture import GaussianMixture

import numpy as np
import os


class GMVAE(nn.Module):
    def __init__(self, n_input, 
        n_hidden = [100,50],
        latent_size = 32, 
        n_clusters = 2, 
        warm_up_weight_z = 1,
        warm_up_weight_y = 1):

        super(GMVAE, self).__init__()

        self.n_clusters = n_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = Encoder(n_input, n_hidden, latent_size)
        self.decoder = Decoder(latent_size, n_hidden, n_input)

        self.pi_c = nn.Parameter(torch.ones(n_clusters)/n_clusters)  # pc
        self.mu_c = nn.Parameter(torch.zeros(latent_size, n_clusters)) # mu
        self.var_c = nn.Parameter(torch.ones(latent_size, n_clusters))
        self.theta = nn.Softplus(nn.Parameter(torch.ones(n_input)))
    def forward(self, x, batch_ids = None):
        
        z, z_mean, z_var = self.encoder(x)
        mu, theta, pi = self.decoder(z)

        
        #reconstruct_loss = -log_zinb_positive(x, mu, theta, pi).sum(dim = 1)

        pi_c_post, mu_c, var_c, pi_c = get_pi_c_posterior(z, (self.pi_c, self.mu_c, self.var_c))

        likelihood, kld_z, kld_c = elbo_scaleRNA((mu,theta,pi), x, pi_c_post, (mu_c, var_c, pi_c), (z_mean, z_var))

        mmd = torch.tensor(0.)
        if batch_ids is not None:
            z_mean_1 = z_mean[batch_ids == 1.]
            z_mean_0 = z_mean[batch_ids == 0.]
            mmd = compute_mmd(z_mean_0, z_mean_1)
            n = x.size(0)
            mmd = n*mmd
        return likelihood, kld_z, kld_c, mmd

    def init_gmm(self, x):

        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
        _,_,z = self.encoder(x)
        gmm.fit(z.detach().numpy())
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def get_latent_y(self, x):
        
        with torch.no_grad():
            self.eval()
            _,z_mean,_ = self.encoder(x)
            pi_c_post, mu_c, var_c, pi_c = get_pi_c_posterior(z_mean, (self.pi_c, self.mu_c, self.var_c))
            self.train()

        return pi_c_post

    def get_latent_z(self,x):
        with torch.no_grad():
            self.eval()
            _, z_mean, _ = self.encoder(x)
            self.train()
        return z_mean

class Encoder(nn.Module):
    def __init__(self, 
                 n_in, 
                 n_hidden, 
                 n_out):
      
      super(Encoder, self).__init__()


      modules = []
      modules.append(nn.Linear(n_in, n_hidden[0]))
      for layer in range(len(n_hidden)-1):
        modules.append(nn.Linear(n_hidden[layer], n_hidden[layer + 1]))
        modules.append(nn.BatchNorm1d(n_hidden[layer+1]))
        modules.append(nn.ReLU())

      self.encoder = nn.Sequential(*modules)

      self.mean_encoder = nn.Linear(n_hidden[-1], n_out)
      self.var_encoder = nn.Sequential(
          nn.Linear(n_hidden[-1], n_out),
          nn.Softplus()
      )
        
    def forward(self, x):
      q = self.encoder(x)
      
      q_m = self.mean_encoder(q)
      q_v = self.var_encoder(q)

      q_z_given_x_y = Normal(q_m, torch.sqrt(q_v))

      z_mean = q_z_given_x_y.mean
      z_var = q_z_given_x_y.scale ** 2
      
      z = q_z_given_x_y.rsample(torch.Size([1]))

      return z.squeeze(), z_mean, z_var



class Decoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):

        ## n_clusters to be removed later !!!
        
        super(Decoder, self).__init__()

        modules = []
        modules.append(nn.Linear(n_in, n_hidden[0]))
        for layer in range(len(n_hidden)-1):
          modules.append(nn.Linear(n_hidden[layer], n_hidden[layer + 1]))
          modules.append(nn.BatchNorm1d(n_hidden[layer+1]))
          modules.append(nn.ReLU())

        self.px_decoder = nn.Sequential(*modules)
        
        self.decoder_mu = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
            nn.BatchNorm1d(n_out)
        )

        self.decoder_theta = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
            nn.BatchNorm1d(n_out),
            nn.Softplus(),
        )

        self.decoder_pi = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
        )
        
    def forward(self, z: torch.Tensor):
        z = self.px_decoder(z)
        
        mu = torch.exp(self.decoder_mu(z))
        theta = self.decoder_theta(z)
        pi = self.decoder_pi(z)

        return mu, theta, pi