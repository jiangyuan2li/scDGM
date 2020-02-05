import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# Q(Y|X)
class qy_given_x_encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_clusters):
        super(qy_given_x_encoder, self).__init__()
        modules = []
        modules.append(nn.Linear(n_in, n_hidden[0]))
        for layer in range(len(n_hidden)-1):
          modules.append(nn.Linear(n_hidden[layer], n_hidden[layer + 1]))
          modules.append(nn.BatchNorm1d(n_hidden[layer+1]))
          modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*modules)
        
        self.logits = nn.Sequential(
            nn.Linear(n_hidden[-1], n_clusters),
            nn.BatchNorm1d(n_clusters),
        )

    def forward(self, x):
        x = self.encoder(x)

        logit = self.logits(x)

        q_y_given_x = Categorical(logits=logit)

        return q_y_given_x

# Q(Z|X,Y)
class qz_given_xy_encoder(nn.Module):
    def __init__(self, 
                 n_in, 
                 n_clusters,
                 n_hidden, 
                 n_out, 
                 n_iw_samples, 
                 n_mc_samples, 
                 latent_size):
      
      super(qz_given_xy_encoder, self).__init__()
      self.n_clusters = n_clusters
      self.n_iw_samples = n_iw_samples
      self.n_mc_samples = n_mc_samples
      self.latent_size = latent_size

      modules = []
      modules.append(nn.Linear(n_in + n_clusters, n_hidden[0]))
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
        
    def forward(self, x, y):
      y = y.unsqueeze(0).repeat(x.size(0),1)
      cat = torch.cat([x,y], dim=1)
      q = self.encoder(cat)
      
      q_m = self.mean_encoder(q)
      q_v = self.var_encoder(q)

      q_z_given_x_y = Normal(q_m, torch.sqrt(q_v))

      z_mean = q_z_given_x_y.mean

      z = q_z_given_x_y.rsample(torch.Size([
                                  self.n_iw_samples * self.n_mc_samples
                                ]))

      return q_z_given_x_y, z_mean, z

# P(Z|Y)
class pz_given_y_encoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(pz_given_y_encoder, self).__init__()

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
        
    def forward(self, y):
        q = self.encoder(y)
        
        p_m = self.mean_encoder(q)
        p_v = self.var_encoder(q)

        p_z_given_y = Normal(p_m, torch.sqrt(p_v))

        z_mean = p_z_given_y.mean
        
        return p_z_given_y, z_mean

# P(X|Z)
class px_given_z_decoder(nn.Module):
    def __init__(self, n_in, n_clusters, n_hidden, n_out):
        super(px_given_z_decoder, self).__init__()

        modules = []
        modules.append(nn.Linear(n_in, n_hidden[0]))
        for layer in range(len(n_hidden)-1):
          modules.append(nn.Linear(n_hidden[layer], n_hidden[layer + 1]))
          modules.append(nn.BatchNorm1d(n_hidden[layer+1]))
          modules.append(nn.ReLU())

        self.px_decoder = nn.Sequential(*modules)
        
        self.decoder_pi = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
            nn.BatchNorm1d(n_out),
            nn.Softplus(),
        )

        self.decoder_p = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
            nn.BatchNorm1d(n_out),
            nn.Softplus(),
        )

        self.decoder_log_r = nn.Sequential(
            nn.Linear(n_hidden[-1], n_out),
        )
        
    def forward(self, z: torch.Tensor):
        z = self.px_decoder(z)
        
        pi = self.decoder_pi(z)
        p = self.decoder_p(z)
        log_r = self.decoder_log_r(z)

        return pi, p, log_r



