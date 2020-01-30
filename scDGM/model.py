import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence as kl

from .layers import qy_given_x_encoder, qz_given_xy_encoder, pz_given_y_encoder, px_given_z_decoder
from .loss import log_zinb_positive

class GMVAE(nn.Module):
    def __init__(self, 
                 n_input, 
                 n_hidden=128, 
                 latent_size=32, 
                 n_clusters=7, 
                 n_iw_samples=1, 
                 n_mc_samples=1, 
                 kl_weight=1,
                 warm_up_weight=1):
        super(GMVAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_size = latent_size
        self.n_iw_samples = n_iw_samples
        self.n_mc_samples = n_mc_samples
        self.n_clusters = n_clusters
        self.warm_up_weight = warm_up_weight
        self.kl_weight = kl_weight

        self.q_y_x_encoder = qy_given_x_encoder(n_input, n_hidden, n_clusters)
        self.q_z_xy_encoder = nn.ModuleList([])
        self.p_z_y_encoder = nn.ModuleList([])
        self.p_x_z_decoder = nn.ModuleList([])

        for k in range(n_clusters):
            self.q_z_xy_encoder.append(qz_given_xy_encoder(
                n_input, 
                n_clusters, 
                n_hidden, 
                self.latent_size, 
                self.n_iw_samples, 
                self.n_mc_samples, 
                self.latent_size
              ))
            self.p_z_y_encoder.append(
                pz_given_y_encoder(n_clusters, n_hidden, self.latent_size)
            )
            self.p_x_z_decoder.append(
                px_given_z_decoder(self.latent_size, n_clusters, n_hidden, n_input)
            )


        self.q_z_xy = [None] * self.n_clusters
        self.p_z_y = [None] * self.n_clusters
        self.z = [None] * self.n_clusters
        kl_divergence_z_mean = [None] * self.n_clusters
        log_p_x_given_z_mean = [None] * self.n_clusters
        p_x_means = [None] * self.n_clusters
        mean_of_p_x_given_z_variances = [None] * self.n_clusters
        variance_of_p_x_given_z_means = [None] * self.n_clusters
        self.p_z_mean = [None] * self.n_clusters
        self.p_x_z = [None] * self.n_clusters
        self.p_z_means = []
        self.p_z_variances = []
        self.q_z_means = []
        self.q_z_variances = []

        # p(y)
        p_y_probabilities = torch.ones(self.n_clusters)/self.n_clusters
        p_y_logits = torch.log(p_y_probabilities).to(self.device)
        self.p_y = Categorical(logits=p_y_logits)
        self.p_y_probabilities = self.p_y.probs.unsqueeze(0)
        
    def forward(self, x):
      # Y latent space
        # p(y)
        p_y_samples = self.p_y.sample(sample_shape=torch.Size([x.size(0),1]))
        self.p_y_samples = torch.zeros([x.size(0), self.n_clusters]).to(self.device)
        for row in range(len(self.p_y_samples)):
          self.p_y_samples[row][p_y_samples[row]] = 1

        # q(y|x)
        y = torch.eye(self.n_clusters).to(self.device)
        self.q_y_x = self.q_y_x_encoder(x)
        self.q_y_logits = torch.log(self.q_y_x.probs) - torch.log1p(-self.q_y_x.probs)
        self.q_y_probabilities = torch.mean(self.q_y_x.probs, dim=0)


      # Z Latent Space
        z_mean = [None]*self.n_clusters
        for k in range(self.n_clusters):
          # q(z|x,y)
          self.q_z_xy[k], z_mean[k], self.z[k] = self.q_z_xy_encoder[k](x, y[k])
          
          #p(z|y)
          self.p_z_y[k], self.p_z_mean[k] = self.p_z_y_encoder[k](y[k].unsqueeze(0).repeat(x.size(0),1))

        self.y = self.q_y_x.probs
        
      # Decoder X
        pi = [None] * self.n_clusters
        p = [None] * self.n_clusters
        log_r = [None] * self.n_clusters
        for k in range(self.n_clusters):
          # p(x|z)
          pi[k], p[k], log_r[k] = self.p_x_z_decoder[k](self.z[k].squeeze())

      # Loss
        q_y_given_x_entropy = self.q_y_x.entropy()
        p_y_entropy = torch.log(torch.tensor(self.n_clusters).float())
        kl_divergence_y = q_y_given_x_entropy - p_y_entropy
        
        kl_divergence_z_mean = torch.zeros(x.size(0), self.latent_size).to(self.device)
        reconstruct_losses = torch.zeros(self.n_clusters, x.size(0)).to(self.device)
        for k in range(self.n_clusters):
          kl_divergence_z_mean += kl(self.q_z_xy[k],self.p_z_y[k])

          reconstruct_losses[k] = -log_zinb_positive(x, pi[k], p[k], log_r[k]).sum(dim = 1)

        self.kl_divergence_z = torch.mean(kl_divergence_z_mean)
        self.kl_divergence_y = torch.mean(kl_divergence_y)
        self.reconstruction_error = torch.mean(reconstruct_losses)

        self.lower_bound_weighted = (
            self.reconstruction_error
            + self.warm_up_weight * self.kl_weight * (
                self.kl_divergence_z + self.kl_divergence_y
            )
        )

        return self.lower_bound_weighted, self.kl_divergence_z, self.kl_divergence_y, self.reconstruction_error
    
    def get_latent_z(self, x):
      with torch.no_grad():
        y = torch.eye(self.n_clusters).to(self.device)
        latent = torch.zeros(self.n_clusters, x.size(0), self.latent_size)
        for k in range(self.n_clusters):
          self.q_z_xy_encoder[k].eval()
          _, _, z = self.q_z_xy_encoder[k](x.to(self.device), y[k])
          latent[k] = z.squeeze()
          self.q_z_xy_encoder[k].train()

      return latent.permute(1,0,2)


    def get_latent_y(self, x):
      with torch.no_grad():
        self.q_y_x_encoder.eval()
        q_y_x = self.q_y_x_encoder(x)
        self.q_y_x_encoder.train()

      return q_y_x

