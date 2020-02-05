import torch
import torch.nn as nn
from torch.distributions import Categorical

from .layers import qy_given_x_encoder, qz_given_xy_encoder, pz_given_y_encoder, px_given_z_decoder
from .loss import loss

class GMVAE(nn.Module):
    def __init__(self, 
                 n_input, 
                 n_hidden=[100,100], 
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
        self.theta = nn.Parameter(
            torch.rand(n_input)
        )

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
        (self.lower_bound_weighted, 
         self.kl_divergence_z, 
         self.kl_divergence_y, 
         self.reconstruction_error
         ) = loss(
            x,
            (pi, self.theta, log_r),
            self.q_y_x,
            self.q_z_xy,
            self.p_z_y,
            self.n_clusters,
            self.latent_size,
            self.warm_up_weight,
            self.kl_weight,
            self.device
         )

        return self.lower_bound_weighted, self.kl_divergence_z, self.kl_divergence_y, self.reconstruction_error
    
    def get_latent_z(self, x, length):
      with torch.no_grad():
        self.q_z_xy_encoder.eval()
        self.q_z_xy_encoder = self.q_z_xy_encoder.to(self.device)
        y = torch.eye(self.n_clusters).to(self.device)
        latent = torch.zeros(self.n_clusters, length, self.latent_size)
        for i, sample in enumerate(x):
            ind1 = i*sample['x'].size(0)
            ind2 = ind1 + sample['x'].size(0)
            input = sample['x'].to(self.device)
            for k in range(self.n_clusters):
                self.q_z_xy_encoder[k].eval()
                _, _, z = self.q_z_xy_encoder[k](input, y[k])
                latent[k,ind1:ind2,:] = z.squeeze()
                self.q_z_xy_encoder[k].train()
        self.q_z_xy_encoder.train()

      return latent.permute(1,0,2)


    def get_latent_y(self, x, length=None):
        with torch.no_grad():
            if length:
                self.q_y_x_encoder = self.q_y_x_encoder.to(self.device)
                self.q_y_x_encoder.eval()
                latent = torch.zeros(length, self.n_clusters)
                for i, sample in enumerate(x):
                    ind1 = i*sample['x'].size(0)
                    ind2 = ind1 + sample['x'].size(0)
                    input = sample['x'].to(self.device)
                    latent[ind1:ind2] = self.q_y_x_encoder(input).probs     
            else:
                latent = self.q_y_x_encoder(x).probs
            self.q_y_x_encoder.train()


        return latent

