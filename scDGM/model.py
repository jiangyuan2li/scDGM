import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as KL
import torch.nn.functional as F

from .layers import qy_given_x_encoder, qz_given_xy_encoder, px_given_z_decoder, pz_given_y_encoder
from .loss import *

class pz_prior(nn.Module):
    def __init__(self, n_in, n_out):
        super(pz_prior, self).__init__()
        self.mean_encoder = nn.Linear(n_in,n_out, bias = False)
        self.var_encoder = nn.Sequential(
            nn.Linear(n_in,n_out, bias = False),
            nn.Softplus())

    def forward(self, y):

        pz_prior_mean = self.mean_encoder(y)
        pz_prior_var = self.var_encoder(y)
        pz_prior = Normal(pz_prior_mean, torch.sqrt(pz_prior_var))

        return pz_prior, pz_prior_mean

class GMVAE_modified(nn.Module):
    def __init__(self, 
        n_input, 
        n_hidden = [200,100], 
        latent_size = 32, 
        n_clusters = 7,
        n_iw_samples = 1,
        n_mc_samples = 1,
        kl_weight = 1,
        warm_up_weight = 1):

        super(GMVAE_modified, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_size = latent_size
        self.n_iw_samples = n_iw_samples
        self.n_mc_samples = n_mc_samples
        self.n_clusters = n_clusters
        self.warm_up_weight = warm_up_weight
        self.kl_weight = kl_weight

        #self.theta_fixed = nn.Parameter(torch.rand(n_input))
        self.q_y_x_encoder = qy_given_x_encoder(n_input, n_hidden, n_clusters)
        self.pz_prior = pz_prior(n_clusters, latent_size)


        self.q_z_xy_encoder = nn.ModuleList([])
                ## n_clusters to be removed later !!!
        self.p_x_z_decoder = px_given_z_decoder(latent_size, n_clusters, n_hidden, n_input)
        for k in range(self.n_clusters):
            self.q_z_xy_encoder.append(
                qz_given_xy_encoder(
                n_input, 
                n_clusters, 
                n_hidden, 
                self.latent_size, 
                self.n_iw_samples, 
                self.n_mc_samples, 
                self.latent_size))

        ######### results holder #########
        self.q_z_xy = [None] * self.n_clusters
        self.q_z_mean = [None] * self.n_clusters
        self.z = [None] * self.n_clusters

        self.p_z_y = [None] * self.n_clusters
        self.p_z_mean = [None] * self.n_clusters

        self.mu = [None] * self.n_clusters
        self.theta = [None] * self.n_clusters
        self.pi = [None] * self.n_clusters

    def forward(self, x):

        # Y Latent space
        # p(y)
        batch_size = x.shape[0]

        p_y = Categorical(torch.tensor(1/self.n_clusters).repeat(batch_size, self.n_clusters).to(device = self.device))

        # q(y|x)
        self.q_y_x = self.q_y_x_encoder(x)

        kl_divergence_y = KL(self.q_y_x, p_y)

        y = torch.eye(self.n_clusters).to(self.device)

        kl_divergence_z = torch.zeros(batch_size, self.latent_size).to(self.device)
        reconstruct_losses = torch.zeros(self.n_clusters, batch_size).to(self.device)

        for k in range(self.n_clusters):
            self.q_z_xy[k], self.q_z_mean[k], self.z[k] = self.q_z_xy_encoder[k](x, y[k])

            self.p_z_y[k], self.p_z_mean[k] = self.pz_prior(y[k].unsqueeze(0).repeat(batch_size,1))

            kl_divergence_z += KL(self.q_z_xy[k], self.p_z_y[k])

            ########### decoder ################
            self.mu[k], self.theta[k], self.pi[k] = self.p_x_z_decoder(self.z[k].squeeze())

            reconstruct_losses[k] = -log_zinb_positive(x, self.mu[k], self.theta[k], self.pi[k]).mean(dim = 1)

        reconstruct_loss = torch.mean(torch.diag(torch.matmul(self.q_y_x.probs, reconstruct_losses)))
        kl_y = torch.mean(kl_divergence_y)
        kl_z = torch.mean(kl_divergence_z)


        elbo = reconstruct_loss + self.warm_up_weight*( - kl_y + kl_z)

        return elbo, kl_z, kl_y, reconstruct_loss
          # p(x|z)

    def get_latent_y(self, x):
        with torch.no_grad():
            self.eval()
            latent = self.q_y_x_encoder(x).probs
            self.train()
        return latent

    def get_latent_z(self, x):
        with torch.no_grad():
            self.eval()
            y = torch.eye(self.n_clusters).to(self.device)
            latent_y = self.q_y_x_encoder(x).probs
            latent = torch.zeros(x.size(0),self.latent_size)
            latent_z = torch.zeros(self.n_clusters, x.size(0), self.latent_size).to(self.device)
            for k in range(self.n_clusters):
                _,latent_z[k,:,:],_ = self.q_z_xy_encoder(x, y[k])
                latent += latent_z[k,:,:]*latent_y[:,k].unsqueeze(-1).repeat(1,self.latent_size)
            self.train()

        return latent




        ## compute loss

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

# class GMVAE_modified2(nn.Module):
#     def __init__(self, 
#         n_input, 
#         n_hidden = [200,100], 
#         latent_size = 32, 
#         n_clusters = 7,
#         n_iw_samples = 1,
#         n_mc_samples = 1,
#         kl_weight = 1,
#         warm_up_weight = 1):

#         super(GMVAE_modified2, self).__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.latent_size = latent_size
#         self.n_iw_samples = n_iw_samples
#         self.n_mc_samples = n_mc_samples
#         self.n_clusters = n_clusters
#         self.warm_up_weight = warm_up_weight
#         self.kl_weight = kl_weight

#         self.theta_fixed = nn.Parameter(torch.rand(n_input))
        
#         self.q_y_x_encoder = qy_given_x_encoder(n_input, n_hidden, n_clusters)
#         #self.pz_prior = pz_prior(n_clusters, latent_size)
#         self.p_z_y_encoder = nn.ModuleList([])

#         self.q_z_xy_encoder = nn.ModuleList([])
#         #nn.ModuleList([])
#                 ## n_clusters to be removed later !!!
#         self.p_x_z_decoder = nn.ModuleList([])
#         for k in range(self.n_clusters):
#             self.q_z_xy_encoder.append(
#                 qz_given_xy_encoder(
#                 n_input, 
#                 n_clusters, 
#                 n_hidden, 
#                 self.latent_size, 
#                 self.n_iw_samples, 
#                 self.n_mc_samples, 
#                 self.latent_size))
#             self.p_x_z_decoder.append(px_given_z_decoder(latent_size, n_clusters, n_hidden, n_input))
#             self.p_z_y_encoder.append(
#                 pz_given_y_encoder(n_clusters, n_hidden, self.latent_size)
#             )
#         ######### results holder #########
#         self.q_z_xy = [None] * self.n_clusters
#         self.q_z_mean = [None] * self.n_clusters
#         self.z = [None] * self.n_clusters

#         self.p_z_y = [None] * self.n_clusters
#         self.p_z_mean = [None] * self.n_clusters

#         self.mu = [None] * self.n_clusters
#         self.theta = [None] * self.n_clusters
#         self.pi = [None] * self.n_clusters

#     def forward(self, x):

#         # Y Latent space
#         # p(y)
#         theta = torch.where(self.theta_fixed <= 0, torch.zeros(self.theta_fixed.size(0)).to(self.device) + 1e-8, self.theta_fixed)
            
#         batch_size = x.shape[0]

#         p_y = Categorical(torch.tensor(1/self.n_clusters).repeat(batch_size, self.n_clusters).to(device = self.device))

#         # q(y|x)
#         self.q_y_x = self.q_y_x_encoder(x)

#         kl_divergence_y = KL(self.q_y_x, p_y)

#         y = torch.eye(self.n_clusters).to(self.device)

#         kl_divergence_z = torch.zeros(batch_size, self.latent_size).to(self.device)
#         reconstruct_losses = torch.zeros(self.n_clusters, batch_size).to(self.device)

#         for k in range(self.n_clusters):
#             self.q_z_xy[k], self.q_z_mean[k], self.z[k] = self.q_z_xy_encoder[k](x, y[k])

#             self.p_z_y[k], self.p_z_mean[k] = self.p_z_y_encoder[k](y[k].unsqueeze(0).repeat(batch_size,1))

#             kl_divergence_z += KL(self.q_z_xy[k], self.p_z_y[k])

#             ########### decoder ################
#             self.mu[k], self.theta[k], self.pi[k] = self.p_x_z_decoder[k](self.z[k].squeeze())

#             reconstruct_losses[k] = -log_zinb_positive(x, self.mu[k], theta, self.pi[k]).sum(dim = 1)

#         reconstruct_loss = torch.mean(reconstruct_losses)
#         kl_y = torch.mean(kl_divergence_y)
#         kl_z = torch.mean(kl_divergence_z)

#         (self.lower_bound_weighted, 
#          self.kl_divergence_z, 
#          self.kl_divergence_y, 
#          self.reconstruction_error
#          ) = loss(
#             x,
#             (self.mu, theta, self.pi),
#             self.q_y_x,
#             self.q_z_xy,
#             self.p_z_y,
#             self.n_clusters,
#             self.latent_size,
#             self.warm_up_weight,
#             self.kl_weight,
#             self.device
#          )

#         elbo = reconstruct_loss + self.warm_up_weight*( - kl_y + kl_z)

#         return elbo, kl_z, kl_y, reconstruct_loss
# #         return self.lower_bound_weighted, self.kl_divergence_z, self.kl_divergence_y, self.reconstruction_error,elbo, kl_z, kl_y, reconstruct_loss
#     # p(x|z)

#     def get_latent_y(self, x):
#         with torch.no_grad():
#             self.eval()
#             latent = self.q_y_x_encoder(x).probs
#             self.train()
#         return latent

#     def get_latent_z(self, x):
#         with torch.no_grad():
#             self.eval()
#             y = torch.eye(self.n_clusters).to(self.device)
#             latent_y = self.q_y_x_encoder(x).probs
#             latent = torch.zeros(x.size(0),self.latent_size)
#             latent_z = torch.zeros(self.n_clusters, x.size(0), self.latent_size).to(self.device)
#             for k in range(self.n_clusters):
#                 _,latent_z[k,:,:],_ = self.q_z_xy_encoder(x, y[k])
#                 latent += latent_z[k,:,:]*latent_y[:,k].unsqueeze(-1).repeat(1,self.latent_size)
#             self.train()

#         return latent
