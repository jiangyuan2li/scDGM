import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl
import math

def elbo_scaleRNA(zinb_params, x, pi_c_post, c_params, z_params):
    # get conditional likelihood first p(x|z)
    eps = 1e-6
    mu, theta, pi = zinb_params
    reconstruct_loss = -log_zinb_positive(x, mu, theta, pi).sum(dim = 1)

    #####################
    mu_c, var_c, pi_c = c_params
    z_mean, z_var = z_params

    n_clusters = pi_c.shape[1]

    z_mean_expand = z_mean.unsqueeze(2).expand(z_mean.shape[0], z_mean.shape[1], n_clusters)
    z_var_expand = z_var.unsqueeze(2).expand(z_var.shape[0], z_var.shape[1], n_clusters)

    ####################
    # log p(z|c)
    logpzc = -0.5*torch.sum(pi_c_post * torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c + eps) + \
                                           z_var_expand/(var_c + eps) + \
                                           (z_mean_expand - mu_c)**2/(var_c + eps), dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(pi_c_post*torch.log(pi_c + eps), 1)

    # log q(z|x)
    qentropy = -0.5 * torch.sum(1+torch.log(z_var + eps) + math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(pi_c_post * torch.log(pi_c_post + eps), 1)

    kld_z =  qentropy - logpzc 
    kld_c =  logqcx - logpc

    return torch.mean(reconstruct_loss), torch.mean(kld_z), torch.mean(kld_c)



def get_pi_c_posterior(z, prior):
    eps = 1e-10
    pi_c, mu_c, var_c = prior
    #print(pi_c)
    pi_c = F.relu(pi_c)
    #var_c = F.softplus(var_c)
    
    n_clusters = pi_c.shape[0]
    batch_size = z.shape[0]
    N = batch_size

    z = z.unsqueeze(2).expand(z.shape[0], z.shape[1], n_clusters)

    pi_c = pi_c.repeat(N,1) # N*D
    mu_c = mu_c.repeat(N,1,1) # N*D*K
    var_c = var_c.repeat(N,1,1) # N*D*K


    pi_c_prob = torch.exp(torch.log(pi_c) - torch.sum(0.5 * torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim = 1)) + eps
    # pi_c_prob[pi_c_prob != pi_c_prob] = 0.
    # pi_c_prob = torch.clamp(pi_c_prob, 1e-6, 1e6)
    pi_c_post = pi_c_prob / torch.sum(pi_c_prob, dim = 1, keepdim = True)

    return pi_c_post, mu_c, var_c, pi_c



def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
