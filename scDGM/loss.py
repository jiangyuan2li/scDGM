import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl

def loss(
        x,
        p_x_z,
        q_y_x,
        q_z_xy,
        p_z_y,
        n_clusters,
        latent_size,
        warm_up_weight,
        kl_weight,
        device
    ):
    pi, theta, log_r = p_x_z
    theta = torch.where(theta <= 0, torch.zeros(theta.size(0)).to(device) + 1e-8, theta)
    q_y_given_x_entropy = q_y_x.entropy()
    p_y_entropy = torch.log(torch.tensor(n_clusters).float())
    kl_divergence_y = q_y_given_x_entropy - p_y_entropy
    
    kl_divergence_z_mean = torch.zeros(x.size(0), latent_size).to(device)
    reconstruct_losses = torch.zeros(n_clusters, x.size(0)).to(device)
    for k in range(n_clusters):
      kl_divergence_z_mean += kl(q_z_xy[k],p_z_y[k])

      reconstruct_losses[k] = -log_zinb_positive(x, pi[k], theta, log_r[k]).sum(dim = 1)

    kl_divergence_z = torch.mean(kl_divergence_z_mean)
    kl_divergence_y = torch.mean(kl_divergence_y)
    reconstruction_error = torch.mean(reconstruct_losses)

    lower_bound_weighted = (
        reconstruction_error
        + warm_up_weight * kl_weight * (
            kl_divergence_z + kl_divergence_y
        )
    )

    return lower_bound_weighted, kl_divergence_z, kl_divergence_y, reconstruction_error

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
  
def log_normal(x, mu, var, eps = 0.0, dim = 0):
    if eps > 0.0:
        var = var + eps
    return torch.sum(-1/2 * (torch.log(2 * torch.tensor(np.pi)) + torch.log(var) + (x - mu)**2 / var), dim = 1)