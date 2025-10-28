import torch
import torch.nn as nn

from vaes.modules.vae_module import Likelihood


def reconstruction_loss(
    x_recon: torch.Tensor, x: torch.Tensor, likelihood: Likelihood | str = Likelihood.BERNOULLI
) -> torch.Tensor:
    lik = likelihood if isinstance(likelihood, Likelihood) else Likelihood(likelihood)
    if lik == Likelihood.BERNOULLI:
        # Bernoulli distribution: for data normalized to [0,1]
        # Decoder output is restricted to [0,1] with sigmoid
        return nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    elif lik == Likelihood.GAUSSIAN:
        # Gaussian distribution: for continuous-valued data
        # Decoder output is unrestricted
        return nn.functional.mse_loss(x_recon, x, reduction='sum')
    else:
        raise ValueError(f"likelihood must be 'bernoulli' or 'gaussian', got '{likelihood}'")


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    likelihood: Likelihood | str = Likelihood.BERNOULLI,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction error
    recon_loss = reconstruction_loss(x_recon, x, likelihood)

    # KL divergence: KL(q(z|x)||p(z))
    # p(z) = N(0, I), q(z|x) = N(mu, diag(exp(logvar)))
    # KL = -0.5 * Σ[1 + logvar - mu^2 - exp(logvar)]
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # β-VAE loss: L = Reconstruction error + β * KL divergence
    # β = 1.0: Standard VAE (ELBO)
    # β > 1.0: Emphasize KL divergence (promote disentanglement)
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


def log_density_gaussian(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # log p(z) = log N(z; mu, diag(exp(logvar)))
    # = -0.5 * [log(2π) + logvar + (z - mu)^2 / exp(logvar)]
    normalization = -0.5 * (torch.log(torch.tensor(2.0 * torch.pi)) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((z - mu) ** 2 * inv_var)
    return log_density


def beta_tcvae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    likelihood: Likelihood | str = Likelihood.BERNOULLI,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction error
    recon_loss = reconstruction_loss(x_recon, x, likelihood)

    # Batch size
    batch_size = z.size(0)

    # log q(z|x) = log N(z; mu, diag(exp(logvar)))
    # shape: (batch_size, latent_dim)
    log_qz_given_x = log_density_gaussian(z, mu, logvar).sum(dim=1)

    # log q(z) ≈ log (1/M) Σ_m q(z|x_m)
    # Evaluate z with mu, logvar of each sample and take minibatch average
    # shape: (batch_size, batch_size, latent_dim)
    z_expanded = z.unsqueeze(1)
    mu_expanded = mu.unsqueeze(0)
    logvar_expanded = logvar.unsqueeze(0)

    # log q(z|x_m) for all pairs (z_i, x_m)
    # shape: (batch_size, batch_size, latent_dim)
    log_qz_given_xm = log_density_gaussian(z_expanded, mu_expanded, logvar_expanded)

    # log q(z) = log (1/M) Σ_m q(z|x_m)
    # = log (1/M) Σ_m exp(Σ_j log q(z_j|x_m))
    # = log (1/M) + log Σ_m exp(Σ_j log q(z_j|x_m))
    # shape: (batch_size,)
    log_qz = torch.logsumexp(log_qz_given_xm.sum(dim=2), dim=1) - torch.log(
        torch.tensor(batch_size, dtype=torch.float32, device=z.device)
    )

    # log Π_j q(z_j) ≈ Σ_j log (1/M) Σ_m q(z_j|x_m)
    # shape: (batch_size,)
    log_prod_qzj = (
        torch.logsumexp(log_qz_given_xm, dim=1)
        - torch.log(torch.tensor(batch_size, dtype=torch.float32, device=z.device))
    ).sum(dim=1)

    # log p(z) = log N(z; 0, I)
    # shape: (batch_size,)
    log_pz = log_density_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)

    # Compute each term of KL decomposition
    # I(z;x) = E_q[log q(z|x) - log q(z)]
    index_code_mi = (log_qz_given_x - log_qz).mean()

    # TC = E_q[log q(z) - log Π_j q(z_j)]
    total_correlation = (log_qz - log_prod_qzj).mean()

    # Dimension-wise KL = E_q[log Π_j q(z_j) - log p(z)]
    dimension_wise_kl = (log_prod_qzj - log_pz).mean()

    # β-TCVAE loss
    # L = Reconstruction error + α * I(z;x) + β * TC + γ * Dimension-wise KL
    total_loss = (
        recon_loss
        + alpha * index_code_mi * batch_size
        + beta * total_correlation * batch_size
        + gamma * dimension_wise_kl * batch_size
    )

    return (
        total_loss,
        recon_loss,
        index_code_mi * batch_size,
        total_correlation * batch_size,
        dimension_wise_kl * batch_size,
    )


def sparse_vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    logit_s: torch.Tensor,
    s: torch.Tensor,
    sparsity_prior: float = 0.5,
    beta: float = 1.0,
    likelihood: Likelihood | str = Likelihood.BERNOULLI,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction error
    recon_loss = reconstruction_loss(x_recon, x, likelihood)

    # KL(q(z|x)||p(z)): KL divergence for Gaussian distribution
    # KL for gated latent variables: s * KL(N(mu, sigma^2) || N(0, 1))
    # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * [1 + log(sigma^2) - mu^2 - sigma^2]
    kl_z = -0.5 * torch.sum(s * (1 + logvar - mu.pow(2) - logvar.exp()))

    # KL(q(s|x)||p(s)): KL divergence for Bernoulli distribution
    # q(s|x) = Bernoulli(sigmoid(logit_s))
    # p(s) = Bernoulli(sparsity_prior)
    # KL(Bernoulli(p) || Bernoulli(q)) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    prob_s = torch.sigmoid(logit_s)
    eps = 1e-8

    # KL(q(s|x) || p(s))
    kl_s = prob_s * (torch.log(prob_s + eps) - torch.log(torch.tensor(sparsity_prior + eps)))
    kl_s = kl_s + (1 - prob_s) * (
        torch.log(1 - prob_s + eps) - torch.log(torch.tensor(1 - sparsity_prior + eps))
    )
    kl_s = torch.sum(kl_s)

    # Sparse VAE loss
    # L = Reconstruction error + β * (KL(q(z|x)||p(z)) + KL(q(s|x)||p(s)))
    total_loss = recon_loss + beta * (kl_z + kl_s)

    return total_loss, recon_loss, kl_z, kl_s
