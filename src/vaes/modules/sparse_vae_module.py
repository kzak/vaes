import torch
import torch.nn as nn

from vaes.modules.vae_module import Likelihood


class SparseVAEModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        latent_dim: int = 20,
        likelihood: Likelihood | str = Likelihood.BERNOULLI,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.likelihood = (
            likelihood if isinstance(likelihood, Likelihood) else Likelihood(likelihood)
        )

        # Encoder: x -> h -> (mu_z, logvar_z, logit_s)
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Logit for sparse gate
        self.encoder_fc_logit_s = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> h -> x_recon
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.encoder_fc1(x))
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)
        logit_s = self.encoder_fc_logit_s(h)
        return mu, logvar, logit_s

    def reparameterize_gaussian(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def reparameterize_bernoulli(self, logit_s: torch.Tensor) -> torch.Tensor:
        # Sample binary variables using Gumbel-Softmax trick
        # Training: Use Concrete distribution (differentiable)
        # Inference: Hard binarization
        if self.training:
            # Concrete distribution (Gumbel-Softmax)
            temperature = 0.5
            u = torch.rand_like(logit_s)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            s = torch.sigmoid((logit_s + gumbel) / temperature)
        else:
            # Hard binarization during inference
            s = (torch.sigmoid(logit_s) > 0.5).float()
        return s

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.decoder_fc1(z))
        x_recon = self.decoder_fc2(h)
        if self.likelihood == Likelihood.BERNOULLI:
            x_recon = torch.sigmoid(x_recon)
        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, logit_s = self.encode(x)
        z_tilde = self.reparameterize_gaussian(mu, logvar)
        s = self.reparameterize_bernoulli(logit_s)
        # Gated latent variable
        z = s * z_tilde
        x_recon = self.decode(z)
        return x_recon, mu, logvar, logit_s, s
