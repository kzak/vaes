from enum import Enum

import torch
import torch.nn as nn


class Likelihood(Enum):
    BERNOULLI = 'bernoulli'
    GAUSSIAN = 'gaussian'


class VAEModule(nn.Module):
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

        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.encoder_fc1(x))
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.decoder_fc1(z))
        x_recon = self.decoder_fc2(h)
        if self.likelihood == Likelihood.BERNOULLI:
            x_recon = torch.sigmoid(x_recon)
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
