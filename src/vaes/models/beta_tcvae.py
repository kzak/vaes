import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vaes.models.vae import VAE
from vaes.modules import VAEModule, beta_tcvae_loss
from vaes.modules.vae_module import Likelihood


class BetaTCVAE(VAE):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        latent_dim: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        likelihood: Likelihood | str = Likelihood.BERNOULLI,
        device: str | None = None,
        verbose: bool = True,
        random_state: int = 42,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            beta=beta,
            likelihood=likelihood,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.history_['train_mi'] = []
        self.history_['train_tc'] = []
        self.history_['train_dwkl'] = []

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> tuple[float, float, float]:
        self.model_.train()
        total_loss = 0.0
        total_recon = 0.0
        total_mi = 0.0
        total_tc = 0.0
        total_dwkl = 0.0
        n_samples = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not self.verbose)
        for (batch,) in progress_bar:
            batch = batch.to(self.device)
            self.optimizer_.zero_grad()

            x_recon, mu, logvar = self.model_(batch)
            z = self.model_.reparameterize(mu, logvar)
            loss, recon_loss, mi, tc, dwkl = beta_tcvae_loss(
                x_recon,
                batch,
                mu,
                logvar,
                z,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                likelihood=self.likelihood,
            )

            loss.backward()
            self.optimizer_.step()

            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_mi += mi.item()
            total_tc += tc.item()
            total_dwkl += dwkl.item()
            n_samples += batch_size

            progress_bar.set_postfix(
                {
                    'loss': f'{loss.item() / batch_size:.4f}',
                    'recon': f'{recon_loss.item() / batch_size:.4f}',
                    'mi': f'{mi.item() / batch_size:.4f}',
                    'tc': f'{tc.item() / batch_size:.4f}',
                    'dwkl': f'{dwkl.item() / batch_size:.4f}',
                }
            )

        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_mi = total_mi / n_samples
        avg_tc = total_tc / n_samples
        avg_dwkl = total_dwkl / n_samples

        self.history_['train_mi'].append(avg_mi)
        self.history_['train_tc'].append(avg_tc)
        self.history_['train_dwkl'].append(avg_dwkl)

        return avg_loss, avg_recon, avg_mi

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | None = None) -> 'BetaTCVAE':
        X_tensor = self._to_tensor(X)

        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        if X_tensor.shape[1] != self.input_dim:
            raise ValueError(f'Expected input_dim={self.input_dim}, but got {X_tensor.shape[1]}')

        self.model_ = VAEModule(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            likelihood=self.likelihood,
        ).to(self.device)

        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.verbose:
            logger.info(
                f'Starting training (epochs={self.epochs}, α={self.alpha}, β={self.beta}, γ={self.gamma})'
            )

        for epoch in range(1, self.epochs + 1):
            train_loss, train_recon, train_mi = self._train_epoch(dataloader, epoch)
            self.history_['train_loss'].append(train_loss)
            self.history_['train_recon'].append(train_recon)
            self.history_['train_kl'].append(train_mi)

            if self.verbose:
                logger.info(
                    f'Epoch {epoch}/{self.epochs} - '
                    f'Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, MI: {train_mi:.4f}, '
                    f'TC: {self.history_["train_tc"][-1]:.4f}, DW-KL: {self.history_["train_dwkl"][-1]:.4f})'
                )

        if self.verbose:
            logger.info('Training completed')

        return self

    def save(self, filepath: str) -> None:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        torch.save(
            {
                'model_state_dict': self.model_.state_dict(),
                'optimizer_state_dict': self.optimizer_.state_dict(),
                'history': self.history_,
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'latent_dim': self.latent_dim,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma,
                    'likelihood': self.likelihood,
                    'random_state': self.random_state,
                },
            },
            filepath,
        )

        if self.verbose:
            logger.info(f'Model saved: {filepath}')

    def load(self, filepath: str) -> 'BetaTCVAE':
        checkpoint = torch.load(filepath, map_location=self.device)

        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 6.0)
        self.gamma = config.get('gamma', 1.0)
        self.likelihood = config.get('likelihood', 'bernoulli')
        self.random_state = config['random_state']

        self.model_ = VAEModule(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            likelihood=self.likelihood,
        ).to(self.device)
        self.model_.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])

        self.history_ = checkpoint['history']

        if self.verbose:
            logger.info(f'Model loaded: {filepath}')

        return self
