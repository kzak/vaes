import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vaes.modules import SparseVAEModule, sparse_vae_loss
from vaes.modules.vae_module import Likelihood


class SparseVAE:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        latent_dim: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        sparsity_prior: float = 0.5,
        beta: float = 1.0,
        likelihood: Likelihood | str = Likelihood.BERNOULLI,
        device: str | None = None,
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sparsity_prior = sparsity_prior
        self.beta = beta
        self.likelihood = (
            likelihood if isinstance(likelihood, Likelihood) else Likelihood(likelihood)
        )
        self.verbose = verbose
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        self.model_ = None
        self.optimizer_ = None
        self.history_ = {
            'train_loss': [],
            'train_recon': [],
            'train_kl_z': [],
            'train_kl_s': [],
        }

    def _to_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).float()
        return X.float()

    def _to_numpy(self, X: torch.Tensor) -> np.ndarray:
        return X.detach().cpu().numpy()

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | None = None) -> 'SparseVAE':
        X_tensor = self._to_tensor(X)

        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        if X_tensor.shape[1] != self.input_dim:
            raise ValueError(f'Expected input_dim={self.input_dim}, but got {X_tensor.shape[1]}')

        self.model_ = SparseVAEModule(
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
                f'Starting training (epochs={self.epochs}, sparsity_prior={self.sparsity_prior})'
            )

        for epoch in range(1, self.epochs + 1):
            train_loss, train_recon, train_kl_z, train_kl_s = self._train_epoch(dataloader, epoch)
            self.history_['train_loss'].append(train_loss)
            self.history_['train_recon'].append(train_recon)
            self.history_['train_kl_z'].append(train_kl_z)
            self.history_['train_kl_s'].append(train_kl_s)

            if self.verbose:
                logger.info(
                    f'Epoch {epoch}/{self.epochs} - '
                    f'Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, '
                    f'KL_z: {train_kl_z:.4f}, KL_s: {train_kl_s:.4f})'
                )

        if self.verbose:
            logger.info('Training completed')

        return self

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> tuple[float, float, float, float]:
        self.model_.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl_z = 0.0
        total_kl_s = 0.0
        n_samples = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not self.verbose)
        for (batch,) in progress_bar:
            batch = batch.to(self.device)
            self.optimizer_.zero_grad()

            x_recon, mu, logvar, logit_s, s = self.model_(batch)
            loss, recon_loss, kl_z, kl_s = sparse_vae_loss(
                x_recon,
                batch,
                mu,
                logvar,
                logit_s,
                s,
                sparsity_prior=self.sparsity_prior,
                beta=self.beta,
                likelihood=self.likelihood,
            )

            loss.backward()
            self.optimizer_.step()

            batch_size = batch.size(0)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl_z += kl_z.item()
            total_kl_s += kl_s.item()
            n_samples += batch_size

            progress_bar.set_postfix(
                {
                    'loss': f'{loss.item() / batch_size:.4f}',
                    'recon': f'{recon_loss.item() / batch_size:.4f}',
                    'kl_z': f'{kl_z.item() / batch_size:.4f}',
                    'kl_s': f'{kl_s.item() / batch_size:.4f}',
                }
            )

        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_kl_z = total_kl_z / n_samples
        avg_kl_s = total_kl_s / n_samples

        return avg_loss, avg_recon, avg_kl_z, avg_kl_s

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        X_tensor = self._to_tensor(X)
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        self.model_.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            mu, logvar, logit_s = self.model_.encode(X_tensor)
            # Return gated latent variables during inference
            s = (torch.sigmoid(logit_s) > 0.5).float()
            z = s * mu

        return self._to_numpy(z)

    def fit_transform(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | None = None
    ) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        Z_tensor = self._to_tensor(Z)
        if Z_tensor.ndim == 1:
            Z_tensor = Z_tensor.unsqueeze(0)

        self.model_.eval()
        with torch.no_grad():
            Z_tensor = Z_tensor.to(self.device)
            X_recon = self.model_.decode(Z_tensor)

        return self._to_numpy(X_recon)

    def reconstruct(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        X_tensor = self._to_tensor(X)
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        self.model_.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            x_recon, _, _, _, _ = self.model_(X_tensor)

        return self._to_numpy(x_recon)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        self.model_.eval()
        with torch.no_grad():
            # Sampling with sparsity consideration
            # Sample gate from Bernoulli distribution
            s = (
                (torch.rand(n_samples, self.latent_dim) < self.sparsity_prior)
                .float()
                .to(self.device)
            )
            # Sample latent variables from Gaussian distribution
            z_tilde = torch.randn(n_samples, self.latent_dim).to(self.device)
            # Apply gate
            z = s * z_tilde
            samples = self.model_.decode(z)

        return self._to_numpy(samples)

    def get_sparsity(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_ is None:
            raise ValueError('Model has not been fitted yet. Call fit() first.')

        X_tensor = self._to_tensor(X)
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        self.model_.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            _, _, logit_s = self.model_.encode(X_tensor)
            prob_s = torch.sigmoid(logit_s)

        return self._to_numpy(prob_s)

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
                    'sparsity_prior': self.sparsity_prior,
                    'beta': self.beta,
                    'likelihood': self.likelihood,
                    'random_state': self.random_state,
                },
            },
            filepath,
        )

        if self.verbose:
            logger.info(f'Model saved: {filepath}')

    def load(self, filepath: str) -> 'SparseVAE':
        checkpoint = torch.load(filepath, map_location=self.device)

        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.sparsity_prior = config.get('sparsity_prior', 0.5)
        self.beta = config.get('beta', 1.0)
        self.likelihood = config.get('likelihood', 'bernoulli')
        self.random_state = config['random_state']

        self.model_ = SparseVAEModule(
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
