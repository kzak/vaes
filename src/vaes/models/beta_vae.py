from vaes.models.vae import VAE
from vaes.modules.vae_module import Likelihood


class BetaVAE(VAE):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 400,
        latent_dim: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        beta: float = 4.0,
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
