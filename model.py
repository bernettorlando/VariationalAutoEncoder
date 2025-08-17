import jax
import jax.numpy as jnp
from flax import linen as nn

class Encoder(nn.Module):
    "The VAE encoder"
    latent_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        mean = nn.Dense(features=self.latent_dim, name='mean')(x)
        log_var = nn.Dense(features=self.latent_dim, name='log_var')(x)

        return mean, log_var

class Decoder(nn.Module):
    "The VAE decoder"
    hidden_dim: int = 256
    output_dim: int = 784

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(features=self.hidden_dim)(z)
        z = nn.relu(z)

        x = nn.Dense(features=self.output_dim)(z)
        return nn.sigmoid(x)


class VAE(nn.Module):
    "The overall VAE model"
    latent_dim: int

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder()
    
    def reparameterize(self, mu, sigma, key):
        eps = jax.random.normal(key, shape=sigma.shape)
        return mu + sigma * eps
    
    def __call__(self, x, key):
        #Encoder pass
        mu, log_var = self.encoder(x)
        sigma = jnp.exp(0.5 * log_var)
        z = self.reparameterize(mu, sigma, key)

        # Decoder pass
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, log_var

    def decode(self, z):
        return self.decoder(z)
    

