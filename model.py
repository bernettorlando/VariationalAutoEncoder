import jax
import jax.numpy as jnp
from flax import linen as nn

class Encoder(nn.Module):
    "The VAE encoder"
    latent_dim: int
    hidden_dim: int = 256
    hidden_dim_1: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        mean = nn.Dense(features=self.latent_dim, name='mean')(x)
        log_var = nn.Dense(features=self.latent_dim, name='log_var')(x)

        return mean, log_var

class Decoder(nn.Module):
    "The VAE decoder"
    hidden_dim_1: int = 128
    hidden_dim: int = 256
    output_dim: int = 784
 
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(features=7 * 7 * 64)(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 7, 7, 64))

        z = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)

        z = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)

        x = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')(z)
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
    

