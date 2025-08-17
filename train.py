import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from data import get_datasets
from model import VAE
import matplotlib.pyplot as plt

def loss_function(reconstructed_x, x, mu, log_var):
    # 1. Reconstruction loss
    bce_loss = optax.sigmoid_binary_cross_entropy(reconstructed_x, x).sum(axis=-1)

    # 2. KL Divergence loss
    kl_loss = -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var), axis=-1)
    kl_divergence = jnp.mean(kl_loss)

    return jnp.mean(bce_loss + kl_divergence)

@jax.jit
def train_step(state, batch, key):

    def loss_fn(params):
        reconstructed_x, mu, log_var = state.apply_fn({'params': params}, batch, key)
        loss = loss_function(reconstructed_x, batch, mu, log_var)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


if __name__ == '__main__':
    # Hyperparams
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    LATENT_DIM = 2

    train_images, test_images = get_datasets()
    num_train_steps = len(train_images) // BATCH_SIZE

    key = jax.random.PRNGKey(0)
    key, model_key, train_key = jax.random.split(key, 3)

    model = VAE(latent_dim=LATENT_DIM)

    dummy_input = jnp.ones((BATCH_SIZE, train_images.shape[1]))
    params = model.init(model_key, dummy_input, model_key)['params']

    optimizer = optax.adamw(learning_rate=LEARNING_RATE)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    for epoch in range(NUM_EPOCHS):
        perm = jax.random.permutation(train_key, len(train_images))
        train_key, _ = jax.random.split(train_key)
        shuffled_images = train_images[perm]

        epoch_loss = 0

        for i in range(num_train_steps):
            batch_images = shuffled_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

            step_key, train_key = jax.random.split(train_key)

            state, loss = train_step(state, batch_images, step_key)
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_train_steps
        print(F"Epoch {epoch + 1} / {NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    key, gen_key = jax.random.split(key)
    
    num_generated_images = 10
    random_z = jax.random.normal(gen_key, (num_generated_images, LATENT_DIM))

    generated_images = model.apply({'params': state.params}, random_z, method=model.decode)
    fig, axes = plt.subplots(1, num_generated_images, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()