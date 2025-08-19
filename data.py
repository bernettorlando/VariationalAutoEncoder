import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

def get_datasets():
    """Load and preprocess MNIST dataset"""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_images = (train_ds['image'].astype(jnp.float32) / 255.0)
    test_images = (test_ds['image'].astype(jnp.float32) / 255.0)

    # train_images = jnp.round(train_images)
    # test_images = jnp.round(test_images)

    return train_images, test_images

