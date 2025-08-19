# Variational Autoencoder (VAE) with JAX/Flax

A convolutional Variational Autoencoder implementation using JAX and Flax, trained on the MNIST dataset.

## Overview

This project implements a VAE with:
- **Convolutional encoder**: Two conv layers (32, 64 filters) followed by dense layers
- **Convolutional decoder**: Dense layer followed by two transposed conv layers  
- **2D latent space**: For easy visualization and generation
- **MSE reconstruction loss**: With KL divergence regularization

## Files

- `model.py` - VAE architecture (Encoder, Decoder, VAE classes)
- `train.py` - Training loop and image generation
- `data.py` - MNIST data loading and preprocessing
- `requirements.txt` - Dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

## Model Architecture

**Encoder**:
- Conv(32) → Conv(64) → Flatten → Dense(16) → Dense(latent_dim)
- Outputs mean and log-variance for latent distribution

**Decoder**: 
- Dense(7×7×64) → Reshape → ConvTranspose(64) → ConvTranspose(32) → ConvTranspose(1)
- Reconstructs 28×28 MNIST images

## Training

- 50 epochs with Adam optimizer (lr=1e-3)
- Batch size: 128
- Loss: MSE reconstruction + β×KL divergence (β=1.0)
- Generates 10 sample images after training

## Generated Output

The model generates new digit images by sampling from the learned latent space and displays them using matplotlib.