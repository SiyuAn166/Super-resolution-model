This repository is for CMPT732 project: Super-Resolution Diffusion Models for Faces (SRDM4Faces), which belongs to Siyu An, Sidharth Singh and Hyeon Lee. 

Here we regularly update related papers with a short summary and implementations (w/ modifications).

#1 Saharia, Chitwan, et al. "Image super-resolution via iterative refinement." TPAMI (2022). (https://arxiv.org/abs/2104.07636)
- Adopts the standard DDPM with slight modifications of the U-Net.
- The key-point is to use a condtioning mechanism in the reverse process for SR.

#2 Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR (2022). (https://arxiv.org/abs/2112.10752)
- Also known as the renowned StableDiffusion. 
- Train a global autoencoder to enable diffusion processes in the latent space.

#3 Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems." ICML 2023 under review. (https://arxiv.org/abs/2209.14687)
- Adopts the standard DDPM, but the key difference is re-modeling the reverse process.
- Circumvents the intractable likelihood term by approximating with Tweedie's formula or Laplace approximation.
