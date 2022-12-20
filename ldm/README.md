# About

We will be discussing the configuration to train the latent diffusion for FFHQ dataset. Latent Diffusion is available at the following repository:

https://github.com/CompVis/latent-diffusion

## Requirements

Please make sure you have RTX 20 series or lower graphics card. This repository does not work with anything with RTX 30 series generation of NVIDIA GPUs.
The resonsing is because of CUDA version mistmatch creating conflicts with the environment provided in the repository.

Preferrable GPU is NVIDIA V100 , at least 8 GPU if you prefer to use batch size of 256.

>Remember to create the conda environment with the provided ```environment.yml``` file. That will also download the *taming transformer* repository.


## Data Preparation

Download FFHQ dataset in 1024x1024 from their website. Extract all the images into one folder. Create a *symlink* in the *<Path_to>latent-diffusion\src\taming-transformers\data\ffhq* the *ffhq* symlink should point to the folder where all images are extracted.

## Configuring Models

There is a configuration attached for vq-4 with attention for ffhq in ```first_stage_models```, and same configuration for FFHQ ldm training is available in ```ldm\bsr_sr```. 

1. Paste _ffhq.py_ in *<Path_to_>latent-diffusion\src\taming-transformers\taming\data*
2. Open both configuration files
3. In ```data:``` find the ```target:```
4. Update the path to _ffhq.py_ provided in the locations
   >The path is usually ```taming.data.faceshq.FFHQTrain``` and ```taming.data.faceshq.FFHQValidation``` for first stage models.

   >For ldm, it is usually ```taming.data.faceshq.FFHQSRTrain``` and ```taming.data.faceshq.FFHQSRValidation```

Follow the insturctions in the repository to run the code with each configuration.
> For first stage models the ```main.py``` from *taming transformer* folder should run. Use the configuration from first_stage_models for it.

>For ldm run the ```main.py``` from *root*. Use the configuration for ldm.

### Inference

You may use the script from this link for inference:
https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing

Update the models to your own models if you choose to train from scratch for whichever model you do so for.