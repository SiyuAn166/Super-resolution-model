DPS repo is referenced from the original Diffusion Posterior Sampling for General Noisy Inverse Problems (https://github.com/DPS2022/diffusion-posterior-sampling)

#1 Clone this repo

#2 Download the pretrained DDPM model (https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)
- Create a folder named "models"
- Paste it under the "models" directory

#3 Install requirements.txt

```
Create a virtual environment (or with Conda) and activate

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

#4 Run the code below

```
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={configs/super_resolution_config.yaml};
```
