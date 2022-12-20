#python "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/sr.py" -p train -c "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/config/sr_sr3_16_128.json" -enable_wandb


python "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/sr.py" -p train -c "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/config/sr_sr3_64_256.json" -enable_wandb

python "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/infer.py" -c "/home/saa204/sfuhome/Image-Super-Resolution-via-Iterative-Refinement/config/sr_sr3_64_256.json"