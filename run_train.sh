#bin/bash


python fine_tune.py --pref "att_unet_augment" --use_wandb --config "config/att_unet.json"

python fine_tune.py --pref "att_unet" --use_wandb --config "config/att_unet_no_aug.json"

python fine_tune.py --pref "psp_net" --use_wandb --config "config/psp_net.json"