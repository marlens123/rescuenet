# rescuenet
Contribution to the RescueNet Challenge 2023.
- Announcement: https://www.hadr.ai/2023-rescuenet-challenge
- GitHub: https://github.com/BinaLab/RescueNet-Challenge2023
- Data: https://drive.google.com/drive/folders/1Gv7fr5YiIkNrEdVKMiIIKugTxuB6dHQH
- Paper: https://arxiv.org/pdf/2202.12361.pdf 


- Image Size: 704 (needs to be divisible by 32)
- Augmentation methods and probabilities can be changed in ```utils/augmentation.py```

Run training with ```python fine_tune.py```. Will use ```config/att_unet.json```. Hyperparameters can be changed by writing a new config in the same style. For more information on possible values see ```config/README.md```. Logs will be stored in ```metrics/scores/fine_tune```. Adding ```--use_wandb``` option to train call will additionally use wandb for monitoring.

Hyperparameter tuning and prediction scripts are not tested yet.