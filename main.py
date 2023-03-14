import torch
from phenaki import CViViT, CViViTTrainer
import logging
import gdown

## config logging
logging.basicConfig(filemode="w", filename="train.log", format='%(asctime)s - %(message)s', level=logging.INFO)

def main(device):

    cvivit = CViViT(
        dim=512,
        codebook_size=5000,
        image_size=256,
        patch_size=32,
        temporal_patch_size=2,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=64,
        heads=8
    ).to(device=device)
    
    trainer = CViViTTrainer(
        cvivit,
        folder='./images/images/',
        batch_size=4,
        grad_accum_every=4,
        train_on_images=True,
        use_ema=False,
        num_train_steps=1000
    ) 
    
    trainer.train()
    
if __name__ == "__main__":
    
    # download image from google drive
    logging.info("Start download image....")
    url = 'https://drive.google.com/drive/folders/1sVE0YB2-pjob6GHvaXb5I43QkaypkeRl?usp=share_link'
    gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)
    
    # training
    logging.info("Start training process....")
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    main(device=device)

