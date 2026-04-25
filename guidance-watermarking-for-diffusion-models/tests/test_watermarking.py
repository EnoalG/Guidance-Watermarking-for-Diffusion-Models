from ..tests.watermarking.vae_post_hoc import test_stable_signature
from ..tests.watermarking.trustmark import test_trustmark_Q100bits
from ..tests.watermarking.guidance_watermark import test_sd2_guidance, test_flux_guidance, test_sana_guidance

from . import FLICKRSET

from dataclasses import dataclass

@dataclass
class Params:
    """Class for keeping track of an item in inventory."""
    seed: int
    output_dir: str
    msg_decoder_path: str

    train_dir:str
    val_dir:str
    


    # Training params
    loss_w:str
    loss_i:str
    img_size:int
    batch_size:int
    lambda_i:float =0.2
    lambda_w:float =1.0
    optimizer:str = "AdamW,lr=5e-4"
    steps:int =100
    warmup_steps : int = 20

    num_keys:int = 1

    #Logging
    log_freq:int=10
    save_img_freq:int=1000






###############################################


def main():




    print("--- Testing Watermarker ---")

    im_dir = FLICKRSET
    nsamples=100
    im_size=512
    batch_size=8

    import os
    print(os.environ['HF_HOME'])

    test_stable_signature(im_dir, im_size=im_size,nsamples=nsamples,batch_size=batch_size)
    test_trustmark_Q100bits(im_dir, im_size=im_size,nsamples=nsamples,batch_size=batch_size)
    test_sd2_guidance(wm_scale=0.11, eta=1.,num_inference_steps=5, im_size=512, seed=43)
    test_flux_guidance(wm_scale=0.15, eta=1.,num_inference_steps=5, im_size=256, seed=43)
    test_sana_guidance(wm_scale=0.15, eta=1.,num_inference_steps=5, im_size=256, seed=43)




    


if __name__ == "__main__":
     main()