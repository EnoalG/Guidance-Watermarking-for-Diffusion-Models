from diffusers.models import AutoencoderKL

from ...detector.detector import StableSignatureDetector 
from ...watermarking.watermark import VAEWatermarker


from ...benchmarks.datasets import ImageDataset

from ...util.util_images import normalize_vqgan,unnormalize_vqgan,identity_t
from ...util.util_benchmark import create_wm_json_single_key



from ...tests import MODELDIR, LOCALDATA
from ...tests.test_detector import test_detector_mutiple_wm


from os import path,makedirs

import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

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




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_vae_post_hoc(watermarker : VAEWatermarker,
                      im_dir, outdir,
                      im_size=512, batch_size =8,nsamples=100, ext='.jpg'):
    print('\t Testing VAE-based watermarker')
    if not path.isdir(outdir): makedirs(outdir)
    imdataset = ImageDataset(im_dir=im_dir, 
                          transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size) ]), ext=ext)
    imdataset =  data_utils.Subset(imdataset, range(nsamples))
    test_dataloader = DataLoader(imdataset, batch_size=batch_size, shuffle=False)
    max_epochs =len(test_dataloader)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            print(f"Iter: {i+1}/{max_epochs}",end='\r')
            im, _, imname = data
            ims_wm = watermarker(im.to(device))
            for i,im_wm in enumerate(ims_wm):
                save_image(im_wm, path.join(outdir, imname[i] ))



def test_stable_signature(im_dir,  im_size=512,nsamples=100,batch_size=8):
    print("\t Testing Stable Signature on SD2 (VAE)")
   

    wm_name = "stable-sig-sd2"
    outdir = path.join(LOCALDATA, 'test_ssig')
    out_data_dir = path.join(LOCALDATA, f'test_{wm_name}_flickr_ssig.json')


    checkpoint_path = path.join("/path/to/vae_ssig", 'checkpoint_sd2.pth')#LOCALDATA, 'ssig-finetune', 'checkpoint_000.pth')
    
    print("Loading checkpoint: ", checkpoint_path)
    ldm_ae = AutoencoderKL.from_pretrained(path.join(LOCALDATA, "vae", "stable-diffusion-2-1-base"))
    state_dict = torch.load(checkpoint_path)
    ldm_ae.to(device)
    ldm_ae.load_state_dict(state_dict['ldm_decoder'],strict=False)

    ssig_watermarker = VAEWatermarker(ldm_ae,  preprocess_t = identity_t, postprocess_t=identity_t)



    
    test_vae_post_hoc(ssig_watermarker,
                      im_dir, outdir,
                      im_size=im_size, batch_size =batch_size,nsamples=nsamples,ext='.jpg')
    
    
    create_wm_json_single_key(outdir,out_data_dir,ext='.jpg') # Fixed key"

    print("\t Testing if message can be decoded")
    
    M=48
    msg_decoder_path = path.join(MODELDIR ,'dec_48b_whit.torchscript.pt')
    model= torch.jit.load(msg_decoder_path).to(device)
    model.eval()

    detector = StableSignatureDetector(model, M=M,im_size=im_size).to(device)

    test_detector_mutiple_wm(detector,outdir, out_data_dir, batch_size =batch_size,nsamples=nsamples)