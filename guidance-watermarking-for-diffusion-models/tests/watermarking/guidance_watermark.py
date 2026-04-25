from ...detector.detector import StableSignatureDetector, TrustmarkDetector 
from ...watermarking.watermark import GuidanceWatermarker
from ...watermarking.modified_diffusers.stable_diffusion import sd2_config
from ...watermarking.modified_diffusers.flux import flux_config
from ...watermarking.modified_diffusers.sana import sana_config


from ...transforms.transforms import TransformSet
from ...transforms.geometric import *
from ...transforms.valuemetric import *


from ...benchmarks.datasets import ImageDataset

from ...util.util_benchmark import create_wm_json_single_key, fix_randomness
from ...util.util_detector import strtobool



from ...tests import MODELDIR, LOCALDATA, MSCOCO


from os import path,makedirs

import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils


from ...benchmarks.datasets import PromptDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_guidance_ssig(diffusion_model,wm_scale, eta, difkwargs):
    prompt = "The disfigured black sun screams a bad omen to the holy otter. Numerous jellyfish arrive in the background."


    
    M=48
    attacks = TransformSet([
        CenterCrop(0.5),
        Brightness(0.2),
        Contrast(2.),
        JPEG(50, 'kornia'),
        JPEG(80, 'kornia'),
    ]
    )

    weights = [1]*(len(attacks.transform_set)+1)


    key = torch.Tensor(strtobool('001000101100110101000011001011100110011001100011')).unsqueeze(0).to(device)
    msg_extractor = torch.jit.load( path.join(MODELDIR ,'dec_48b_whit.torchscript.pt'), map_location=device).to(device)
    detector = StableSignatureDetector( model=msg_extractor,M=M,im_size=difkwargs['width'])



    difkwargs["prompt"] = prompt
    difkwargs["key"] = key

    guidance_watermark = GuidanceWatermarker(model = diffusion_model,
                                        detector=detector,
                                        transforms_set=attacks,
                                        transform_weights=weights,
                                        wm_scale=wm_scale,
                                        eta = eta,)
                                        
    

    image = diffusion_model(**difkwargs).images[0]
    image.save(path.join(LOCALDATA, f"baseline_generated_image_{diffusion_model.__class__.__name__}.png"))
    



    wm_image = guidance_watermark.generate(**difkwargs).images[0]
    wm_image.save(path.join(LOCALDATA, f"watermarked_generated_image_{diffusion_model.__class__.__name__}_ssig.png"))
    save_image(10*torch.abs(T.ToTensor()(wm_image) - T.ToTensor()(image)), path.join(LOCALDATA,f'residual_generated_image_{diffusion_model.__class__.__name__}_ssig.png'))

    twm_im = T.ToTensor()(wm_image).unsqueeze(0).to(device)
    aug_data = attacks.parallel_forward(twm_im)
    res_atk = {}
    import numpy as np

    for (augmentation_name, augmented_image) in aug_data.items():
        res = detector(augmented_image, key=key)
        res_atk[augmentation_name] =  -np.log10(res['pval'][0])
    for atk, pval in res_atk.items():
        print(f"{atk}:\t\t{np.round(pval,2)}")


def test_multiple_prompts_guidance_ssig(diffusion_model,wm_scale, eta, difkwargs, batch_size=1, nsamples=10):
    M=48 # Stable signature

    msg_extractor = torch.jit.load( path.join(MODELDIR ,'dec_48b_whit.torchscript.pt'), map_location=device).to(device)
    detector = StableSignatureDetector( model=msg_extractor,M=M,im_size=difkwargs['width'])

    attacks = TransformSet([
        CenterCrop(0.5),
        Brightness(0.2),
        Contrast(2.),
        JPEG(50, 'kornia'),
        JPEG(80, 'kornia'),
    ]
    )
    weights = [1]*(len(attacks.transform_set)+1)

    prompt_data = PromptDataset(MSCOCO,generate_keys=True)
    prompt_data =  data_utils.Subset(prompt_data, range(nsamples))
    prompt_dataloader = DataLoader(prompt_data, batch_size=batch_size, shuffle=False)

    guidance_watermark = GuidanceWatermarker(model = diffusion_model,
                                        detector=detector,
                                        transforms_set=attacks,
                                        transform_weights=weights,
                                        wm_scale=wm_scale,
                                        eta = eta,)

    for i, (id, prompts, keys) in enumerate(prompt_dataloader):
        
        print(f"{i+1}/{len(prompt_dataloader)}")
        if (i+1) == len(prompt_dataloader): difkwargs["latents"] = difkwargs["latents"][:len(prompts)]
        with torch.no_grad():
            gen_x =  diffusion_model(prompt=list(prompts), output_type='pt', **difkwargs)[0].cpu()

        wm_x = guidance_watermark(list(prompts), keys.to(device),**difkwargs).detach().cpu()
        for j in range(len(prompts)):
            save_image(wm_x[j], path.join(LOCALDATA, f"{id[j].item()}_wm_{diffusion_model.__class__.__name__}_ssig.png"))
            save_image(gen_x[j], path.join(LOCALDATA, f"{id[j].item()}_base_{diffusion_model.__class__.__name__}_ssig.png"))
            save_image(10*torch.abs(gen_x[j] - wm_x[j]), path.join(LOCALDATA, f"{id[j].item()}_res_{diffusion_model.__class__.__name__}_ssig.png"))



        











def test_sd2_guidance(wm_scale, eta=1.,num_inference_steps=25, im_size=512, seed=43,batch_size=3):
    print(f"\t Testing Stable Signature guidance on SD2 -- Seed {seed}")
    diffusion_model, difkwargs = sd2_config(num_inference_steps, im_size, seed,batch_size)
    test_multiple_prompts_guidance_ssig(diffusion_model,wm_scale, eta, difkwargs, batch_size=batch_size, nsamples=10)


def test_flux_guidance(wm_scale, eta=1.,num_inference_steps=25, im_size=256, seed=43,batch_size=3):
    print(f"\t Testing Stable SIgnature guidance on Flux -- Seed {seed}")
    diffusion_model, difkwargs = flux_config(num_inference_steps, im_size, seed,batch_size)

    test_guidance_ssig(diffusion_model,wm_scale=wm_scale, eta=eta, difkwargs=difkwargs)


def test_sana_guidance(wm_scale, eta=1.,num_inference_steps=25, im_size=512, seed=43,batch_size=3):
    print(f"\t Testing Stable SIgnature guidance on Sana -- Seed {seed}")
    
    diffusion_model, difkwargs = sana_config(num_inference_steps, im_size, seed,batch_size)

    test_multiple_prompts_guidance_ssig(diffusion_model,wm_scale, eta, difkwargs, batch_size=1, nsamples=10)



