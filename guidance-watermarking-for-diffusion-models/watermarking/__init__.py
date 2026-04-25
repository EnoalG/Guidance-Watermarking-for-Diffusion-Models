from .modified_diffusers.stable_diffusion import *
from .modified_diffusers.flux import *

from .modified_diffusers.sana import *

from .watermark import * 


from diffusers.models import AutoencoderKL, AutoencoderDC
from os import path


NAME_TO_WM = {'norm-guidance' : NormGuidanceWatermarker,
              'prob-guidance': ProbGuidanceWatermarker,
              'trustmark' : TrustmarkNoECCWatermarker,
              'stable-signature' : VAEWatermarker,
              'vae-watermarker' : VAEWatermarker,
              'tree-ring': TreeRingWatermarker,
              'videoseal': VideoSealWatermarker,
              }

VAE_PATHS = {'stable-signature' : path.join(path.dirname(__file__), 'models/ssig_vae'),
             'trustmark' : path.join(path.dirname(__file__), 'models/trustmark_vae'),
             'videoseal' : path.join(path.dirname(__file__), 'models/videoseal_vae')}

ENC_PATHS = { 'trustmark' : path.join(path.dirname(__file__), 'models/Trustmark_Q_noECC_100bits_256to245res_encoder.pt'),
              'videoseal' : path.join(path.dirname(__file__), 'models/videoseal_y_256b_img_enc.pth'),
            }



def get_diffusion_classname_from_name(name):
    if 'stable-diffusion' in name.lower():
        diff_config = 'ModifiedStableDiffusionPipeline'
    elif 'flux' in name.lower():
        diff_config = 'ModifiedFluxPipeline'
    elif 'sana' in name.lower():
        diff_config = 'ModifiedSanaPipeline'
    else:
        raise NotImplementedError("Unknown class of models")
    return(diff_config)
def get_diffusion_model_from_name(name,params):
    if 'stable-diffusion' in name.lower():
        diff_config = sd2_config
    elif 'flux' in name.lower():
        diff_config = flux_config
    elif 'sana' in name.lower():
        diff_config = sana_config
    else:
        raise NotImplementedError("Unknown class of models")
    return(diff_config(model_id=name, **params))
def get_watermarker_from_conf(conf, **kwargs):
    wm_config = NAME_TO_WM[conf.name]
    return(wm_config(**conf, **kwargs))
def get_vae_from_name(name, conf):
    print(f"Loading VAE from {VAE_PATHS[conf.vae], name}")
    if 'sana' in name.lower():
        ldm_ae = AutoencoderDC.from_pretrained(name,subfolder='vae')
    else:
        ldm_ae = AutoencoderKL.from_pretrained(name,subfolder='vae')

    vae_path = path.join(VAE_PATHS[conf.vae], name, 'checkpoint_000.pth')
    state_dict = torch.load(vae_path)
    key = get_vae_key_from_name(name,conf)

    ldm_ae.to(device)
    ldm_ae.load_state_dict(state_dict['ldm_decoder'],strict=False)

    ldm_ae.eval()
    ldm_ae.to(device)
    return(ldm_ae,key)

def get_vae_key_from_name(name,conf):
   

    vae_path = path.join(VAE_PATHS[conf.vae], name, 'checkpoint_000.pth')
    key_file = path.join(VAE_PATHS[conf.vae], name, 'keys.txt')
    key = None
    with open(key_file, 'r') as f:
        for line in f:
            
            line = line.split('\t')
            if vae_path== line[0]: 
                key =  line[1].strip('\n')
                break
    if key is None: raise ValueError('Key not found for ', vae_path)

    return(key)
def get_posthoc_encoder(name):
    model_path = ENC_PATHS[name]
    model= torch.jit.load(model_path)
    model = model.eval()
    model = model.to(device)
    return(model)