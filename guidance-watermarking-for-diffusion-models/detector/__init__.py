from .detector import *
from os import path,listdir
from diffusers import DDIMInverseScheduler
from torch import load as tload

MODELDIR = path.join(path.dirname(__file__), './models')
NAME_TO_DETECTOR = {'stable-signature': StableSignatureDetector,
                    'trustmark': TrustmarkDetector,
                    'tree-ring' : TreeRingDetector,
                    'videoseal' : VideoSealDetector,
                    }
DETECTOR_PATH = {'stable-signature' : path.join(MODELDIR ,'dec_48b_whit.torchscript.pt'),
                 'trustmark': path.join(MODELDIR ,'Trustmark_Q_noECC_100bits_245res_resnet50.pt'),
                 'tree-ring' : DDIMInverseScheduler,
                 'videoseal' : path.join(MODELDIR, 'videoseal_y_256b_img_dec.pt')
                 }
WHITENER_PATH = '/path/to/detector/whitener/d3'

def get_detector_from_conf(conf,model=None):
    name = conf.name
    detector_class = NAME_TO_DETECTOR[name]
    detector_path = DETECTOR_PATH[name]
    whitener = None
    if name == 'tree-ring':
        model.scheduler = detector_path.from_config(model.scheduler.config) # Set inverse scheduler on pipeline model
        
    else:
        model = torch.jit.load(detector_path)
        if conf.white:
            whitener_path = path.join(WHITENER_PATH, conf.name, str(conf.im_size))
            print("[INFO] Loading whitened version of the detector:",whitener_path )

            whitener_list = [path.join(whitener_path,f) for f in listdir(whitener_path) if f.endswith('samples.pt')]
            whitener = tload(whitener_list[0], weights_only=False)
        else:
            print("[WARNING] Loading non-whitened version of the detector, this might lead to unsound p-values!")



    return(detector_class(model = model,whitener=whitener,**conf))
