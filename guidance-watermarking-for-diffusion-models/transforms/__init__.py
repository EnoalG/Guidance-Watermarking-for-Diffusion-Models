from .geometric import *
from .valuemetric import *
from .attacks import *

from .quality import *

NAME_TO_TRANSFORM = {
    'jpeg' : JPEG,
    'brighness' : Brightness,
    'contrast' : Contrast,
    'center_crop' : CenterCrop,
    'random_crop' : RandomCrop,
    'hue': Hue,
    'saturation' : Saturation,
    'median_filter' : MedianFilter,
    'gaussian_blur' : GaussianBlur,
    'gaussian_noise' : GaussianNoise,
    'resize' : Resize,
    'rotate' : Rotate
}

NAME_TO_SCORE= {
    'lpips': LPIPS,
    'clip_score' : CLIPScore,
    'psnr' : PSNR
}

def get_transform_from_name(transform_conf):
    t = list(transform_conf.keys())[0]
    params = list(transform_conf.values())[0]
    T = NAME_TO_TRANSFORM[t]

    T = T(**params)
    return(T)

def get_score_from_name(score_conf):
    s = list(score_conf.keys())[0]
    params = list(score_conf.values())[0]
    S = NAME_TO_SCORE[s]

    S = S(**params)
    return(S)