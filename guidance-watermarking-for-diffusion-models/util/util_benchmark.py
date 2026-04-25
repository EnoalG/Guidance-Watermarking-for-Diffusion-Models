import json
from os import path, listdir

import torch
import random
import numpy as np

from omegaconf import OmegaConf

CONF_PATHS = {'wm_conf':'watermarker',
              'transforms_conf': 'transforms',
              'detector_conf': 'detector',
              'test_params_conf': 'test_params' }


def create_wm_json_single_key(data_path,fname, key='001000101100110101000011001011100110011001100011',ext='jpg'):
    with open(fname, 'w') as f: 
        for im in listdir(data_path):
            if im.endswith(ext):
                data = {'name' : im, 'key': key}
                f.write(json.dumps(data) + "\n")
                f.flush()

def fix_randomness(device, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return(generator)

def load_main_config(params):
    p = vars(params)
    confs = []
    for (conf_name,c_path) in CONF_PATHS.items(): 
        if conf_name in params:
            if p[conf_name] is not None: confs.append(OmegaConf.load(path.join(params.conf_path, c_path, p[conf_name])))
    if len(confs) > 0: 
        return(OmegaConf.unsafe_merge(*confs))

def load_model_config( conf_file, params, main_conf=None):
        conf = OmegaConf.load(path.join(params.conf_path, conf_file))
        if "batch_size" in params:
            if params.batch_size is not None : OmegaConf.update(conf,'diffusion_params.batch_size', params.batch_size)
        if "nsamples" in params:
            if params.nsamples is not None : OmegaConf.update(conf,'nsamples', params.nsamples)
        
        if main_conf is not None: conf = OmegaConf.merge(main_conf, conf)
        return(conf)