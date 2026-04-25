import torch
from omegaconf import OmegaConf
from .benchmarks import get_dataset_from_conf
from .watermarking import get_diffusion_model_from_name, get_vae_from_name,get_watermarker_from_conf
from torch.utils.data import DataLoader,Subset
from torchvision.utils import save_image
from .util.util_benchmark import load_main_config,load_model_config
from .util.util_detector import strtobool

import argparse
import json

from os import path,makedirs,remove
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(params,main_conf):
    for conf_file in params.confs:

        conf = load_model_config(conf_file, params, main_conf)

        

        if params.wm_conf is not None and 'key_type' in conf.watermarker:
            dataset = get_dataset_from_conf(conf.dataset,M=1, key_type=conf.watermarker.key_type)
        else: 
            dataset = get_dataset_from_conf(conf.dataset,M=1)

        subdataset = Subset(dataset, range(conf.nsamples))
        prompt_dataloader = DataLoader(subdataset, batch_size=conf.diffusion_params.batch_size, shuffle=False)

        diffusion_model, difkwargs = get_diffusion_model_from_name(conf.model, conf.diffusion_params)
        diffusion_model = diffusion_model.to(device)

        result_dir = path.join(params.res_dir, conf.dataset.name, diffusion_model.__class__.__name__, conf.model, 'cover')


        wrapped_model = False

        if params.key is not None: key = strtobool(params.key)
        else:   key = None
        if params.wm_conf is not None:
            if conf.watermarker.name in 'stable-signature' or conf.watermarker.name in 'vae-watermarker':
                vae,key = get_vae_from_name(conf.model,conf.watermarker)
                diffusion_model.vae = vae.to(diffusion_model.dtype)
            elif conf.watermarker.name in 'tree-ring':
                
                wm_diffusion_model = get_watermarker_from_conf(conf.watermarker, diffuser= diffusion_model)
                wm_diffusion_model.to(device)
                wrapped_model = True

            result_dir = path.join(params.res_dir, conf.dataset.name, diffusion_model.__class__.__name__, conf.model, conf.watermarker.name, conf.watermarker.vae, 'images')
        if not path.isdir(result_dir): makedirs(result_dir)

        if params.wm_conf is not None and conf.watermarker.name in 'tree-ring':
            if path.isfile(path.join(result_dir, 'keys.jsonl')): remove(path.join(result_dir, 'keys.jsonl'))

        for i, (prompts, keys,id) in enumerate(prompt_dataloader):
                print(f"{i+1}/{len(prompt_dataloader)}")
                if (i+1) == len(prompt_dataloader): difkwargs["latents"] = difkwargs["latents"][:len(prompts)]
                with torch.no_grad():
                    if not wrapped_model:
                        gen_x =  diffusion_model(prompt=list(prompts), output_type='pt', **difkwargs)[0].cpu()
                    else:
                        gen_x =  wm_diffusion_model(x=list(prompts), key=keys.to(device), output_type='pt', **difkwargs)[0].cpu()
                for j in range(len(prompts)):
                    save_image(gen_x[j], path.join(result_dir, id[j] + '.png'))
                    if params.wm_conf is not None and conf.watermarker.name in 'tree-ring':
                        data = {'name' : id[j], 'key': int(keys[j].item())}
                        with open(path.join(result_dir, 'keys.jsonl'), 'a') as f:
                            f.write(json.dumps(data) + "\n")


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Config params')
    aa("--res_dir", type=str, help="Path to the result data directory",required=True)
    aa("--confs", nargs='+', type=str, help="Config files for generation", required=True)
    aa("--wm_conf",  type=str, help="Config file for watermarler", default=None)


    group = parser.add_argument_group('Experimental params')
    aa("--key",  type=str, help="Fix a given string of characters as key", default=None)

    aa("--nsamples", type=int, help="Number of samples to take from dataset",default=None)
    aa("--batch_size", type=int, help="Batch size during diffusion",default=None)
    aa("--conf_path", nargs='+', type=str, help="Main path containing config files", default='guidance-watermarking-for-diffusion-models/configs/')
    aa("--purge_previous", help="Purge previous data in corresponding folder", action=argparse.BooleanOptionalAction)
    return parser



if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main_conf = load_main_config(params)

    # run experiment
    main(params,main_conf)
