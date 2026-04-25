import torch
from omegaconf import OmegaConf
from .transforms import get_score_from_name
from .benchmarks.datasets import PairedRefDataset
from .transforms.transforms import ScoreSet
import numpy as np
from torch.utils.data import DataLoader,Subset
from .util.util_benchmark import load_main_config,load_model_config
from.watermarking import get_diffusion_classname_from_name

from os import path, makedirs,remove,listdir
import json

import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(params,main_conf):
    for conf_file in params.confs:
        conf = load_model_config(conf_file, params, main_conf)
        scores = ScoreSet([get_score_from_name(s) for s in conf.scores]).to(device)
        conf.dataset.type = 'IMAGE'

        for tp, tp_conf in conf.test_params.items():
            if tp_conf.end == tp_conf.start:
                param_to_test = np.array([tp_conf.start])
            else:
                param_to_test = np.arange(tp_conf.start, tp_conf.end, tp_conf.step)  
            print(param_to_test)
            for p in param_to_test:
                current_param = str(tp) + '_'+ str(p)
                print(current_param)

                mod_class = get_diffusion_classname_from_name(conf.model)
                ref_path = path.join(conf.dataset_ref.path, mod_class, conf.model, 'cover')
                

                data_dir = path.join(params.res_dir, mod_class,conf.model, params.dataset_dir, current_param , 'images')
                print(ref_path ,data_dir)

                if not path.isdir(data_dir): continue
                actual_nsamples = len(listdir(data_dir ))
                if actual_nsamples == 0: continue
                elif actual_nsamples < conf.nsamples: nsamples = actual_nsamples
                else: nsamples = conf.nsamples
    
                dataset = PairedRefDataset(dirs = (data_dir, ref_path),
                                        types=(conf.dataset.type, conf.dataset_ref.type),
                                        exts = ('.png', '.png')
                )



                subdataset = Subset(dataset, range(nsamples))
                paired_dataloader = DataLoader(subdataset, batch_size=params.batch_size, shuffle=False)


                jsonfile = 'quality.jsonl'
                result_path = path.join(data_dir, '../', jsonfile)
                if params.purge_previous and path.isfile(result_path): remove(result_path)

                with torch.no_grad():
                    for i, (x, x_ref,ids) in enumerate(paired_dataloader):
                        print(f"{i+1}/{len(paired_dataloader)}")
                        
                        res = scores(x.to(device), x_ref.to(device))

                        for j, id in enumerate(ids):
                            data = {'id' : id }
                            for metric in res:
                                data[metric] = float(res[metric][j].cpu())
                            with open(result_path, 'a') as f:
                                f.write(json.dumps(data) + "\n")



       

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Config params')
    aa("--confs", nargs='+', type=str, help="Config files for generation", required=True)
    aa("--transforms_conf",  type=str, help="Config file for scores", required=True)
    aa("--dataset_dir",  type=str, help="Main dataset directory", required=True)
    aa("--res_dir",  type=str, help="Main benchmark directory", required=True)

    aa("--test_params_conf",  type=str, help="Config file for parameters to test and their range", required=True)


    group = parser.add_argument_group('Experimental params')
    aa("--nsamples", type=int, help="Number of samples to take from dataset",default=None)

    aa("--batch_size", type=int, help="Batch size during diffusion",default=50)
    aa("--conf_path", nargs='+', type=str, help="Main path containing config files", default='guidance-watermarking-for-diffusion-models/configs/')
    aa("--purge_previous", help="Purge previous data in corresponding folder", action=argparse.BooleanOptionalAction)

    return parser
if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main_conf = load_main_config(params)

    # run experiment
    main(params,main_conf)