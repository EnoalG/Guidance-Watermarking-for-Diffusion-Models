import torch
from omegaconf import OmegaConf
from .transforms import get_transform_from_name
from .benchmarks import get_dataset_from_conf
from .benchmarks.benchmarks import WatermarkEvalBenchmark, DetectorEvalBenchmark
from .watermarking import get_vae_key_from_name, get_posthoc_encoder, get_diffusion_classname_from_name
from .detector import get_detector_from_conf
from .transforms.transforms import TransformSet
from .tests.test_benchmark import test_benchmark_watermarker_eval

from .util.util_images import identity_t,normalize_vqgan, unnormalize_vqgan
from .util.util_detector import strtobool
from .util.util_benchmark import load_main_config,load_model_config
from .watermarking import get_diffusion_model_from_name

import numpy as np

from os import path, makedirs
import argparse


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(params,main_conf):
    for conf_file in params.confs:

        conf = load_model_config(conf_file, params, main_conf)
        if conf.transforms is None:
            attacks = TransformSet([]).to(device)
        else:
            attacks = TransformSet([get_transform_from_name(t) for t in conf.transforms]).to(device)

        mod_class = get_diffusion_classname_from_name(conf.model)

        watermarker_args = {}
        diffusion_model = None
        metrics = ['pval_0bit', 'bit_acc', 'key','dec_message']
        if conf.watermarker.name == 'stable-signature':
            key = get_vae_key_from_name(conf.model)
            key = strtobool(key)
            
            conf.dataset_ref.path = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name,'images')
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name, 'results')
        


        elif conf.watermarker.name == 'trustmark':
            conf.dataset_ref.path = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name,'images')
            key = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name, 'key.jsonl')
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name, 'results')

        elif conf.watermarker.name == 'videoseal':
            conf.dataset_ref.path = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name,'images')
            key = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name, 'key.jsonl')
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name, 'results')

        elif conf.watermarker.name == 'vae-watermarker':
            conf.dataset_ref.path = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, conf.watermarker.name,conf.watermarker.vae, 'images')
            key = get_vae_key_from_name(conf.model,conf.watermarker)
            key = strtobool(key)
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, conf.watermarker.name,conf.watermarker.vae,'results')
            
        
        elif conf.watermarker.name == 'tree-ring':
            key = path.join(conf.dataset_ref.path, mod_class, conf.model, conf.detector.name, 'keys.jsonl')
            diffusion_model, difkwargs = get_diffusion_model_from_name(conf.model, conf.diffusion_params)
            diffusion_model = diffusion_model.to(device)
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'in-gen', conf.watermarker.name)
            metrics = ['pval', 'bit_acc']
        elif conf.watermarker.name == 'gaussian-shading':
            key = path.join(conf.dataset_ref.path, mod_class, conf.model, conf.detector.name, 'keys.jsonl')
            diffusion_model, difkwargs = get_diffusion_model_from_name(conf.model, conf.diffusion_params)
            diffusion_model = diffusion_model.to(device)
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'in-gen', conf.watermarker.name)
        elif conf.watermarker.name == 'dummy':
            result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'dummy-post-hoc', conf.watermarker.name)

            key=None # Random keys
        elif conf.watermarker.name == 'prob-guidance':
            conf.dataset_ref.path = path.join(params.res_dir, conf.dataset.name,  mod_class, conf.model, 'in-gen',
                                    conf.watermarker.name, conf.detector.name, params.opt_param, 'images')
            print(conf.dataset_ref.path )
            key = path.join(conf.dataset_ref.path, '../key.jsonl')
            result_dir = path.join(params.res_dir, conf.dataset.name,  mod_class, conf.model, 'in-gen',
                                    conf.watermarker.name, conf.detector.name, params.opt_param, 'results/')

        else:
            raise NotImplementedError("Unknown watermarking method")

        if  (not 'guidance' in conf.watermarker.name) and (not 'trustmark' in conf.watermarker.name) and (not 'stable-signature'in conf.watermarker.name) and (not 'videoseal'in conf.watermarker.name) and (not 'vae-watermarker' in conf.watermarker.name):
            if params.override_datapath is None  :
                conf.dataset_ref.path = path.join(conf.dataset_ref.path, mod_class, conf.model, conf.detector.name)
            elif params.override_datapath is not None:
                conf.dataset_ref.path = params.override_datapath

        if params.override_datapath is not None:
            conf.dataset_ref.path = params.override_datapath
        print(conf.dataset_ref)
        # conf.diffusion_params.im_size = 1024
        dataset = get_dataset_from_conf(conf.dataset_ref, im_size=conf.diffusion_params.im_size, key=key,M=conf.detector.M, key_type=conf.detector.key_type)
        
        if params.use_dataset_im_size:
            conf.detector.im_size = conf.diffusion_params.im_size

        detector = get_detector_from_conf(conf.detector,model=diffusion_model)
        detector = detector.eval()
        detector = detector.to(device)


        benchmarker = DetectorEvalBenchmark(transform_set=attacks, detector=detector)
        if not path.isdir: makedirs(result_dir)
        with torch.no_grad():
            test_benchmark_watermarker_eval(benchmarker=benchmarker, dataset=dataset,result_dir=result_dir,
                                        difkwargs={},
                                        nsamples=conf.nsamples, batch_size=conf.diffusion_params.batch_size,ext='.png',
                                        metrics = metrics, purge_old_content=True, genimage=False )

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Config params')
    aa("--res_dir", type=str, help="Path to the result data directory",required=True)
    aa("--confs", nargs='+', type=str, help="Config files for generation", required=True)
    aa("--wm_conf",  type=str, help="Config file for watermarker", required=True)
    aa("--detector_conf",  type=str, help="Config file for detector", required=True)
    aa("--transforms_conf",  type=str, help="Config file for transforms", required=True)
    aa("--data_conf",  type=str, help="Config file for datasets (prompts or images)", default='mscoco_train2014.yaml')



    group = parser.add_argument_group('Experimental params')
    aa("--nsamples", type=int, help="Number of samples to take from dataset",default=None)
    aa("--batch_size", type=int, help="Batch size during diffusion",default=48)
    aa("--conf_path", nargs='+', type=str, help="Main path containing config files", default='guidance-watermarking-for-diffusion-models/configs/')
    aa("--purge_previous", help="Purge previous data in corresponding folder", action=argparse.BooleanOptionalAction)
    aa("--use_dataset_im_size", help="Replace detector image size with dataset imsize", action=argparse.BooleanOptionalAction)
    aa("--override_datapath", type=str, help="Disregard data path in conf and use this one instead", default=None )
    aa("--opt_param", type=str, help="Specify a test param folder", default=None )


    return parser

def in_hook():
    parser = get_parser()
    params = parser.parse_args()
    main_conf = load_main_config(params)
    main(params,main_conf)

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main_conf = load_main_config(params)

    # run experiment
    main(params,main_conf)