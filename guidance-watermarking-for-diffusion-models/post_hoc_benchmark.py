import torch
from omegaconf import OmegaConf
from .transforms import get_transform_from_name
from .benchmarks import get_dataset_from_conf
from .benchmarks.benchmarks import WatermarkEvalBenchmark
from .watermarking import  get_watermarker_from_conf,get_vae_from_name,get_posthoc_encoder, get_diffusion_classname_from_name
from .detector import get_detector_from_conf
from .transforms.transforms import TransformSet
from .tests.test_benchmark import test_benchmark_watermarker_eval

from .util.util_images import identity_t,normalize_vqgan, unnormalize_vqgan
from .util.util_detector import strtobool
from .util.util_benchmark import load_main_config,load_model_config


import numpy as np
from os import path
import argparse
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(params,main_conf):
    for conf_file in params.confs:

        conf = load_model_config(conf_file, params, main_conf)

        attacks = TransformSet([get_transform_from_name(t) for t in conf.transforms]).to(device)

        watermarker_args = {}
        if conf.watermarker.name == 'stable-signature':
            vae,key = get_vae_from_name(conf.model)
            watermarker_args ={'vae': vae,'preprocess_t':identity_t, 'postprocess_t':identity_t}
            key = strtobool(key)
        elif conf.watermarker.name in ['trustmark', 'videoseal']:
            print("[INFO] Loading post-hoc encoder: ", conf.watermarker.name)
            watermarker_args ={'model': get_posthoc_encoder(conf.watermarker.name),'enc_size':256}
            key=None # Random keys
        else:
            raise NotImplementedError("Unknown post-hoc watermarking method")

        mod_class = get_diffusion_classname_from_name(conf.model)
        conf.dataset_ref.path = path.join(conf.dataset_ref.path, mod_class, conf.model, 'cover')
        print('Loading ', conf.dataset.path)
        dataset = get_dataset_from_conf(conf.dataset_ref, im_size=conf.diffusion_params.im_size, key=key,M=conf.detector.M)

        if params.use_dataset_im_size:
            conf.detector.im_size = conf.diffusion_params.im_size
        
        detector = get_detector_from_conf(conf.detector)
        detector = detector.eval()
        detector = detector.to(device)

        watermarker = get_watermarker_from_conf(conf.watermarker, **watermarker_args )

        benchmarker = WatermarkEvalBenchmark(transform_set=attacks.to(device), detector=detector, watermarker=watermarker)
        result_dir = path.join(params.res_dir, conf.dataset.name, mod_class, conf.model, 'post-hoc', conf.watermarker.name)
        with torch.no_grad():
            test_benchmark_watermarker_eval(benchmarker=benchmarker, dataset=dataset,result_dir=result_dir,
                                        difkwargs={},
                                        nsamples=conf.nsamples, batch_size=conf.diffusion_params.batch_size,ext='.png',
                                        metrics = ['pval', 'pval_0bit', 'bit_acc', 'key','dec_message'], purge_old_content=params.purge_previous )


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Config params')
    aa("--res_dir", type=str, help="Path to the result data directory",required=True)
    aa("--confs", nargs='+', type=str, help="Config files for generation", required=True)
    aa("--wm_conf",  type=str, help="Config file for watermarler", required=True)
    aa("--detector_conf",  type=str, help="Config file for detector", required=True)
    aa("--transforms_conf",  type=str, help="Config file for transforms", required=True)


    group = parser.add_argument_group('Experimental params')
    aa("--nsamples", type=int, help="Number of samples to take from dataset",default=None)
    aa("--batch_size", type=int, help="Batch size during diffusion",default=None)
    aa("--conf_path", nargs='+', type=str, help="Main path containing config files", default='guidance-watermarking-for-diffusion-models/configs/')
    aa("--purge_previous", help="Purge previous data in corresponding folder", action=argparse.BooleanOptionalAction)
    aa("--use_dataset_im_size", help="Replace detector image size with dataset imsize", action=argparse.BooleanOptionalAction)


    return parser
if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main_conf = load_main_config(params)

    # run experiment
    main(params,main_conf)
