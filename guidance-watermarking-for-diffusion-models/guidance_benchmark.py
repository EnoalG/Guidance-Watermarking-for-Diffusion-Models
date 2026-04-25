import torch
from omegaconf import OmegaConf
from .transforms import get_transform_from_name
from .benchmarks import get_dataset_from_conf
from .benchmarks.benchmarks import WatermarkEvalBenchmark
from .watermarking import get_diffusion_model_from_name, get_watermarker_from_conf,get_vae_from_name
from .detector import get_detector_from_conf
from .transforms.transforms import TransformSet
from .tests.test_benchmark import test_benchmark_watermarker_eval
import numpy as np
from .util.util_detector import strtobool
from .util.util_benchmark import load_main_config,load_model_config


from os import path,makedirs
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(params,main_conf):
    for conf_file in params.confs:

        conf = load_model_config(conf_file, params, main_conf)
        print("Config:", conf)
        
        
        diffusion_model, difkwargs = get_diffusion_model_from_name(conf.model, conf.diffusion_params)
        diffusion_model = diffusion_model.to(device)
        
        if params.key is not None: key = strtobool(params.key)
        else:   key = None

        if 'vae' in conf.watermarker:
            vae,key = get_vae_from_name(conf.model)
            diffusion_model.vae = vae.to(diffusion_model.dtype)
            key = strtobool(key)
        dataset = get_dataset_from_conf(conf.dataset, key=key, M=conf.detector.M)

        if params.use_dataset_im_size:
            conf.detector.im_size = conf.diffusion_params.im_size

        detector = get_detector_from_conf(conf.detector)
        detector = detector.eval()
        detector = detector.to(device)
        
        if conf.transforms is None:
            attacks = TransformSet([]).to(device)
        else:
            attacks = TransformSet([get_transform_from_name(t) for t in conf.transforms]).to(device)

        watermarker_args = { 
            'model' : diffusion_model,
            'transforms_set' : attacks,
            'detector' : detector,
            'eta' : 1.0,
        }

        for tp, tp_conf in conf.test_params.items():
            print(tp, tp_conf) 
            OmegaConf.update(conf.watermarker, tp, float(tp_conf))
        watermarker = get_watermarker_from_conf(conf.watermarker, **watermarker_args )
        try: 
            if conf.watermarker.vae:
                with_vae = 'with_vae'
            else:
                with_vae = 'without_vae'
        except:
            with_vae = 'without_vae'
        
        num_step_guidance = 25
        if num_step_guidance != 25:
            benchmarker = WatermarkEvalBenchmark(transform_set=attacks.to(device), detector=detector, watermarker=watermarker)
            result_dir = path.join(params.res_dir, conf.dataset.name,  diffusion_model.__class__.__name__, conf.model, 'in-gen',
                                conf.watermarker.name, conf.detector.name, f'{num_step_guidance}_steps' , f'{len(attacks)}_attacks', 'wm_scale_'+ str(conf.watermarker.wm_scale) )
        else:

            benchmarker = WatermarkEvalBenchmark(transform_set=attacks.to(device), detector=detector, watermarker=watermarker)
            result_dir = path.join(params.res_dir, conf.dataset.name,  diffusion_model.__class__.__name__, conf.model, 'in-gen',
                                    conf.watermarker.name, conf.detector.name, f'{len(attacks)}_attacks', 'wm_scale_'+ str(conf.watermarker.wm_scale) )
            
        if not path.isdir(result_dir): makedirs(result_dir)
        with open(path.join(result_dir, "conf.yaml"), 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name) # Save config for replicability
        test_benchmark_watermarker_eval(benchmarker=benchmarker, dataset=dataset,result_dir=result_dir,
                                    difkwargs=difkwargs,
                                    nsamples=conf.nsamples, batch_size=conf.diffusion_params.batch_size,ext='.png',
                                    metrics = ['bit_acc', 'key','pval_0bit'], purge_old_content=True )

        

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
    aa("--test_params_conf",  type=str, help="Config file for parameters to test and their range", required=True)
    aa("--key",  type=str, help="Fix a given string of characters as key", default=None)



    group = parser.add_argument_group('Experimental params')
    aa("--nsamples", type=int, help="Number of samples to take from dataset",default=15)
    aa("--batch_size", type=int, help="Batch size during diffusion",default=3)
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