
from os import path,listdir, makedirs,remove

from ..benchmarks.benchmarks import EvalBenchmark
from ..benchmarks.datasets import PromptDataset,WmDataset
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision.utils import save_image

from ..watermarking.modified_diffusers.stable_diffusion import sd2_config
from ..watermarking.watermark import GuidanceWatermarker

from ..transforms.transforms import TransformSet
from ..transforms.geometric import *
from ..transforms.valuemetric import *

from ..detector.detector import StableSignatureDetector, TrustmarkDetector 

from . import MODELDIR,MSCOCO,BENCHMARK_DIR
import json
import torch

from ..util.util_detector import booltostr

def test_benchmark_watermarker_eval(benchmarker: EvalBenchmark, dataset:Dataset, result_dir:str,
                                    difkwargs=None,
                                    nsamples:int=10, batch_size=3,ext='.png',
                                    metrics = ['pval', 'bit_acc', 'key'], purge_old_content=False, genimage=True ):
     
     
    result_image_dir = path.join(result_dir, 'images')
    jsonfiles = {metric : path.join(result_dir, f'{metric}.jsonl') for metric in metrics}
    if not path.isdir(result_image_dir): 
        makedirs(result_image_dir)
    elif purge_old_content: 
        print("Purging files in ", result_dir, " with extension ", ext)
        for jsonfile in jsonfiles.values(): 
            if path.isfile(jsonfile): remove(jsonfile)
        old_ims = [path.join(result_image_dir, f) for f in listdir(result_image_dir) if f.endswith(ext)]
        for f in old_ims: remove(f)


    subdataset = Subset(dataset, range(nsamples))
    data_loader = DataLoader(subdataset, batch_size=batch_size, shuffle=False)

    for i, (xs, keys, imnames) in enumerate(data_loader):
        print(f"{i}/{len(data_loader)}")
        if difkwargs and (i+1) == len(data_loader): 
            tmp_latents = difkwargs["latents"].cpu()
            difkwargs["latents"] = difkwargs["latents"][:len(xs)]
        if type(xs[0]) is str :
            wm_ims, res = benchmarker(list(xs),keys.to(device),**difkwargs)
        else:
            wm_ims, res = benchmarker(xs.to(device),keys.to(device),**difkwargs)
        for j in range(len(wm_ims)):
            outname = imnames[j].split('.')[0]
            if genimage: save_image(wm_ims[j], path.join(result_image_dir, outname  +  ext)) 
            for metric in metrics:
                data = {'id' : outname }
                for atk in res:
                    d = res[atk][metric][j]
                    if metric != "pval" and metric != 'pval_0bit': d=d.cpu().numpy()
                    if metric == 'key' or  metric=='dec_message':
                        d = booltostr(d)
                    else: d = float(d)


                    data[atk] = d
                with open(jsonfiles[metric], 'a') as f:
                    f.write(json.dumps(data) + "\n")
        if difkwargs and (i+1) == len(data_loader): difkwargs["latents"] = tmp_latents.to(device)

def test_sd2_benchmark(wm_scale, eta=1.,num_inference_steps=25, im_size=512, seed=43,batch_size=3,nsamples=10):
    print(f"\t Testing Stable Signature guidance on SD2 -- Seed {seed}")

    # Diffusion model
    diffusion_model, difkwargs = sd2_config(num_inference_steps, im_size, seed,batch_size)

    # Detector
    M=48
    msg_extractor = torch.jit.load( path.join(MODELDIR ,'dec_48b_whit.torchscript.pt'), map_location=device).to(device)
    detector = StableSignatureDetector( model=msg_extractor,M=M,im_size=difkwargs['width'])

    # Attacks

    
    attacks = TransformSet([
        CenterCrop(0.5),
        Brightness(0.2),
        Contrast(2.),
        JPEG(50, 'kornia'),
        JPEG(80, 'kornia'),
    ]
    )

    weights = [1]*(len(attacks.transform_set)+1)

    guidance_wm = GuidanceWatermarker(diffusion_model, detector=detector,
                                       transforms_set=attacks, transform_weights=weights,
                                       wm_scale=wm_scale, eta=eta)
    
    benchmarker = WatermarkEvalBenchmark(transform_set=attacks.to(device), detector=detector, watermarker=guidance_wm)
    dataset = PromptDataset(MSCOCO)
    test_benchmark_watermarker_eval(benchmarker=benchmarker,dataset=dataset,
                                     result_dir=path.join(BENCHMARK_DIR, diffusion_model.__class__.__name__, detector.__class__.__name__,guidance_wm.__class__.__name__, str(guidance_wm.wm_scale) ),
                                     difkwargs=difkwargs,batch_size=batch_size,nsamples=nsamples,purge_old_content=False)
def main():



    wm_scale=0.2
    print("--- Testing Benchmarks ---")
    test_sd2_benchmark(wm_scale=wm_scale, eta=1.,num_inference_steps=25, im_size=512, seed=43,batch_size=4,nsamples=50)
    




    


if __name__ == "__main__":
     main()