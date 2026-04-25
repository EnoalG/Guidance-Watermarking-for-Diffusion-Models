
from datasets import load_dataset
from .benchmarks.datasets import ImageDataset
import torchvision.transforms as T
import torch
from tqdm import tqdm_notebook as tqdm
from os import path,makedirs,listdir,remove
from torch.utils.data import DataLoader
from PIL import Image
import io

from .detector import get_detector_from_conf
import argparse

from .util.util_benchmark import load_main_config,load_model_config
from omegaconf import OmegaConf

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def collate_fn(examples, transform):

    images = []


    for example in examples:
        image =  Image.open(io.BytesIO(example["image"]["bytes"])).convert('RGB')
        image = transform(image)
        images.append(image)


        
    pixel_values = torch.stack(images)


    return  (pixel_values,0,0)

@torch.no_grad
def main(params, conf):

    if 'elsa' in params.blob_path.lower():
        dataset_name = 'd3'
    elif 'flickr' in params.blob_path.lower():
        dataset_name = 'flickr'
    elif 'post-hoc' in params.blob_path.lower():
        dataset_name = 'wm'
    else:
        raise NotImplementedError("Unknown dataset type:", params.blob_path)

    
    detector = get_detector_from_conf(conf.detector,model=None).to(device)
    detector.model = detector.model.to(device)
    print(next(detector.model.parameters()).device)

    im_size = conf.detector.im_size
    blobs = listdir(params.blob_path)





    resize_trans = T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size)])
    collate_fn_trans = lambda x: collate_fn(x, transform = resize_trans )
    save_path = path.join(params.res_dir, dataset_name, conf.detector.name,  str(conf.detector.im_size))
    makedirs(save_path, exist_ok=True)

    if dataset_name == 'wm': blobs = [None]
    for k, blob in enumerate(blobs):
        print(f'{k}/{len(blobs)}', end='\r')
        if path.isfile(path.join(save_path,  f'{k}_0.pt')):continue
        try:
            if dataset_name == 'd3':
                data = load_dataset("parquet", split='train', data_files =path.join(params.blob_path, blob))
                dataloader = DataLoader(data, num_workers=params.num_workers, batch_size=params.batch_size, shuffle=False,collate_fn=collate_fn_trans)

            elif dataset_name == 'flickr':
                data = ImageDataset(path.join(params.blob_path, blob),transform=resize_trans, ext='.jpg',key=None,generate_keys=True,M=conf.detector.M,key_type='bool')
                dataloader = DataLoader(data, num_workers=16, batch_size=100, shuffle=False)
            elif dataset_name == 'wm':
                data = ImageDataset(path.join(params.blob_path),transform=resize_trans, ext='.png',key=None,generate_keys=True,M=conf.detector.M,key_type='bool')
                dataloader = DataLoader(data, num_workers=16, batch_size=100, shuffle=False)

        except:
            print("Could not read blob:", blob)
            continue

        for i, (image,_,_) in enumerate(dataloader):
            xp = detector.preprocess(image.to(device))
            mprime = detector.decode_message(xp)
        

            torch.save(mprime.cpu(), path.join(save_path, f'{k}_{i}.pt'))

    if dataset_name != 'wm':
        xt = []
        ptfiles = listdir(save_path)

        for f in ptfiles:
            if f.endswith('.pt') and not f.startswith('whitener'):
                xt.append(torch.load(path.join(save_path, f),weights_only=False))
        xt = torch.vstack(xt)
        print("[INFO] Samples:", xt.shape)

        mu = torch.mean(xt.T, dim=1).unsqueeze(1)
        C = torch.cov(xt.T -mu)
        L = torch.linalg.inv(torch.linalg.cholesky(C))

        torch.save({'mu' : mu, 'L': L}, path.join(save_path, f'whitener_{xt.shape[0]}_samples.pt'))
def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Config params')
    aa("--res_dir", type=str, help="Path to the result data directory",default='/path/to/guidance-watermarking-for-diffusion-models/tests/__local_data__/guidance_wm_gretsi/whitening/')
    aa("--detector_conf",  type=str, help="Config file for detector", required=True)
    aa("--blob_path", type=str, default='/path/to/huggingface/hub/datasets--elsaEU--ELSA1M_track1/blobs/')

    group = parser.add_argument_group('Experimental params')
    aa("--batch_size", type=int, help="Batch size during diffusion",default=100)
    aa("--num_workers", type=int, help="Number of workers",default=10)


    aa("--conf_path", nargs='+', type=str, help="Main path containing config files", default='guidance-watermarking-for-diffusion-models/configs/')
    aa("--purge_previous", help="Purge previous data in corresponding folder", action=argparse.BooleanOptionalAction)


    return parser
if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    conf = load_main_config(params)
    # run experiment
    main(params,conf)
