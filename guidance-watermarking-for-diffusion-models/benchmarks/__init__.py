from .datasets import *
from os import path
import torchvision.transforms as T

def get_dataset_from_conf(conf, im_size=None, key=None,M=None, key_type='bool'):
    data_path = conf.path
    if conf.type == 'PROMPT':
        if key is None:
            dataset = PromptDataset(data_path,generate_keys=True,M=M,key_type=key_type)
        else:
            dataset = PromptDataset(data_path,key=key,key_type=key_type)
    if conf.type == 'IMAGE':
        assert im_size is not None
        im_dir = path.join(data_path)
        if key is None: 
            dataset = ImageDataset(im_dir,generate_keys=True,transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size)]),M=M,key_type=key_type)
        else:
            dataset = ImageDataset(im_dir,key=key,transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size) ]),M=M,key_type=key_type)

    return(dataset)
