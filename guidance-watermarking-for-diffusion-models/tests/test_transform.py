from os import path
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image, make_grid

from ..transforms.valuemetric import *
from ..transforms.geometric import *

from ..transforms.transforms import TransformSet

from ..util.util_images import load_image

from . import MODELDIR, IMTESTDIR,LOCALDATA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_all_valuemetrics(im):
    valuemetric_set= [
        Brightness(0.5),
        Contrast(1.5),
        Saturation(1.5),
        Hue(0.5),
        JPEG(50, 'kornia'),
        JPEG(50, 'augly'),
        GaussianBlur((9,9),sigma=(1.5,1.5)),
        MedianFilter((5,5)),
        TransformSet([Hue(0.25), GaussianBlur((5,5), sigma=(1.2,1.2)), JPEG(80)])
        
    ]

    value_trans = TransformSet(valuemetric_set).to(device)
    aug_x = value_trans.parallel_forward(im)
    print(f"Transforms: {list(aug_x.keys())}")

    for k in aug_x:
        save_image(aug_x[k], path.join(LOCALDATA, f'valuemetrics_{k}.png')) 


def test_all_geometrics(im):
    valuemetric_set= [
        Rotate(45),
        Resize(0.25),
        CenterCrop(0.5),
        RandomCrop(0.1),
        HorizontalFlip(),
        TransformSet([Rotate(90), Resize(0.80), CenterCrop(0.7), Rotate(-45), HorizontalFlip(), RandomCrop(0.4)])
        
    ]

    value_trans = TransformSet(valuemetric_set).to(device)
    aug_x = value_trans.parallel_forward(im)
    print(f"Transforms: {list(aug_x.keys())}")

    for k in aug_x:
        save_image(aug_x[k], path.join(LOCALDATA, f'valuemetrics_{k}.png')) 


def main():
    im_path = path.join(IMTESTDIR ,'1.jpg')
    transform = T.Compose([T.ToTensor()])
    im = transform(load_image(im_path)).unsqueeze(0).to(device)
    print("1. Testing valuemetrics")
    test_all_valuemetrics(im)
    print("2. Testing geometrics")
    test_all_geometrics(im)

if __name__ == "__main__":
     main()
    