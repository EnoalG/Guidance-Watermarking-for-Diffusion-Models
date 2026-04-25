import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F


from ..transforms.transforms import Score

import lpips
from torchmetrics.functional.multimodal import clip_score
from functools import partial
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LPIPS(Score):
    def __init__(self, type='vgg'):
        super().__init__()
        self.type = type
        self.loss =  lpips.LPIPS(net=type).to(device)
    def get_name(self):
        return(f"LPIPS_{self.type}")
    def forward(self, image, image_ref):
       
        score =self.loss(image,image_ref)
        return score

class CLIPScore(Score):
    def __init__(self, model="openai/clip-vit-base-patch16"):
        super().__init__()
        self.model = model
        self.loss =  partial(clip_score, model_name_or_path=model)
    def get_name(self):
        return(f"CLIPScore_{self.model}")
    def forward(self, image, caption_ref):
       
        score =self.loss(image,caption_ref)
        return(score)
class PSNR(Score):
    def __init__(self,**kwargs):
        super().__init__()
    def get_name(self):
        return(f"PSNR")
    def forward(self, image, image_ref):
        bsz = image.shape[0]
        score = -10*torch.log10(torch.mean(image.view(bsz, -1) - image_ref.view(bsz,-1)**2,dim=1))
        return(score)