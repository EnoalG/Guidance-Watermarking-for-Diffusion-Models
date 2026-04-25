
import torch
from torch import nn
from torchvision import transforms as T

from ..detector.detector import Detector
from ..transforms.transforms import TransformSet
from ..watermarking.watermark import Watermarker

class EvalBenchmark(nn.Module):
    def __init__(self, 
                 transform_set: TransformSet,
                 ):
        super(EvalBenchmark, self).__init__()
        self.transform_set = transform_set
    def forward(self, x, keys,**kwargs):
        """
        x: str or Tensor -> Can be either a prompt for seed based watermarker or Tensor images
        """

        raise NotImplementedError
class WatermarkEvalBenchmark(EvalBenchmark):
    def __init__(self, 
                 *args,
                 detector: Detector,
                 watermarker:Watermarker = None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.watermarker = watermarker
        self.detector = detector
    
    def forward(self, x, keys,**kwargs):
        """
        x: str or Tensor -> Can be either a prompt for seed based watermarker or Tensor images
        """
        if self.watermarker is not None:
            im_w = self.watermarker(x, keys, **kwargs)
        else:
            if type(x[0]) is not str: im_w = x
            else: raise ValueError("Prompts cannot be used as input of the detector.")
        im_w = im_w.detach()
        with torch.no_grad():
            aug_im_w = self.transform_set.parallel_forward(im_w.float())
            res = {}
            for t in aug_im_w:
                res[t] = self.detector(aug_im_w[t], keys )
        return(im_w, res)
    
class DetectorEvalBenchmark(EvalBenchmark):
    def __init__(self, 
                 *args, 
                 detector: Detector,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.detector = detector
    
    def forward(self, x, keys,**kwargs):
        """
        x: str or Tensor -> Can be either a prompt for seed based watermarker or Tensor images
        """

        aug_x= self.transform_set.parallel_forward(x.float())
        res = {}
        for t in aug_x:
            res[t] = self.detector(aug_x[t], keys )
        return(x, res)
