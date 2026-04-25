import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from kornia import filters as kfilters
from kornia.augmentation import RandomJPEG
from kornia.enhance import JPEGCodecDifferentiable


from ..transforms.transforms import Transform
from augly.image import functional as aug_functional # For stable signature compatibility

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class JPEG(Transform):
    def __init__(self, quality_factor, compressor='kornia'):
        super().__init__()
        self.quality_factor = quality_factor
        self.compressor_type = compressor
        if compressor == 'kornia':
            self.compressor = lambda x:  JPEGCodecDifferentiable()(x, jpeg_quality=torch.tensor([quality_factor]*x.shape[0]).to(device))#RandomJPEG(jpeg_quality=torch.tensor(quality_factor).to(device), p=1.) #To device is ugly but i don't know how to fix the "not on same device" bug otherwise...
        elif compressor == 'augly': # Not differentiable
            self.compressor = lambda x : self.augly_jpeg_compress(x, quality_factor=quality_factor)
        else:
            raise NotImplementedError("Unknown JPEG compression algorithm")
    
    def get_name(self):
        return(f"JPEG_{self.quality_factor}_{self.compressor_type}")
    def augly_jpeg_compress(self, x, quality_factor):
        """ Apply jpeg compression to image
        Args:
            x: PIL image
            quality_factor: quality factor
        """
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        img_aug = torch.zeros_like(x, device=x.device)
        for ii,img in enumerate(x):
            pil_img = to_pil(img)
            img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
        return img_aug
    def forward(self, image):

        aug_image =self.compressor(image.float())
        return aug_image


class GaussianBlur(Transform):
    def __init__(self, kernel_size=(3,3),sigma=(1.0,1.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma=sigma
    def get_name(self):
        return(f"GaussianBlur_{self.kernel_size[0]}_{self.kernel_size[1]}")
    def forward(self, image):
        aug_image =kfilters.gaussian_blur2d(image.float(), self.kernel_size, self.sigma)
        return aug_image
    
class GaussianNoise(Transform):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std/255.0
    def get_name(self):
        return(f"GaussianNoise_{self.mean}_{self.std}")
    def forward(self, image):
        noise = torch.randn_like(image) * self.std + self.mean
        aug_image = image + noise
        return aug_image.clamp(0, 1)


class MedianFilter(Transform):
    def __init__(self, kernel_size=(3,3)):
        super().__init__()
        self.kernel_size=kernel_size
    def get_name(self):
        return(f"MedianFilter_{self.kernel_size[0]}_{self.kernel_size[1]}")
    def forward(self, image,):
       
        aug_image =kfilters.median_blur(image.float(), self.kernel_size)
        return aug_image


class Brightness(Transform):
    def __init__(self,bias):
        super().__init__()
        self.bias=bias
    def get_name(self):
        return(f"Brightness_{self.bias}")
    def forward(self, image):
        
        aug_image = image + self.bias
        return aug_image


class Contrast(Transform):
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = factor
    def get_name(self):
        return(f"Contrast_{self.factor}")
    def forward(self, image):
        aug_image =F.adjust_contrast(image, self.factor)
        return aug_image


class Saturation(Transform):
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = factor
    def get_name(self):
        return(f"Saturation_{self.factor}")
    def forward(self, image,):
        aug_image =F.adjust_saturation(image, self.factor)
        return aug_image


class Hue(Transform):
    def __init__(self, factor= 1.):
        super().__init__()
        self.factor = factor
    def get_name(self):
        return(f"Hue_{self.factor}")
    def forward(self, image):
        aug_image =F.adjust_hue(image, self.factor)
        return aug_image