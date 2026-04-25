import torchvision.transforms.functional as F
import torchvision.transforms as T


from ..transforms.transforms import Transform


class Rotate(Transform):
    def __init__(self,angle=90):
        super(Rotate, self).__init__()
        self.angle=angle
    def get_name(self):
        return(f"Rotate_{self.angle}")
    def forward(self, image):
        aug_image = F.rotate(image, self.angle)
        
        return aug_image


class Resize(Transform):
    def __init__(self, factor):
        super(Resize, self).__init__()
        self.factor = factor

    def get_name(self):
        return(f"Resize_{self.factor}")
    def forward(self, image):
        h, w = image.shape[-2:]
        output_size = (int(self.factor* h), int(self.factor * w))
        aug_image = F.resize(image, output_size, antialias = True,interpolation=T.InterpolationMode.BILINEAR)
        
        return aug_image


class CenterCrop(Transform):
    def __init__(self, factor):
       super(CenterCrop, self).__init__()
       self.factor = factor
    def get_name(self):
        return(f"CenterCrop_{self.factor}")

    def forward(self, image):
        h, w = image.shape[-2:]
     
        output_size = (int(self.factor  * h), int(self.factor  * w))
        aug_image = F.center_crop(image, output_size)
        
        return aug_image

class RandomCrop(Transform):
    def __init__(self, factor):
       super(RandomCrop, self).__init__()
       self.factor = factor
    def get_name(self):
        return(f"RandomCrop_{self.factor}")

    def forward(self, image):
        h, w = image.shape[-2:]
     
        output_size = (int(self.factor * h), int(self.factor  * w))

        i, j, h, w = T.RandomCrop.get_params(image, output_size=output_size)
        aug_image = F.crop(image, i, j, h, w)
        
        return aug_image


class HorizontalFlip(Transform):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def get_name(self):
        return(f"HorizontalFlip")
    def forward(self, image):
        aug_image = F.hflip(image)
        return aug_image


