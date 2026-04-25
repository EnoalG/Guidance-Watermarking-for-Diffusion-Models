import torchvision.transforms as T


from ..transforms.transforms import Transform


class VAEAttack(Transform):
    def __init__(self,vae, vae_name:str):
        super(VAEAttack, self).__init__()
        self.vae=vae
        self.vae_name = vae_name
    def get_name(self):
        return(f"VAEAttack_{self.vae_name}")
    def forward(self, image):
        z = self.vae.encode(image).latent_dist.mean
        aug_image= self.vae.decode(z).sample
        return aug_image
    

class Purification(Transform):
    def __init__(self,purificator, timestep):
        super(VAEAttack, self).__init__()
        self.purificator=purificator
        self.timestep= timestep # Might directly put in in the purificator class
    def get_name(self):
        return(f"Purification_{self.purificator.__class__.__name__}_{self.timestep}")
    def forward(self, image):
        aug_image= self.purificator(image, self.timestep)
        return aug_image
    
class InTheSandAttack(Transform):
    def __init__(self,model, score_oracle):
        super(VAEAttack, self).__init__()
        self.model = model
        self.score_oracle= self.score_oracle
    def get_name(self):
        return(f"Purification_{self.model.__class__.__name__}_{self.score_oracle.__class__.__name__}")
    def forward(self, image):
        raise NotImplementedError
        return aug_image

