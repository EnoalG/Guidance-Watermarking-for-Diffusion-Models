import torch
from torch import nn
from ..detector.detector import Detector
from .losses import LossFunction

import random
class GradientAggregator(nn.Module):
    def __init__(self, *args, detector : Detector, type:str, loss: LossFunction, **kwargs):
          super(GradientAggregator, self).__init__(*args, **kwargs)
          self.detector = detector
          self.type = type
          self.loss = loss

    def noid(self, augmented_data,latents,key):
            misclassified_bits_sum = 0
            latent_dims = tuple(range(1, len(latents.shape)))
            grad = torch.zeros_like(latents)
            for (augmentation_name, augmented_image) in augmented_data.items():
                
                res = self.detector(augmented_image, key = key)
                misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

                mloss =  self.loss(res['message'], res['key'])

                output = -torch.log(mloss)
                output.backward(retain_graph=True)
                if augmentation_name == 'Identity':
                    pass
                else:
                    aug_grad = latents.grad.clone().detach()
                    aug_grad = aug_grad/torch.linalg.vector_norm(aug_grad, dim=latent_dims, keepdim=True)
                    
                    grad += aug_grad
                

                
                latents.grad.zero_()
            grad = grad/(len(augmented_data.keys())-1)

            grad = grad/torch.linalg.vector_norm(grad, dim=latent_dims, keepdim=True)
            return(grad,misclassified_bits_sum )
        

    def cosim(self, augmented_data,latents,key):
            misclassified_bits_sum = 0
            latent_dims = tuple(range(1, len(latents.shape)))
            bsz = latents.shape[0]

            


            grad = torch.zeros_like(latents)
            id_grad = torch.zeros_like(latents)
            for (augmentation_name, augmented_image) in augmented_data.items():
                res = self.detector(augmented_image, key = key)
                misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

                mloss =  bsz*torch.mean(self.loss(res['message'], res['key'])) # Multiply by batch size so that individual gradients are computed correctly

                output = -torch.log(mloss)
                
                output.backward(retain_graph=True)
                if augmentation_name == 'Identity':
                    id_grad = latents.grad.clone().detach()
                    id_grad = id_grad/torch.linalg.vector_norm(id_grad, dim=latent_dims, keepdim=True)
                    grad += id_grad
                else:
                    aug_grad = latents.grad.clone().detach()
                    aug_grad = aug_grad/torch.linalg.vector_norm(aug_grad, dim=latent_dims, keepdim=True)
                    theta = torch.sum(id_grad*aug_grad, dim=latent_dims)

                    grad += (1-theta.view(theta.shape[0], *([1]*latent_dims[-1])))*aug_grad #Reshape theta so it has dims of the latents
                

                
                latents.grad.zero_()
            grad = grad/len(augmented_data.keys())

            return(grad,misclassified_bits_sum )
    def pcgrad(self, augmented_data,latents,key):
        misclassified_bits_sum = 0
        latent_dims = tuple(range(1, len(latents.shape)))
        bsz = latents.shape[0]
        grad = {}
        adjusted_grad = {}
        
        for (augmentation_name, augmented_image) in augmented_data.items():
            res = self.detector(augmented_image, key = key)
            misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

            output =  torch.mean(self.loss(res['message'], res['key'])) 
            output.backward(retain_graph=True)

            # Multiply by batch size so that individual gradients are computed correctly
            grad[augmentation_name] = bsz*latents.grad.clone().detach()
            adjusted_grad[augmentation_name] = bsz*latents.grad.clone().detach()
            latents.grad.zero_()
        #PCGrad: https://arxiv.org/abs/2001.06782

        aug_keys = list(reversed(list(grad.keys())))
        random.shuffle(aug_keys)
        
        for (aug, aug2) in zip(grad.keys(),aug_keys) :
            if aug != aug2:
                cosim = torch.sum(adjusted_grad[aug]*grad[aug2], dim=latent_dims,keepdim=True)/torch.linalg.vector_norm(grad[aug2], dim=latent_dims, keepdim=True)**2
                mask = (cosim < 0).ravel()
                adjusted_grad[aug][mask] -= cosim[mask]*grad[aug2][mask]
        #Final aggregation
        grad = torch.zeros_like(latents)
        for aug in adjusted_grad.keys(): grad +=adjusted_grad[aug]
        grad = grad / len(adjusted_grad.keys())
        
        return(grad, misclassified_bits_sum)
             
                           
    def mean(self, augmented_data,latents,key):
        misclassified_bits_sum = 0
        latent_dims = tuple(range(1, len(latents.shape)))
        bsz = latents.shape[0]
        grad = torch.zeros_like(latents)
        
        for (augmentation_name, augmented_image) in augmented_data.items():
            res = self.detector(augmented_image, key = key)
            misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

           
            output =  torch.mean(self.loss(res['message'], res['key'])) 
            output.backward(retain_graph=True)

            # Multiply by batch size so that individual gradients are computed correctly
            grad += bsz*latents.grad.clone().detach()
            latents.grad.zero_()
        

        grad = grad / len(augmented_data.keys())
        
        return(grad, misclassified_bits_sum)
    
    def hypercone(self, augmented_data,latents,key):
        raise NotImplementedError
        misclassified_bits_sum = 0
        latent_dims = tuple(range(1, len(latents.shape)))
        bsz = latents.shape[0]
        grad = torch.zeros_like(latents)
        
        grad  = torch.zeros_like(key) + key
        grad = grad.to(latents.dtype)
        grad[grad == 0] = -1
        grad = grad/torch.linalg.vector_norm(grad, dim=-1,keepdim=True)#math.sqrt(M)

        with torch.no_grad():
            for (augmentation_name, augmented_image) in augmented_data.items():
                res = self.detector(augmented_image, key = key)
                misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

                
        return(grad, misclassified_bits_sum)

    def median(self, augmented_data,latents,key):
            misclassified_bits_sum = 0
            latent_dims = tuple(range(1, len(latents.shape)))
            bsz = latents.shape[0]
            grad = []
            for (augmentation_name, augmented_image) in augmented_data.items():
                
                res = self.detector(augmented_image, key = key)
                misclassified_bits_sum += (1-res['bit_acc'])*self.detector.M

                output =  torch.mean(self.loss(res['message'], res['key'])) 
                output.backward(retain_graph=True)

                unorm_grad = latents.grad.clone().detach()
                grad.append(bsz*unorm_grad)
                
                latents.grad.zero_()
            grad = torch.stack(grad)
            print(grad.shape)

            grad = torch.median(grad, dim=0).values
            print(grad.shape)

            return(grad, misclassified_bits_sum)
    
    def forward(self,aug_x, latents,key):
        if self.type == 'noid':
            f = self.noid
        elif self.type == 'cosim':
            f = self.cosim
        elif self.type == 'median':
            f = self.median
        elif self.type == 'pcgrad':
            f = self.pcgrad
        elif self.type == 'hypercone':
            f = self.hypercone
        elif self.type == 'mean':
            f = self.mean
        else:
            raise NotImplementedError("Unknown gradient aggregator")
        return(f(aug_x, latents=latents, key=key)) 