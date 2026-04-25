from typing import List, Union, Optional, Dict, Any, Callable
import copy
import numpy as np

import torch
from torch import nn
from torchvision import transforms as T

from ..watermarking.losses import LossFunction
from .aggregation import GradientAggregator

from ..detector.detector import Detector
from ..transforms.transforms import TransformSet

from ..util.util_images import circle_mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import warnings
import lpips
import sys
from io import StringIO

os.environ['LPIPS_VERBOSE'] = '0'
warnings.filterwarnings('ignore')

def lpips_distance(img1, img2):
    """
    Calcule la distance LPIPS entre deux images.
    
    Args:
        img1: première image (tensor PyTorch ou array numpy)
        img2: deuxième image (tensor PyTorch ou array numpy)
    
    Returns:
        float: distance LPIPS (plus faible = plus similaire)
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    loss_fn = lpips.LPIPS(net='alex')
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    if img1.max() <= 1.0:
        img1 = img1 * 2.0 - 1.0
    if img2.max() <= 1.0:
        img2 = img2 * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_value = loss_fn(img1, img2)
    
    return lpips_value.item()

def psnr(img1, img2, max_val=1.0):
    """
    Calcule le PSNR entre deux images.
    
    Args:
        img1: première image (tensor PyTorch ou array numpy)
        img2: deuxième image (tensor PyTorch ou array numpy)
        max_val: valeur maximale possible des pixels (1.0 pour [0,1], 255 pour [0,255])
    
    Returns:
        float: valeur PSNR en dB
    """
    if torch.is_tensor(img1):
        img1 = img1.numpy()
    if torch.is_tensor(img2):
        img2 = img2.numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr_value = 20 * np.log10(max_val / np.sqrt(mse))
    
    return psnr_value

class Watermarker(nn.Module):
    # Placeholder module class for watermarking algorithms
    def __init__(self,*args, **kwargs):
        super(Watermarker, self).__init__()
    
    def forward(self,x,key,**kwargs):
        raise NotImplementedError
    

class VideoSealWatermarker(Watermarker):
    def __init__(self,*args, model,enc_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.enc_size = enc_size
        self.preprocess_transform = T.Compose([
            T.Resize(enc_size), # BILINEAR
            T.CenterCrop(enc_size),
        ])
    def forward(self, x,key):
        im_size =  x.shape[2:]
        key = key.float()
        y = self.preprocess_transform(x)
        y= y* 2.0 - 1.0 


        wm_x = (self.model(y, key) +1.0)/2
        return(wm_x)

class TrustmarkNoECCWatermarker(Watermarker):
    def __init__(self,*args, model,enc_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.enc_size = enc_size
        self.preprocess_transform = T.Compose([
            T.Resize(enc_size,interpolation=T.InterpolationMode.BILINEAR), # BILINEAR
            T.CenterCrop(enc_size),
        ])
    def forward(self, x,key):
        im_size =  x.shape[2:]
        key = key.float()
        y = self.preprocess_transform(x)
        y= y* 2.0 - 1.0 


        stego = self.model(y, key)
        residual = stego.clamp(-1, 1) - y

        residual_mean_c = residual.mean(dim=(2,3), keepdim=True)  # remove color shifts per channel
        residual = (residual - residual_mean_c)

        residual = torch.nn.functional.interpolate(residual, size=(im_size[0], im_size[1]), mode='bilinear')
        wm_x = (torch.clip(residual + (x*2.0-1.0),-1,1)+1)/2

        return(wm_x)
class VAEWatermarker(Watermarker):
    def __init__(self, *args, vae, preprocess_t, postprocess_t, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = vae
        self.preprocess = preprocess_t # Not optimal, but prevents mistakes in defining the preprocess/postprocess op
        self.postprocess = postprocess_t
    def forward(self, x,key=None):
        y = self.preprocess(x)
        z = self.vae.encode(y)
        if z.__class__.__name__ == 'EncoderOutput':
            z = z.latent
        else:
            z = z.latent_dist.mean

        wm_y = self.vae.decode(z).sample
        wm_x = self.postprocess(wm_y)
        return(wm_x)
    
class SeedWatermarker(Watermarker):
    def __init__(self, *args, diffuser,**kwargs):
        self.diffuser = diffuser
        super().__init__(*args,**kwargs)
    def watermark_seed(self,latent, key):
        raise NotImplementedError
    def forward(self, x, key,latents=None, **kwargs):
        #x: prompt
        assert latents is not None
        zT = self.watermark_seed(latents, key)
        x0 = self.diffuser(prompt=x, latents=zT,**kwargs)
        return(x0)


class TreeRingWatermarker(SeedWatermarker):
    def __init__(self, *args, diameter=None, mask_type='ring', num_channels=1, **kwargs):
        super().__init__(*args,**kwargs)
        self.mask_type =mask_type
        self.max_diameter = diameter
        self.num_channels=num_channels

    @staticmethod
    def generate_watermark(latent, key, max_diameter=None, mask_type='ring',num_channels=1):
        if max_diameter is None: max_radius = latent.shape[-1] //2
        else: max_radius = max_diameter //2

        bsz = latent.shape[0]

        generator = torch.Generator(device='cpu') # To allow detection on cpu,since cuda and cpu give different results for same seed
        gt_init = torch.zeros(bsz, *latent.shape[1:],dtype=latent.dtype).to(device)

        for i in range(bsz):
            generator.manual_seed(key[i].item())
            gt_init[i] = torch.randn(latent.shape[1:],device='cpu',
                                        generator=generator,dtype=latent.dtype).to(device)

        latent_w_fft = torch.fft.fftshift(torch.fft.fft2(gt_init))
        
        if 'rand' in mask_type:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in mask_type:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init)) * 0
        elif 'ring' in mask_type:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init))

        for i in range(max_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i, y_offset=-1) # Offset to allow symmetry in the fft, getting real values during fft
            tmp_mask = torch.tensor(tmp_mask)
            
            for j in range(num_channels):
                latent_w_fft[:, j, tmp_mask] = gt_patch[:, j, 0, i].unsqueeze(1)#.item()
                if i == max_radius:
                    latent_w_fft[:, j, ~tmp_mask] = 0
            latent_w_fft[:, num_channels:] = 0
            
        


        # CLEANING UP
        latent_w_fft.imag = 0

        return(latent_w_fft)

    def watermark_seed(self,latent, key):
        wm = TreeRingWatermarker.generate_watermark(latent, key, 
                                                    max_diameter=self.max_diameter, 
                                                    mask_type=self.mask_type,
                                                    num_channels=self.num_channels)
        wm_mask = wm !=0

        latent_w_fft = torch.fft.fftshift(torch.fft.fft2(latent))

        latent_w_fft[wm_mask]= wm[wm_mask]


        latent_w = torch.fft.ifft2(torch.fft.ifftshift(latent_w_fft)).real # No loss of watermark information by taking the real parts since we ensured symetry
        
        latent_w = latent_w.to(latent.dtype)
        return(latent_w)#,wm, latent_w_fft)


    
        
class GuidanceWatermarker(Watermarker):
    """
    A pipeline for generating images using a diffusion model with watermark detection and noise guidance.

    Attributes:
        model (obj): A diffusion model (e.g., Stable Diffusion, FLUX, etc.) for image generation.
        detector (obj): A detector (e.g., a watermarking detector) that analyzes the generated images.
        transforms_set (dict): A dictionary of transformation functions or operations to be applied to the generated latents or images.
        transform_weights (list or tensor): A set of weights associated with each transformation in `transforms_set`.
    """
    def __init__(self, *args, model, detector : Detector, transforms_set : TransformSet,# transform_weights : List[float],
                    loss_type : str = 'dist', loss_threshold : float = 0.1,
                    gradient_aggregator_type : str = 'cosim',
                    wm_scale=0.125,eta=1., vae_bsz=4,top_percent=0.1, max_norm=0.5, **kwargs,
                ):
        """
        Initializes the class with the given parameters.

        Parameters:
            diffusion_model (obj): A diffusion model (e.g., Stable Diffusion, FLUX, etc.) used for image generation. 
            detector (obj): A detector (e.g., a watermarking detector) that analyzes the generated images. 
            transforms_set (list): A list of transformation functions or operations to be applied to the generated latents or images. 
            transform_weights (list or tensor): A set of weights associated with each transformation in `transforms_set`. 

            height (int): Height of the generated image.
            width (int): Width of the generated image.
        """
        #assert(sum(transform_weights) == 1)
        super().__init__(*args, **kwargs)
        self.model = model
        self.detector  = detector
        self.transforms_set  = transforms_set
        #self.transform_weights = transform_weights
        self.loss = LossFunction(loss_type, threshold=loss_threshold)()
        self.gradient_agg = GradientAggregator(detector=self.detector, loss=self.loss, type=gradient_aggregator_type)
        self.wm_scale= wm_scale
        self.top_percent = top_percent
        self.max_norm = max_norm
        print(self.top_percent, self.max_norm)
        self.eta = eta
        self.vae_bsz = vae_bsz

        #self.best_score = self.detector.M*len(transform_weights)
        """
        self.height = height
        self.width = width

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.generator = generator
        """

        

    def step(self, *args, **kwargs):
        """
        Calls the `step` method of the diffusion model.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The output tensor from the diffusion model.
        """
        return self.model.step(*args, **kwargs)
    
    def load_params(self, *args, **kwargs):
        """
        Calls the `load_params` method of the diffusion model.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: The parameters loaded from the diffusion model.
        """
        return self.model.load_params(*args, **kwargs)

    
         
    def update_noise(self, 
                     prompt,
                     height,
                     width,
                     remaining_steps,
                     latents,
                     guidance_scale,
                     generator,
                     noise,
                     cum_alpha_t,
                     step,
                     timesteps,
                     scheduler,
                     **kwargs,
                     ):
        """
        Updates the noise tensor by calculating gradients based on the watermark detector's feedback.

        Parameters:
            noise (torch.Tensor): The current noise tensor.
            cum_alpha_t (torch.Tensor): Cumulative alpha values for diffusion steps.
            step (int): The current diffusion step.
            Other parameters: Various settings for image generation and transformation.

        Returns:
            torch.Tensor: The updated noise tensor.
        """

        raise NotImplementedError("This is the virtual class for GuidanceWatermarker and should not be used.")
    
    def generate(self, 
                    prompt: Union[str, List[str]] = None,
                    key: bool = None,
                    latents: Optional[torch.Tensor] = None,
                    height: int = 512,
                    width: int = 512,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 7.5,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    output_type: str = 'pil',
                    **kwargs,
                ):
                    
                    """
                    Generates images using the diffusion model with watermark detection and guided noise adjustment.

                    Parameters:
                        prompt (str or list): The text prompt for image generation.

                        num_inference_steps (int): Number of diffusion steps.
                        key (str): The watermark key.
                        Other parameters: Various settings for guidance and image generation.

                    Returns:
                        torch.Tensor or PIL.Image: The generated image.
                    """ 
                    
                    print(len(prompt), latents.shape, key.shape)

                    self.key = key # TODO: Thats's kind of bad, key should not be a global variable in the class
                    bsz = len(prompt)


                    self.best_score = [self.detector.M*(len(self.transforms_set.transform_set)+1)]*bsz
                    self.best_latents = torch.zeros_like(latents)
                    with torch.no_grad():
                        # PARAMS
                        params = self.model.load_params(
                                    prompt = prompt,
                                    height = height,
                                    width = width,
                                    num_inference_steps = num_inference_steps,
                                    latents = latents,
                                    guidance_scale = guidance_scale,
                                    generator = generator,
                                    **kwargs,)
                        
                    kwargs.update(params)

                    timesteps = kwargs.pop('timesteps')
                    prompt_embeds = kwargs.pop('prompt_embeds')
                    cum_alpha_t = kwargs.pop('cum_alpha_t')
                    latents = kwargs.pop('latents')

                    scheduler = kwargs.pop('scheduler')
                    
                    
                    self.last_latents = latents
                    self.back_step = False
                    
                    wm_scale = self.wm_scale + 0.0
                    self.best_i = 0

                    for i, t in enumerate(timesteps):
                        self.wm_scale = wm_scale
                        remaining_steps = num_inference_steps - i
                        kwargs.update({'scheduler': scheduler})
                        noise_pred = self.step(latents=latents, t=t, prompt_embeds = prompt_embeds, **kwargs)
                        kwargs.pop('scheduler')
                        
                        # watermarking guidance
                        if hasattr(scheduler, "model_outputs"):
                            scheduler.model_outputs = [
                                x.detach().clone().cpu() if isinstance(x, torch.Tensor) else copy.deepcopy(x)
                                for x in scheduler.model_outputs
                                ]


                        scheduler_to_finish = copy.deepcopy(scheduler)

                        if hasattr(scheduler, "model_outputs") and scheduler.model_outputs is not None:
                            scheduler.model_outputs = [
                                x.to("cuda") if isinstance(x, torch.Tensor) else x
                                for x in scheduler.model_outputs
                                ]

                        noise_pred = self.update_noise(prompt = prompt,
                                                        height = height,
                                                        width = width,
                                                        remaining_steps = remaining_steps,
                                                        latents = latents,
                                                        generator = generator,
                                                        noise = noise_pred, 
                                                        cum_alpha_t = cum_alpha_t, 
                                                        step = i,
                                                        timesteps = timesteps[i:],
                                                        scheduler = scheduler_to_finish,
                                                        **kwargs,
                                                        )         




                        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        
                        del noise_pred
                        torch.cuda.empty_cache()


                    with torch.no_grad():
                        latents = self.best_latents.to(latents.device)

                        return self.model(prompt = prompt,
                                    height = height,
                                    width = width,
                                    num_inference_steps = 0,
                                    latents = latents,
                                    start_first_step = False,
                                    output_type=output_type)
    
    def forward(self, x, key, **kwargs):
        wm_x = self.generate(output_type='pt', prompt=x, key=key, **kwargs)[0]
        return wm_x

class ProbGuidanceWatermarker(GuidanceWatermarker):
     def update_noise(self, 
                     prompt,
                     height,
                     width,
                     remaining_steps,
                     latents,
                     guidance_scale,
                     generator,
                     noise,
                     cum_alpha_t,
                     step,
                     timesteps,
                     scheduler,
                     **kwargs,
                     ):
        """
        Updates the noise tensor by calculating gradients based on the watermark detector's feedback.

        Parameters:
            noise (torch.Tensor): The current noise tensor.
            cum_alpha_t (torch.Tensor): Cumulative alpha values for diffusion steps.
            step (int): The current diffusion step.
            Other parameters: Various settings for image generation and transformation.

        Returns:
            torch.Tensor: The updated noise tensor.
        """

        if hasattr(scheduler, "model_outputs") and scheduler.model_outputs is not None:
            scheduler.model_outputs = [
                x.to("cuda") if isinstance(x, torch.Tensor) else x
                for x in scheduler.model_outputs
                ]

        torch.cuda.empty_cache()
        latents = latents.detach().clone().requires_grad_(True)
        latents_tmp = latents.clone().detach().cpu()
        latent_dims = tuple(range(1, len(latents.shape)))
        output_type = 'latent'
        alpha_t = cum_alpha_t[step]
        final_latent = self.model(prompt = prompt,
                            height = height,
                            width = width,
                            num_inference_steps = remaining_steps,
                            latents = latents,
                            guidance_scale = guidance_scale,
                            generator = generator,
                            timesteps = timesteps.cpu(),
                            output_type = output_type,
                            start_first_step = False,
                            scheduler = scheduler,
                            **kwargs,
                            )
        
        # First compute gradient wrt to Unets
        final_latent = final_latent.images

        final_latent.backward(gradient=torch.ones_like(final_latent))

        print(torch.abs(latents_tmp- latents.detach().cpu()).max())


        dx = latents.grad.clone()

        latents = latents.detach().cpu()

        # Then compute wrt to VAE, most costly in terms of memory
        final_latent = final_latent.detach()#.requires_grad_(True)
        with torch.no_grad():
            image = self.model.vae_decode(final_latent)

        #Finally, we- compute the gradient wrt to the image
        image= image.detach().requires_grad_(True)

        if step == 0:
              self.best_latents = final_latent.detach().clone().cpu()


        
        bsz = latents.shape[0]

        augmented_data = self.transforms_set.parallel_forward(image.float())
        dl,misclassified_bits_sum  = self.gradient_agg(aug_x=augmented_data,latents=image,key=self.key)

        # Efficiently compute the jacobian of the VAE using the vjp (costs 1 more pass through the vae)       
        _ ,dy = torch.autograd.functional.vjp(lambda x : self.model.vae_decode(x, generator), final_latent, v=dl)



        print(f'Number of augmented samples: {len(augmented_data.keys())}, Misclassified bits sum: {misclassified_bits_sum}, Misclassified bits ratio: {misclassified_bits_sum/len(augmented_data.keys())}, Best score: {self.best_score}')
        grad = dx*dy

        for ii in range(bsz):
            if self.best_score[ii] > misclassified_bits_sum[ii]:
                self.best_score[ii] = misclassified_bits_sum[ii]
                self.best_latents[ii] = final_latent[ii].clone().detach().cpu()

        with torch.no_grad():

            grad_norm = torch.linalg.vector_norm(grad,dim=latent_dims, keepdim=True)

            new_grad = []
            for grads in grad:
                k = int(grads.view(-1).shape[0]*self.top_percent)
                top_k_values, _ = torch.topk(torch.abs(grads).view(-1), k)
                threshold = top_k_values[-1]
                grads = grads.clip(min=-threshold, max=threshold)
                new_grad.append(grads)
            grad = torch.stack(new_grad, dim=0)
            
            mask = (grad_norm >= self.max_norm)
            norm_factor = torch.where(mask, self.max_norm/grad_norm, torch.ones_like(grad_norm))
            grad = grad*norm_factor

            grad_norm = torch.linalg.vector_norm(grad,dim=latent_dims, keepdim=True)

            
        if not torch.isnan(grad).any():
            wm_noise =  self.wm_scale*torch.sqrt(1-alpha_t)*grad

            updated_noise = noise - wm_noise

        else:
            updated_noise = noise
        return updated_noise


    
class NormGuidanceWatermarker(GuidanceWatermarker):
    def update_noise(self, 
                     prompt,
                     height,
                     width,
                     remaining_steps,
                     latents,
                     guidance_scale,
                     generator,
                     noise,
                     cum_alpha_t,
                     step,
                     timesteps,
                     scheduler,
                     **kwargs,
                     ):
        """
        Updates the noise tensor by calculating gradients based on the watermark detector's feedback.

        Parameters:
            noise (torch.Tensor): The current noise tensor.
            cum_alpha_t (torch.Tensor): Cumulative alpha values for diffusion steps.
            step (int): The current diffusion step.
            Other parameters: Various settings for image generation and transformation.

        Returns:
            torch.Tensor: The updated noise tensor.
        """

        if hasattr(scheduler, "model_outputs") and scheduler.model_outputs is not None:
            scheduler.model_outputs = [
                x.to("cuda") if isinstance(x, torch.Tensor) else x
                for x in scheduler.model_outputs
                ]

        torch.cuda.empty_cache()
        latents = latents.detach().clone().requires_grad_(True)
        output_type = 'pt'
        alpha_t = cum_alpha_t[step]
        image,final_latent = self.model(prompt = prompt,
                            height = height,
                            width = width,
                            num_inference_steps = remaining_steps,
                            latents = latents,
                            guidance_scale = guidance_scale,
                            generator = generator,
                            timesteps = timesteps.cpu(),
                            output_type = output_type,
                            start_first_step = False,
                            scheduler = scheduler,
                            **kwargs,
                            )
        if step == 0:
              self.best_latents = final_latent.clone().detach().cpu()



        bsz = latents.shape[0]
        latent_dims = tuple(range(1, len(latents.shape)))

        dx =  image.backward(gradient=torch.ones_like(image)) # Compute gradient through VAE only once
        image = image.detach() 
        image.requires_grad = True

        augmented_data = self.transforms_set.parallel_forward(image.float())
        dl,misclassified_bits_sum  = self.gradient_agg(aug_x=augmented_data,latents=image,key=self.key)

        grad = dx *dl
        grad = grad/torch.linalg.vector_norm(grad, dim=latent_dims, keepdim=True)
        print(len(augmented_data.keys()), misclassified_bits_sum, misclassified_bits_sum/len(augmented_data.keys()), self.best_score)


        for ii in range(bsz):
            if self.best_score[ii] > misclassified_bits_sum[ii]:
                self.best_score[ii] = misclassified_bits_sum[ii]
                self.best_latents[ii] = final_latent[ii].clone().detach().cpu()


        if not torch.isnan(grad).any():
            wm_noise = torch.sqrt(1-alpha_t)*grad
            #wm_noise = 1.0*grad
            norm_wm_noise = torch.linalg.vector_norm(wm_noise, dim=latent_dims, keepdim=True)
            wm_noise = wm_noise/norm_wm_noise
            norm_noise = torch.linalg.vector_norm(noise, dim=latent_dims, keepdim=True)
            updated_noise = ((1-self.wm_scale)*noise/norm_noise - self.wm_scale*wm_noise) 
            updated_noise /= torch.linalg.vector_norm(updated_noise, dim=latent_dims, keepdim=True)
            updated_noise *= norm_noise 
        else:
            updated_noise = noise
        return updated_noise
    