import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
from scipy.special import betainc, betaincc # Should ultimately be transfered to the upcoming implementation in pytorch
from scipy.stats import ncx2
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Detector(nn.Module):
    def __init__(self, 
                 model,
                 M: int,
                 whitener= None, # dict with the inverse cholesky decomposition L of the covariance and a mean mu
                 **kwargs):
        super(Detector, self).__init__()
        self.model = model
        self.M = M # Size of the message in bits
        if whitener is not None:
            self.L = whitener['L'].to(device)
            self.mu = whitener['mu'].to(device)
        else:
            self.L = None
            self.mu = None
            
    def decode_message(self,x):
        y = self.model(x)

        if self.L is not None: 
            y = torch.matmul(self.L, y.T -self.mu ).T # Whitening to ensure uniformity under H0
        return(y)
        
    def process_message(self,mprime):
        return(mprime>0)
        
    def bit_acc(self, mprime,m):
        mdec = self.process_message(mprime)
        return(torch.sum(mdec == m,dim=-1)/self.M)
    @torch.no_grad() 
    
    def pval(self, mprime, m):
        mdec = self.process_message(mprime)
        h = torch.sum(mdec == m,dim=-1).cpu().numpy()
        return(betainc(h, self.M-h +1,0.5 ))
    
    def pval_0bit(self, mprime, m):
        key = m.to(torch.float).clone()
        key[key == 0] = -1
        s = F.cosine_similarity(key, mprime, dim=1).to(torch.float64).detach().cpu().numpy()
        pval = np.zeros_like(s)
        
        # s > 0
        pos_mask = s > 0
        pval[pos_mask] = 0.5 * betaincc(0.5, (self.M - 1) / 2, s[pos_mask]**2)

        # s < 0
        neg_mask = s < 0
        pval[neg_mask] = 0.5 * (1 + betainc(0.5, (self.M - 1) / 2, s[neg_mask]**2))

        return pval
    
    def preprocess(self,x):
        raise NotImplementedError
        
    def forward(self,x, key=None):
        xp = self.preprocess(x)
        mprime = self.decode_message(xp)


        pval = None
        bit_acc = None
        if key is not None: 
            pval = self.pval(mprime, key)
            bit_acc = self.bit_acc(mprime,key)
            pval_0bit = self.pval_0bit(mprime, key)
        return({'message':mprime,
                'dec_message': self.process_message(mprime),
                'pval': pval,
                'pval_0bit' : pval_0bit,
                'bit_acc': bit_acc,
                'key':key})
    


class StableSignatureDetector(Detector):
    def __init__(self, *args, im_size=256, **kwargs):
        super().__init__(*args, **kwargs)   
        self.im_size = im_size
        self.preprocess_transform = T.Compose([
            T.Resize(im_size),
            T.CenterCrop(im_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self,x):
        return(self.preprocess_transform(x))
    

class VideoSealDetector(Detector):
    def __init__(self, *args, im_size=256, **kwargs):
        super().__init__(*args, **kwargs)   
        self.im_size = im_size
        self.preprocess_transform = T.Compose([
            T.Resize(im_size,interpolation=T.InterpolationMode.BILINEAR), # BILINEAR
            T.CenterCrop(im_size),
        ]) # We assume aspect ratio <= 2.0, otherwise Trustmark does another transform


    def preprocess(self,x):
        x= self.preprocess_transform(x)
        x= x* 2.0 - 1.0 
        return(x)
    
class TrustmarkDetector(Detector):
    def __init__(self, *args, im_size=245, **kwargs):
        super().__init__(*args, **kwargs)   
        self.im_size = im_size
        self.preprocess_transform = T.Compose([
            T.Resize(im_size,interpolation=T.InterpolationMode.BILINEAR), # BILINEAR
            T.CenterCrop(im_size),
        ]) # We assume aspect ratio <= 2.0, otherwise Trustmark does another transform


    def preprocess(self,x):
        x= self.preprocess_transform(x)
        x= x* 2.0 - 1.0 
        return(x)

class TreeRingDetector(Detector):
    def __init__(self, *args, diameter=None, mask_type='ring', num_channels=1,**kwargs):
        
        super().__init__(*args,**kwargs)
        self.mask_type =mask_type
        self.max_diameter = diameter
        self.num_channels=num_channels
        

    def preprocess(self,x,prompt=None):
        if prompt is None:
            prompt = [""]*x.shape[0]
        return(self.model(prompt=prompt, latent=x, output_type='latent',num_inference_steps=50).images)
    

    def decode_message(self,x):
        return( torch.fft.fftshift(torch.fft.fft2(x)))
    def process_message(self,mprime):
        return(mprime)
    @torch.no_grad
    def pval(self, mprime, m,tol=1e-6):
        from ..watermarking.watermark import TreeRingWatermarker
        bsz = mprime.shape[0]
        orth = torch.sqrt(torch.tensor(mprime.shape[-1] * mprime.shape[-2]))
        latent_w_fft = mprime/orth 



        wm = TreeRingWatermarker.generate_watermark(latent_w_fft, m, 
                                                    max_diameter =self.max_diameter,
                                                    mask_type=self.mask_type,
                                                    num_channels=self.num_channels)
        wm_mask = wm !=0
 

        
        wm /= orth
        df = torch.sum(wm_mask.view(bsz, -1),dim=-1) # Number of COMPLEX numbers
        print(wm.shape,wm_mask.shape)


        mu_sq = torch.sum(wm[wm_mask].view(bsz, -1).real**2, dim=-1)
        score = torch.sum(torch.abs((latent_w_fft[ wm_mask]- wm[wm_mask])**2).view(bsz, -1), dim=-1)
        score[score < tol] = tol
        pval = ncx2.cdf(score.cpu().numpy(), df=df.cpu().numpy(),nc = mu_sq.cpu().numpy())
        return(score, pval)
    

    def forward(self,x, key,prompt=None):
        xp = self.preprocess(x,prompt)
        mprime = self.decode_message(xp)
        score, pval = self.pval(mprime, key)
        return({'message':torch.tensor([-1]),
                'dec_message': torch.tensor([-1]),
                'pval': pval,
                'bit_acc': score,
                'key':key})


def main():
    pass


if __name__ == "__main__":
    main()
