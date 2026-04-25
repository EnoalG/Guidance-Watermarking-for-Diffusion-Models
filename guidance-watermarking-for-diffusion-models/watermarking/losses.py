import torch
import torch.nn.functional as F
import math
class LossFunction:
    def __init__(self,loss_type:str, threshold:float=0.1, gamma:float=10.0):
        self.loss_type = loss_type
        self.threshold = threshold
        self.gamma = gamma

    def mvg_pval(self, m, m_prime):
        """
        P-value assuming gaussianity under H0
        Assumes the detector has been whitened ! 
        """
        M= m_prime.shape[-1]
        mask = (m_prime ==0).to(float)

        diff = m_prime != (m>0)
        h = M- torch.sum(diff,dim=-1)

        #

        
        with torch.no_grad():
            if torch.sum(diff) !=0:
                 worse_dec = torch.max(m[diff].abs(),dim=-1)
                 tau = worse_dec.values #TODO: Works with identity only but not when combined with other transforms
                 tau[tau > self.threshold] = self.threshold
            else: 
                tau = self.threshold
        
        m[~diff] = torch.clip(m[~diff], -tau, tau)
        nm = torch.sum(m*(1-mask) -m*(mask), dim=-1)/math.sqrt(M)

        sf = 0.5*(torch.erfc(nm/math.sqrt(2))) # Gaussian sf, erfc has better precision
        lpval =  -torch.log10(sf)# Still uniform under H0 thanks to gaussian symmetry

        return(lpval)
    def poly_betainc_prec(self,m,m_prime):
        M= m_prime.shape[-1]
        nm = torch.sigmoid(self.threshold*m)
        h = 1- torch.sum(torch.abs(m_prime.float() - nm.float()),dim=-1)/M

        """
        with torch.no_grad():
            print(torch.max(m, dim=-1).values, torch.min(m, dim=-1).values)

            print(torch.max(nm, dim=-1).values, torch.min(nm, dim=-1).values)
            print(h)
        """
        coefs = torch.tensor([ 0.28950812,  3.3101934 , 11.83071497, 14.80796954,  0.791596  , -8.32291843, -1.45621417,  2.63563475,  3.47601597,  2.532678  ]) # Polynomial approcimation of degree 9 of -logbetainc
        
        x = 2*h -1.0 
        x = x.unsqueeze(1)
        pw = torch.arange(0, coefs.shape[0]).to(x.device)
        coefs = coefs.unsqueeze(0).to(x.device)
        lpval =  torch.sum(coefs * torch.pow(x,pw) ,dim=-1) 
        return(lpval)
        
    def poly_betainc(self, m, m_prime):
        M= m_prime.shape[-1]
        nm = torch.sigmoid(self.threshold*m)
        h = 1- torch.sum(torch.abs(m_prime.float() - nm.float()),dim=-1)/M
        """
        with torch.no_grad():
            print(torch.max(m, dim=-1).values, torch.min(m, dim=-1).values)

            print(torch.max(nm, dim=-1).values, torch.min(nm, dim=-1).values)
            print(h)
        """

        x = 2*h -1.0 
        lpval =  7.39082232*x + 21.66892571*x**3 # Polynomial approcimation of degree 2 of odd -logbetainc
        return(lpval)
        
    def cossim(self, m, m_prime):

        M= m_prime.shape[-1]
        x = m/torch.linalg.vector_norm(m, dim=-1,keepdim=True)
        mt = torch.zeros_like(m_prime) + m_prime
        mt = mt.float()

        
        with torch.no_grad():
            mt[mt == 0] = -1
            y = mt/torch.linalg.vector_norm(mt, dim=-1,keepdim=True)
        loss = torch.sum(x*y, dim=-1)
        diff = (y>0) != (m>0)
        return(loss)


    def dist(self,m, m_prime):

        m_dec = m>0
        M = m.shape[-1]

        diff_mask = (m_dec== m_prime) & (torch.abs(m) < self.threshold)

        diff_mask2 = (m_dec!= m_prime)
        l1 = (torch.abs(diff_mask2*m) + diff_mask2*self.threshold)**2
        l2 = (diff_mask*self.threshold - torch.abs(diff_mask*m))**2
        mse_custom = 1/M * (torch.sum(l1,dim=-1) + torch.sum(l2,dim=-1) )
        return -torch.log(mse_custom)

    def square_loss(self,tensor, key):
        loss = (tensor - key.float()) ** 2
        return -torch.log(torch.mean(loss, dim=-1))
        
    def focal_loss(self, m, m_prime, threshold=10.0,alpha=0.25):
        BCE_loss = F.binary_cross_entropy_with_logits(m.float(), m_prime.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_weights = alpha * (1 - pt) ** threshold
        focal_loss = focal_weights * BCE_loss  

        return -torch.log(torch.mean(focal_loss,dim=-1))


    def square_loss_with_threshold(self,tensor, key):
        
        
        loss = (tensor - key) ** 2
        mask = torch.abs(tensor - key) > self.threshold
        loss = loss * mask.float()
        return -torch.log(torch.mean(loss,dim=-1))
    
    def bce_loss_with_threshold(self, tensor, key):
        raise NotImplementedError("Reimplement threshold for bce")
        loss = F.binary_cross_entropy_with_logits(tensor, key, reduction='none')

        return loss.mean()

    def logistic_loss_with_logits(self,tensor, key):
        
        
        bce_loss = torch.nn.BCEWithLogitsLoss()
        return -torch.log(bce_loss(tensor, key))

    def logistic_loss_with_threshold(self,tensor, key):
        
        
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none') 
        loss = bce_loss(tensor, key)
        mask = torch.abs(torch.sigmoid(tensor) - key) > self.threshold
        loss = loss * mask.float()
        return -torch.log(torch.mean(loss,dim=-1))

    def hinge_loss(self,tensor, key):
        
        
        key = 2 * key - 1
        loss = torch.clamp(1 - tensor * key, min=0)
        return -torch.log(torch.mean(loss,dim=-1))

    def squared_hinge_loss(self,tensor, key):
        
        
        key = 2 * key - 1
        loss = torch.clamp(self.threshold - tensor * key, min=0) ** 2
        return -torch.log(torch.mean(loss,dim=-1))


    def exponential_loss(self,tensor, key):
        
        
        key = 2 * key - 1
        loss = torch.exp(-tensor * key)
        return -torch.log(torch.mean(loss,dim=-1))
    
    def __call__(self):
        if self.loss_type == 'dist':
            return self.dist
        elif self.loss_type == 'square_loss':
            return self.square_loss
        elif self.loss_type == 'cossim':
            return self.cossim
        elif self.loss_type == 'mvg_pval':
            return self.mvg_pval
        elif self.loss_type == 'poly_betainc':
            return self.poly_betainc
        elif self.loss_type == 'poly_betainc_prec':
            return self.poly_betainc_prec
        elif self.loss_type == 'square_loss_with_threshold':
            return self.square_loss_with_threshold
        elif self.loss_type == 'logistic_loss_with_logits':
            return self.logistic_loss_with_logits
        elif self.loss_type == 'logistic_loss_with_threshold':
            return self.logistic_loss_with_threshold
        elif self.loss_type == 'hinge_loss':
            return self.hinge_loss
        elif self.loss_type == 'focal_loss':
            return self.focal_loss
        elif self.loss_type == 'squared_hinge_loss':
            return self.squared_hinge_loss
        elif self.loss_type == 'exponential_loss':
            return self.exponential_loss
        elif self.loss_type == 'binary_cross_entropy':
            return self.bce_loss_with_threshold
        elif self.loss_type == 'focal_loss':
            return self.focal_loss
        else:
            raise NotImplementedError(f"Unknown loss_type : {self.loss_type}")
