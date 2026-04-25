import torch
import torch.nn as nn
import torchvision.transforms as T

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
    def get_name(self):
        raise NotImplementedError

class TransformSet(Transform):
    def __init__(self, transform_set):
        super().__init__()
        self.transform_set = transform_set
    def __len__(self):
        return len(self.transform_set)
    def get_name(self):
        return '_'.join([t.get_name() for t in self.transform_set])
    def forward(self, x):
        aug_x = x.clone()
        for t in self.transform_set: aug_x = t(aug_x)
        return(aug_x)
    def parallel_forward(self, x):
        res = {'Identity' : x}
        for i,t in enumerate(self.transform_set):
            res[t.get_name()+'_'+str(i)] = t(x)
        return(res)
    

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
    def get_name(self):
        raise NotImplementedError
    
class ScoreSet(nn.Module):
    def __init__(self, score_set):
        super(ScoreSet, self).__init__()
        self.score_set = score_set
    def get_name(self):
        return '_'.join([s.get_name() for s in self.score_set])

    def forward(self, x,x_ref):
        res = {}
        for i,s in enumerate(self.score_set):
            res[s.get_name()] = s(x,x_ref)
        return(res)