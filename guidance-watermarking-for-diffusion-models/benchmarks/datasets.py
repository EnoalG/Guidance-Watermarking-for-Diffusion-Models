
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor, tensor
from torch import rand as trand
from torch import randint as trandint

from torchvision.transforms import ToTensor

from os import path,listdir

from ..util.util_images import load_image
from ..util.util_detector import strtobool, booltostr


"""
All benchmark datasets adhere to the standard (x, key, other_data) where x can be either a prompt or a tensor image

"""

def generate_keys(key_type='bool',size=48):
    if key_type == 'bool':
        return(trand(size) > 0.5)
    elif key_type == 'int':
        return(trandint(low=0, high=2**32-1, size=(1,1)))
    else:
        raise NotImplemented("Unknown key type:", key_type)


class WmDataset(Dataset):
    """
    Loads images and their keys
    """
    def __init__(self, label_file, im_dir,transform=ToTensor(), M=48, ext ='.png'):
        self.im_dir = im_dir
        self.labels = pd.read_json(label_file, lines=True,dtype={'key': str})
        self.transform = transform
        self.M = M # Size of the expected message
        self.ext = ext

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):

        im_idx = idx

        key = self.labels.iloc[im_idx]['key']
        if key != 'none': 
            key = Tensor(strtobool(key))
        else: 
            key = Tensor(strtobool('0'*self.M))

        imname = self.labels.iloc[im_idx]['name']
        img_path = path.join(self.im_dir, imname + self.ext)
        image = self.transform(load_image(img_path))
                
        return image, key, imname
    
class ImageDataset(Dataset):
    """
    Loads images from directory
    """
    def __init__(self, im_dir,transform=ToTensor(), ext='.png',key=None,generate_keys=False,M=48,key_type='bool'):
        self.im_dir = im_dir
        self.im_list = [im for im in listdir(im_dir) if im.endswith(ext)]
        self.transform = transform
        self.key_type = key_type
        self.fixed_key = None
        self.generate_keys = generate_keys
        self.M=M
        self.ext = ext

        self._load_keys(key)

        
            
        
    def _load_keys(self,key):
        if key is not None:
            if type(key) is str and key.endswith('.jsonl') :
                synonyms = {'id': 'name', 'Identity':'key'}
                tmp = pd.read_json(key, lines=True,dtype={'name': str, 'key': str})
                if 'id' in tmp.keys(): 
                    tmp = pd.read_json(key, lines=True,dtype={'id': str, 'Identity': str})

                    tmp = tmp.rename(columns=synonyms)

                key_transform = None
                if self.key_type == 'bool': key_transform = lambda x: Tensor(strtobool(x))
                elif self.key_type == 'int': key_transform = lambda x: tensor([int(x)])
                else: raise NotImplementedError("Unknown key type:", self.key_type)

                self.keys = {im: key_transform(tmp[tmp['name'] == im.split(self.ext)[0]]['key'].item()) for im in self.im_list }
            else:
                self.fixed_key = Tensor(key)
                self.keys = {}
        else:
            self.keys = {}

    def __len__(self):
        return len(self.im_list)


    def __getitem__(self, idx):

        imname = self.im_list[idx]
        img_path = path.join(self.im_dir, imname)
        try:
            image = self.transform(load_image(img_path))
        except:
            raise ValueError("Invalid image at: ", img_path) # Tremendously helps debug corrupted datasets

        if imname in self.keys: 
            key = self.keys[imname]
        else:
            
            if self.generate_keys: 
                key = None
                if imname not in self.keys: 
                    self.keys[imname] = generate_keys(size=self.M, key_type=self.key_type)

                key = self.keys[imname]
            else:
                key = self.fixed_key
                
        return image, key, self.im_list[idx]

class PromptDataset(Dataset):
    """
    Generate a prompt dataset along with random watermarkign keys 
    """
    def __init__(self, prompt_file, key=None,generate_keys=False,M=48,key_type='bool'):

        self.prompts= pd.read_json(prompt_file, lines=True)
        self.generate_keys = generate_keys

        self.key_type = key_type
        self.fixed_key = None
        if key is not None: self.fixed_key = Tensor(key)
        
        self.M = M
        self.keys = {} # Don't generate keys in advance in case of subseting
    def __len__(self):
        return len(self.prompts)


    def __getitem__(self, idx):
        prompt = self.prompts.iloc[idx]['caption']
        id= self.prompts.iloc[idx]['id']
        key = self.fixed_key

        if self.generate_keys: 
            if idx not in self.keys: 
                self.keys[idx] = generate_keys(size=self.M, key_type=self.key_type)

            key = self.keys[idx]
        return prompt, key,str(id)
    
class PairedRefDataset(Dataset):
    def __init__(self,dirs:tuple, types:tuple, exts:tuple):

        assert len(dirs) == 2
        self.dirs = []
        self.data = []
        self.types = types
        self.exts = exts
        self.transform = ToTensor()

        for dir,t, ext in zip(dirs, types, exts) :
            if t.upper() == 'IMAGE':
                self.dirs.append(dir)
                self.data.append([im for im in listdir(dir) if im.endswith(ext)])
            elif t.upper() == 'PROMPT':
                self.dirs.append(None)
                self.data.append(pd.read_json(dir, lines=True))
            else:
                raise ValueError("Unknown dataset type : ", t)

        def __len__(self):
            return len(self.data[0])


    def __getitem__(self, idx):

        if self.types[0] == 'IMAGE':
            img_path = self.data[0][idx]
            id = img_path.split('.')[0]
            x= self.transform(load_image(path.join(self.dirs[0], img_path)))

        elif  self.types[0] == 'PROMPT':
            x = self.data[0].iloc[idx]['caption']
            id= self.data[0].iloc[idx]['id']


        if self.types[1] == 'IMAGE':
            y= self.transform(load_image(path.join(self.dirs[1], id + self.exts[1])))

        elif  self.types[1] == 'PROMPT':
            matching_ref = self.data[1]['id'] == id 
            y = self.data[1].loc[matching_ref]['caption']
            yid = self.data[1].loc[matching_ref]['id']
            assert id == yid


        
        
        return x,y,id