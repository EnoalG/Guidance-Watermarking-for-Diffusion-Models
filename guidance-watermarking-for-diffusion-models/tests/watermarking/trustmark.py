    

from ...detector.detector import TrustmarkDetector 
from ...watermarking.watermark import TrustmarkNoECCWatermarker


from ...util.util_detector import strtobool, booltostr
from ...benchmarks.datasets import ImageDataset

from ...tests import MODELDIR, LOCALDATA,WMDIR
from ...tests.test_detector import test_detector_mutiple_wm


from os import path,listdir,makedirs
import json
import numpy as np



import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def test_trustmark_post_hoc(watermarker : TrustmarkNoECCWatermarker,
                      im_dir, outdir, jsonfile,
                      im_size=512, batch_size =8,nsamples=100, ext='.jpg',M=100):
    print('\t Testing Trustmark watermarker')
    if not path.isdir(outdir): makedirs(outdir)
    imdataset = ImageDataset(im_dir=im_dir, 
                          transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size) ]), ext=ext)
    imdataset =  data_utils.Subset(imdataset, range(nsamples))
    test_dataloader = DataLoader(imdataset, batch_size=batch_size, shuffle=False)
    max_epochs =len(test_dataloader)
    json_data = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            print(f"Iter: {i+1}/{max_epochs}",end='\r')
            im, _, imname = data
            keys = (torch.rand(im.shape[0], M) >0.5).float()
            ims_wm = watermarker(im.to(device),keys.to(device))
            for i,im_wm in enumerate(ims_wm):
                save_image(im_wm, path.join(outdir, imname[i] ))
                json_data.append({'name' : imname[i], 'key': booltostr(keys[i].cpu().numpy())})
    
    with open(jsonfile, 'w') as f:
        for d in json_data:
            f.write(json.dumps(d) + "\n")
            f.flush()
### HELPER TESTS (Ugly not for production) ###

def test_og_trustmark(im_dir, im_size=512,nsamples=300,    ext='.jpg'):
    print('\t 2.3 OG Trustmark')
    from trustmark import TrustMark
    from PIL import Image
    # init
    tm=TrustMark(verbose=True, model_type='Q', use_ECC=False, encoding_type=None) # or try P
    batch_size=nsamples


    imdataset = ImageDataset(im_dir=im_dir, 
                          transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size) ]), ext=ext)
    imdataset =  data_utils.Subset(imdataset, range(nsamples))
    test_dataloader = DataLoader(imdataset, batch_size=batch_size, shuffle=False)
    bit_acc = []
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            bintxt = '1'*100
            im,_, imname = data
            for i in range(nsamples):
                print(f'{i+1}/{nsamples}')

                
                tim = T.ToPILImage()(im[i])
                wimout = tm.encode(tim, bintxt,MODE='binary')
                wimout.save(f'/path/to/guidance-watermarking-for-diffusion-models/tests/__local_data__/test_og_tm/{imname[i]}')
                T.ToPILImage()(10*torch.abs(T.ToTensor()(wimout) - T.ToTensor()(tim))).save(f'/path/to/guidance-watermarking-for-diffusion-models/tests/__local_data__/test_og_tm/res_{imname[i]}')
                wmim = Image.open(f'/path/to/guidance-watermarking-for-diffusion-models/tests/__local_data__/test_og_tm/{imname[i]}').convert('RGB')
                wm_secret, wm_present, wm_schema = tm.decode(wmim)
                bit_acc.append(np.sum(strtobool(wm_secret))/100)
                
    print(np.mean(bit_acc))

def test_res_og_trustmark(im_dir, im_size=512,nsamples=300,  ext='.jpg'):
    print('\t 2.3 OG Trustmark')
    from trustmark import TrustMark
    from PIL import Image
    # init
    tm=TrustMark(verbose=True, model_type='Q', use_ECC=False, encoding_type=None) # or try P
    batch_size=nsamples


    imdataset = ImageDataset(im_dir=im_dir, 
                          transform=T.Compose([T.ToTensor(), T.Resize(im_size),T.CenterCrop(im_size) ]), ext=ext)
    imdataset =  data_utils.Subset(imdataset, range(nsamples))
    test_dataloader = DataLoader(imdataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            im,_, imname = data
            for i in range(nsamples):
                print(f'{i+1}/{nsamples}')

                
                tim = T.ToPILImage()(im[i])
                wmim = Image.open(f'/path/to/guided-diffusion/images/sd2_model/sd2/{imname[i]}').convert('RGB')

                T.ToPILImage()(10*torch.abs(T.ToTensor()(wmim) - T.ToTensor()(tim))).save(f'/path/to/guidance-watermarking-for-diffusion-models/tests/__local_data__/res_sd2_tm/res_{imname[i]}')

###############################################


def test_trustmark_Q100bits(im_dir, im_size=512,nsamples=100,batch_size=8 ):
    print("\t Testing Trustmark / Quality, No ECC (Post-hoc)")

    wm_name = "trusmark100b_noecc"
    outdir = path.join(LOCALDATA, 'test_tm')
    out_data_dir = path.join(LOCALDATA, f'test_{wm_name}_flickr.json')

    encoder_path = path.join(WMDIR ,'Trustmark_Q_noECC_100bits_256to245res_encoder.pt')
    model= torch.jit.load(encoder_path).to(device)
    model.eval()

    tm_watermarker = TrustmarkNoECCWatermarker(model, enc_size=256)

    test_trustmark_post_hoc(tm_watermarker,
                      im_dir, outdir, out_data_dir,
                      im_size=im_size, batch_size =batch_size,nsamples=nsamples, ext='.jpg',M=100)
    
    print("\t Testing if message can be decoded")
    
    dec_size=245
    M=100

    msg_decoder_path = path.join(MODELDIR ,'Trustmark_Q_noECC_100bits_245res_resnet50.pt')
    model= torch.jit.load(msg_decoder_path).to(device)
    model.eval()

    detector = TrustmarkDetector(model, M=M,im_size=dec_size).to(device)
    test_detector_mutiple_wm(detector,im_dir=outdir,data_path=out_data_dir, batch_size=batch_size)
