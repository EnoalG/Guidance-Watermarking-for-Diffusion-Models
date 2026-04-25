from ..detector.detector import StableSignatureDetector, TrustmarkDetector
from ..util.util_detector import strtobool, booltostr
from ..benchmarks.datasets import WmDataset

import torch
import numpy as np
from os import path
from . import MODELDIR, IMTESTDIR,DATATESTDIR,WMDATASET
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

from torchvision import transforms as T

from ..util.util_images import load_image
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_detector_single(detector):

    print("\t1. Single Image")
    transform = T.Compose([T.ToTensor()])
    key = torch.Tensor(strtobool('0'*detector.M)).to(device)


    

    im_path = path.join(IMTESTDIR ,'0.jpg')
    im = transform(load_image(im_path)).unsqueeze(0).to(device)
    res = detector(im,key)

    print(f"Extracted message: {booltostr(res['dec_message'].cpu().numpy()[0])}, Bit accuracy: {res['bit_acc']}, Log_10 P-val: {np.log10(res['pval'])}")

def test_detector_multiple(detector):
    print("\t2. Batched Images")
    data_path = path.join(DATATESTDIR, 'cover_test.jsonl')
    imdataset = WmDataset(label_file=data_path, im_dir=IMTESTDIR,M=detector.M, 
                            transform=T.Compose([T.ToTensor(),T.Resize(detector.im_size),
                                T.CenterCrop(detector.im_size)]))

    test_dataloader = DataLoader(imdataset, batch_size=2, shuffle=True)

    for i, data in enumerate(test_dataloader):
        im, key,_ = data
        res = detector(im.to(device),key.to(device))
        print(f"Extracted message: {list(map(booltostr, res['dec_message'].cpu().numpy()))}, Bit accuracy: {res['bit_acc']}, Log_10 P-val: {np.log10(res['pval'])}")

def test_detector_mutiple_wm(detector,im_dir, data_path, batch_size =8,nsamples=100):
    print("\t3. Watermarked Images")
        
    imdataset = WmDataset(label_file=data_path, im_dir=im_dir,M=detector.M, 
                          transform=T.Compose([T.ToTensor()]))
    imdataset =  data_utils.Subset(imdataset, range(nsamples))
    test_dataloader = DataLoader(imdataset, batch_size=batch_size, shuffle=False)

    bit_acc = []
    pvals = []
    max_epochs =len(test_dataloader)
    print(f"Dataset size: {max_epochs}")
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            print(f"Iter: {i+1}/{max_epochs}",end='\r')
            im, key, _ = data
            res = detector(im.to(device),key.to(device))
            bit_acc.append(res['bit_acc'].detach().cpu().numpy())
            pvals.append(res['pval'])
        bit_acc = np.concatenate(bit_acc)
        pvals = np.concatenate(pvals)
        print(f" Summary ||| Bit accuracy: {np.mean(bit_acc)}, Mean P-val: {np.mean(-np.log10(pvals))}")



def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



    
    print("--- Testing Stable Signature ---")
    M=48
    msg_decoder_path = path.join(MODELDIR ,'dec_48b_whit.torchscript.pt')
    model= torch.jit.load(msg_decoder_path).to(device)
    model.eval()

    im_size=256
    detector = StableSignatureDetector(model, M=M,im_size=im_size).to(device)
    test_detector_single(detector)

    im_size=256
    detector = StableSignatureDetector(model, M=M,im_size=im_size).to(device)
    test_detector_multiple(detector)

    im_size=512
    detector = StableSignatureDetector(model, M=M,im_size=im_size).to(device)
    im_dir = path.join(WMDATASET, 'only_vae')
    data_path = path.join(DATATESTDIR, 'sd2_ssig.jsonl')
    test_detector_mutiple_wm(detector,im_dir=im_dir,data_path=data_path, batch_size=16)

    
    print("--- Testing Trustmark / Quality, No ECC ---")
    im_size=245
    M=100

    msg_decoder_path = path.join(MODELDIR ,'Trustmark_Q_noECC_100bits_245res_resnet50.pt')
    model= torch.jit.load(msg_decoder_path).to(device)
    model.eval()

    detector = TrustmarkDetector(model, M=M,im_size=im_size).to(device)
    test_detector_single(detector)

    detector = TrustmarkDetector(model, M=M,im_size=im_size).to(device)
    test_detector_multiple(detector)

    detector = TrustmarkDetector(model, M=M,im_size=im_size).to(device)
    im_dir = path.join(WMDATASET, 'sd2_trustmark')
    data_path = path.join(DATATESTDIR, 'sd2_trustmark.jsonl')
    test_detector_mutiple_wm(detector,im_dir=im_dir,data_path=data_path, batch_size=16)


if __name__ == "__main__":
     main()