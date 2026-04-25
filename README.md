# Guidance Watermarking for Diffusion Model demo

This package aims to provide a demo to use the guidance from the paper with three Diffusion Models (Sana, Flux and Stable-Diffusion 2) and a detector (VideoSeal).

## Installation

1. Open the unzip file in a terminal.

2. Install the package:
```
cd guidance_watermarking/

# Ideally create and activate a venv
# python -m venv ./venv
# source ./venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --editable .
```

3. Downloads VideoSeal detector from: https://github.com/facebookresearch/videoseal

Add models/ and whitener/ folders to: path/to/guidance_watermarking/guidance-watermarking-for-diffusion-models/detector/

4. Update the WHITENER_PATH in path/to/guidance_watermarking/guidance-watermarking-for-diffusion-models/detector/init.py.

## Getting started

1. Update the prompt dataset path in path/to/guidance_watermarking/guidance-watermarking-for-diffusion-models/configs/sana.yaml, flux.yaml and sd2.yaml.

2. Three scripts are available in path/to/guidance_watermarking/guidance-watermarking-for-diffusion-models/ for the three diffusion models. The first command line generates images with the guidance in the folder results. Defaults parameters are those used in the paper. You can change them modifying the corresponding test_params file in the configs/ folder. The second command line will execute the benchmark on the attacks presentd in the paper. To observ the results, go to the results folder, you'll find the images, bit accuracy and pfas for the script you ran.  
