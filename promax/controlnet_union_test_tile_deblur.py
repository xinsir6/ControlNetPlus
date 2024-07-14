# diffusers测试ControlNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('..')
import cv2
import time
import torch
import random
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from guided_filter import FastGuidedFilter
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline



def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

def apply_guided_filter(image_np, radius, eps, scale):
    filter = FastGuidedFilter(image_np, radius, eps, scale)
    return filter.filter(image_np)



device=torch.device('cuda:0')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
controlnet_model = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    # scheduler=ddim_scheduler,
    scheduler=eulera_scheduler,
)

pipe = pipe.to(device)


prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'




seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)

controlnet_img = cv2.imread("your image path")
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
W, H = int(width * ratio), int(height * ratio)

crop_w, crop_h = 0, 0
controlnet_img = cv2.resize(controlnet_img, (W, H))

# random blur
blur_strength = random.sample([i / 10. for i in range(10, 301, 2)], k=1)[0]
radius = random.sample([i for i in range(1, 60, 2)], k=1)[0]
eps = random.sample([i / 1000. for i in range(1, 101, 2)], k=1)[0]
scale_factor = random.sample([i / 10. for i in range(10, 321, 5)], k=1)[0]

if random.random() > 0.5:
    controlnet_img = apply_gaussian_blur(controlnet_img, ksize=int(blur_strength), sigmaX=blur_strength / 2)            

if random.random() > 0.5:
    controlnet_img = apply_guided_filter(controlnet_img, radius, eps, scale_factor)

# Resize image
controlnet_img = cv2.resize(controlnet_img, (int(W / scale_factor), int(H / scale_factor)), interpolation=cv2.INTER_AREA)
controlnet_img = cv2.resize(controlnet_img, (W, H), interpolation=cv2.INTER_CUBIC)

controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
controlnet_img = Image.fromarray(controlnet_img)

width, height = W, H

# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
# 6 -- tile
# 7 -- repaint
images = pipe(prompt=[prompt]*1,
            image_list=[0, 0, 0, 0, 0, 0, controlnet_img, 0], 
            negative_prompt=[negative_prompt]*1,
            generator=generator,
            width=width, 
            height=height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]),
            )

images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")


    
