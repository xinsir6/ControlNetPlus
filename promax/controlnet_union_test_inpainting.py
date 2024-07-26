# diffusers测试ControlNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('..')
import cv2
import copy
import torch
import random
import numpy as np
from PIL import Image
from mask import get_mask_generator
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_inpaint_sd_xl import StableDiffusionXLControlNetUnionInpaintPipeline


device=torch.device('cuda:0')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
# you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
controlnet_model = ControlNetModel_Union.from_pretrained("./controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16, use_safetensors=True)


pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    # scheduler=ddim_scheduler,
    scheduler=eulera_scheduler,
)


pipe = pipe.to(device)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

mask_gen_kwargs = {
            "irregular_proba": 1,
            "irregular_kwargs": {
                "max_angle": 4,
                "max_len": 200 * 4,
                "max_width": 100 * 4,
                "max_times": 1,
                "min_times": 1
            },
            "box_proba": 1,
            "box_kwargs": {
                "margin": 10,
                "bbox_min_size": 30 * 4,
                "bbox_max_size": 150 * 4,
                "max_times": 1,
                "min_times": 1
            },
        }

mask_gen = get_mask_generator(kind='mixed', kwargs=mask_gen_kwargs)


prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


seed = random.randint(0, 2147483647)

# The original image you want to repaint.
original_img = cv2.imread("your image path")
# # inpainting support any mask shape
# # where you want to repaint, the mask image should be a binary image, with value 0 or 255.
# mask = cv2.imread("your mask image path") 

height, width, _  = original_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
W, H = int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8
original_img = cv2.resize(original_img, (W, H))
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

import copy
controlnet_img = copy.deepcopy(original_img)
controlnet_img = np.transpose(controlnet_img, (2, 0, 1))
mask = mask_gen(controlnet_img)
controlnet_img = np.transpose(controlnet_img, (1, 2, 0))
mask = np.transpose(mask, (1, 2, 0))

controlnet_img[mask.squeeze() > 0.0] = 0
mask = HWC3((mask * 255).astype('uint8'))

controlnet_img = Image.fromarray(controlnet_img)
original_img = Image.fromarray(original_img)
mask = Image.fromarray(mask)

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
            image=original_img,
            mask_image=mask,
            control_image_list=[0, 0, 0, 0, 0, 0, 0, controlnet_img], 
            negative_prompt=[negative_prompt]*1,
            # generator=generator,
            width=width, 
            height=height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1]),
            ).images

controlnet_img.save("control_inpainting.webp")
images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")

