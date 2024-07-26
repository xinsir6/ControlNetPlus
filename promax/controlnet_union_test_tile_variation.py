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
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline


device=torch.device('cuda:0')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
# you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
controlnet_model = ControlNetModel_Union.from_pretrained("./controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16, use_safetensors=True)


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
            ).images

controlnet_img.save("controlnet_variation.webp")
images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")

