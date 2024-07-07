import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from controlnet_aux import LineartDetector
from diffusers import EulerAncestralDiscreteScheduler
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline



device=torch.device('cuda:0')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

controlnet_model = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=eulera_scheduler,
)

pipe = pipe.to(device)


processor = LineartDetector.from_pretrained('lllyasviel/Annotators').to(device)

prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


controlnet_img = cv2.imread("your image path")
controlnet_img = processor(controlnet_img, output_type='cv2')


# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
controlnet_img = Image.fromarray(controlnet_img)


seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)


# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
images = pipe(prompt=[prompt]*1,
            image_list=[0, 0, 0, controlnet_img, 0, 0], 
            negative_prompt=[negative_prompt]*1,
            generator=generator,
            width=width, 
            height=height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 1, 0, 0]),
            ).images

images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")


