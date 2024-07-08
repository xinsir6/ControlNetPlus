import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler
from controlnet_aux import HEDdetector
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


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

processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


# you can use either hed to generate a fake scribble given an image or a sketch image totally draw by yourself
if random.random() > 0.5:
    # Method 1 
    # if you use hed, you should provide an image, the image can be real or anime, you extract its hed lines and use it as the scribbles
    # The detail about hed detect you can refer to https://github.com/lllyasviel/ControlNet/blob/main/gradio_fake_scribble2image.py
    # Below is a example using diffusers HED detector

    image_path = Image.open("your image path, the image can be real or anime, HED detector will extract its edge boundery")
    controlnet_img = processor(image_path, scribble=False)
    controlnet_img.save("a hed detect path for an image")

    # following is some processing to simulate human sketch draw, different threshold can generate different width of lines
    controlnet_img = np.array(controlnet_img)
    controlnet_img = nms(controlnet_img, 127, 3)
    controlnet_img = cv2.GaussianBlur(controlnet_img, (0, 0), 3)

    # higher threshold, thiner line
    random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
    controlnet_img[controlnet_img > random_val] = 255
    controlnet_img[controlnet_img < 255] = 0
    controlnet_img = Image.fromarray(controlnet_img)

else:
    # Method 2
    # if you use a sketch image total draw by yourself
    control_path = "the sketch image you draw with some tools, like drawing board, the path you save it"
    controlnet_img = Image.open(control_path) # Note that the image must be black-white(0 or 255), like the examples we list


# must resize to 1024*1024 or same resolution bucket to get the best performance
width, height  = controlnet_img.size
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = controlnet_img.resize((new_width, new_height))


seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)


# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
images = pipe(prompt=[prompt]*1,
            image_list=[0, 0, controlnet_img, 0, 0, 0], 
            negative_prompt=[negative_prompt]*1,
            generator=generator,
            width=new_width, 
            height=new_height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 1, 0, 0, 0]),
            ).images

images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")


