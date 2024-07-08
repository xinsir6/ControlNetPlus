import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from controlnet_aux import OpenposeDetector, MidasDetector, ZoeDetector
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

# Example to show(openpose + depth)
processor_pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(device)
processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators").to(device)
processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)


prompt = "your prompt, the longer the better, you can describe it as detail as possible"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


source_img = cv2.imread("your image path")
controlnet_img_pose = processor_pose(source_img, hand_and_face=False, output_type='cv2')
# Or you can choose another img for depth detection, the content should be compatible with the pose skeleton.
if random.random() > 0.5:
    controlnet_img_depth = processor_zoe(source_img, output_type='cv2')
else:
    controlnet_img_depth = processor_midas(source_img, output_type='cv2')


# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
height, width, _  = controlnet_img_pose.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)

controlnet_img_pose = cv2.resize(controlnet_img_pose, (new_width, new_height))
controlnet_img_depth = cv2.resize(controlnet_img_depth, (new_width, new_height))
controlnet_img_pose = Image.fromarray(controlnet_img_pose)
controlnet_img_depth = Image.fromarray(controlnet_img_depth)


seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)

# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
# image_list must be compatible with union_control_type
images = pipe(prompt=[prompt]*1,
            image_list=[controlnet_img_pose, controlnet_img_depth, 0, 0, 0, 0], 
            negative_prompt=[negative_prompt]*1,
            generator=generator,
            width=new_width, 
            height=new_height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([1, 1, 0, 0, 0, 0]),
            crops_coords_top_left=(0, 0),
            target_size=(new_width, new_height),
            original_size=(new_width * 2, new_height * 2),
            ).images

images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")


