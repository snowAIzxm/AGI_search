import os.path
import random
from dataclasses import dataclass

import cv2
from PIL import Image, ImageOps
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDPMScheduler
)
import torch
from PIL import Image
import numpy as np


@dataclass
class CarParameter:
    car_image: Image
    mask_image: Image
    text_image: Image


class CarPosterGenerator:
    def __init__(self):
        self.model_input_w = 800
        self.model_input_h = int(self.model_input_w * 1.5)
        self.background_title_list = [
            "brick wall",  # 墙砖
            "Blank canvas",  # 空白画布
            "Solid color background",  # 纯色背景
            "Plain white wall",  # 纯白墙壁
            "Simple gradient background",  # 简单渐变背景
            "Minimalistic background",  # 极简背景
            "Neutral tone backdrop",  # 中性色调背景
            "Soft texture background",  # 柔和纹理背景
            "Subtle patterned wallpaper",  # 低调图案壁纸
            "Light-colored wooden wall",  # 浅色木墙
            "Smooth plaster wall",  # 光滑灰泥墙
        ]
        self.negativate_prompt = "cartoon,Easy Negative,worst quality,low quality,normal quality,lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers"
        self.pipe = self.init_pipe()

    def generate_prompt(self, ind=None):
        if ind is None:
            return f"a car, high resolution,street level view,studio warm light,{random.choice(self.background_title_list)},4k"
        else:
            return f"a car, high resolution,street level view,studio warm light,{self.background_title_list[ind]},4k"

    def process_car_parameter(self, car_parameter: CarParameter):
        origin_image = car_parameter.car_image
        mask_image = car_parameter.mask_image
        text_image = car_parameter.text_image
        w = self.model_input_w
        origin_w, origin_h = origin_image.size
        origin_image = origin_image.resize((w, round(origin_h / origin_w * w)))
        mask_image = mask_image.resize((w, round(origin_h / origin_w * w)))
        pad_h = self.model_input_h - mask_image.size[1]
        origin_image = ImageOps.expand(origin_image, border=(0, pad_h, 0, 0), fill='white')
        mask_image = ImageOps.expand(mask_image, border=(0, pad_h, 0, 0), fill='white')
        return origin_image, mask_image

    @staticmethod
    def init_pipe():
        controlnet = ControlNetModel.from_pretrained(
            "alimama-creative/EcomXL_controlnet_inpaint",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        return pipe

    @staticmethod
    def make_inpaint_condition(init_image, mask_image, edges=None):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel
        if edges is not None:
            edges = edges > 0
            init_image[edges] = 1.0  # set as edge pixel

        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image

    @staticmethod
    def add_fg(full_img, fg_img, mask_img):
        full_img = np.array(full_img).astype(np.float32)
        fg_img = np.array(fg_img).astype(np.float32)
        mask_img = np.array(mask_img).astype(np.float32) / 255.
        full_img = full_img * mask_img[:, :, np.newaxis] + fg_img * (1 - mask_img[:, :, np.newaxis])
        return Image.fromarray(np.clip(full_img, 0, 255).astype(np.uint8))

    def generate_background_image(self, image, mask, edges):
        control_image = self.make_inpaint_condition(image, mask,edges)
        prompt = self.generate_prompt()
        generator = torch.Generator(device="cuda").manual_seed(1234)

        res_image = self.pipe(
            prompt,
            negative_prompt=self.negativate_prompt,
            image=control_image,
            num_inference_steps=25,
            guidance_scale=7,
            width=self.model_input_w,
            height=self.model_input_h,
            controlnet_conditioning_scale=0.5,
            generator=generator,
        ).images[0]

        res_image = self.add_fg(res_image, image, mask)
        return res_image

    def load_image(self, file_dir):
        mask_image = Image.open(os.path.join(file_dir, "mask.png"))
        mask_image = Image.fromarray(255 - np.array(mask_image))
        image = Image.open(os.path.join(file_dir, "image.png"))
        text_image = Image.open(os.path.join(file_dir, "text.png"))
        w, h = image.size
        image = image.resize((self.model_input_w, int(h / w * self.model_input_w)))
        mask_image = mask_image.resize((self.model_input_w, int(h / w * self.model_input_w)))
        pad_h = self.model_input_h - mask_image.size[1]
        image = ImageOps.expand(image, border=(0, pad_h, 0, 0), fill='white')
        mask_image = ImageOps.expand(mask_image, border=(0, pad_h, 0, 0), fill="white")
        w, h = text_image.size
        text_image = text_image.resize((self.model_input_w, int(h / w * self.model_input_w)))
        text_image = text_image.crop([0, 0, self.model_input_w, self.model_input_h])
        gray_image = cv2.cvtColor(np.array(text_image), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # 使用 Canny 进行边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        return image, mask_image, edges
