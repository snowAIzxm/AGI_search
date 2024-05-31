import os

import numpy as np
import torch
from PIL import Image

from work.controlnet_forward.control_mask_generate import ControlImageGenerator
from work.controlnet_forward.model import pipe
from work.controlnet_forward.prompt_generator import generate_ind_prompt, negative_prompt

control_image_generator = ControlImageGenerator()
# 主函数，文件内容参考 demo/controlnet_forward_data 里面的文件

def process_clue_dir_image(clue_dir):
    car_image = Image.open(os.path.join(clue_dir, "image.png"))
    car_mask = Image.open(os.path.join(clue_dir, "mask.png"))
    text_image = Image.open(os.path.join(clue_dir, "text.png"))
    text_image = np.array(text_image)
    car_image = np.array(car_image)
    car_mask = np.array(car_mask)
    car_image = control_image_generator.process_car_image(car_image, car_mask)
    car_h = car_image.size[1]
    text_image = control_image_generator.process_text_image(text_image)
    text_h = text_image.size[1]
    edge_image = control_image_generator.get_text_image_region(np.array(text_image)[..., 3])
    control_image = control_image_generator.make_inpaint_condition(car_image, edge_image)
    prompt = generate_ind_prompt(None)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    res_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=25,
        guidance_scale=7,
        width=control_image_generator.model_input_w,
        height=car_h + text_h,
        controlnet_conditioning_scale=0.5,
        generator=generator,
    ).images[0]

    text_mask = Image.fromarray(np.array(text_image)[..., 3])
    res_image.paste(text_image, mask=text_mask)
    car_mask = Image.fromarray(np.array(car_image)[..., 3])
    res_image.paste(car_image, [0, text_h, control_image_generator.model_input_w, car_h + text_h, ], mask=car_mask)

    return res_image
