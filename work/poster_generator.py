import random
from dataclasses import dataclass
from PIL import Image


@dataclass
class CarParameter:
    car_image: Image
    mask_image: Image


class CarPosterGenerator:
    def __init__(self):
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
        self.negativate_prompt = "Easy Negative,worst quality,low quality,normal quality,lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers"

    def generate_prompt(self, ind=None):
        if ind is None:
            return f"a car, high resolution,street level view,{random.choice(self.background_title_list)},4k"
        else:
            return f"a car, high resolution,street level view,{self.background_title_list[ind]},4k"

    def generate_poster(self, ):
        pass
