import random

background_title_list = [
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
negative_prompt = "cartoon,Easy Negative,worst quality,low quality,normal quality,lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers"


def generate_ind_prompt(ind):
    if ind is None:
        value = random.choice(background_title_list)
    else:
        value = background_title_list[ind % (len(background_title_list))]
    return f"a car, high resolution,street level view,warm light,{value},4k"
