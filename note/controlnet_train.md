# 美学评价

    https://github.com/christophschuhmann/improved-aesthetic-predictor
    未找到特别好的，目前先选用图片质量模型，先进行初步筛选和训练
    图片质量模型 https://modelscope.cn/models/iic/cv_resnet_image-quality-assessment-mos_youtubeUGC/summary

# 图片分割，准备control image

## 方案一

    利用深度预测寻找前景，然后根据点让sam进行分割

## 方案二

    对于现成的类别，用对应的分割模型进行分割

## 分割结果评判

    需要判断分割模型的好坏（可考虑模型分类的方式，todo）

# 字幕生产

    blip

# 开始训练
## 因为本身实际需要文字框，所以需要实际效果可以根据边缘生成一些类似的图片
### 方案一。只用类似于control image训练
### 方案二。尝试增加一些框
    