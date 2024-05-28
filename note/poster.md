# 流程设计

## 背景

    提示词，brick wall or others

## 车体分割

    sam or 常见车体分割模型(sam-b case尝试效果不佳，最好具体情况具体分析)

## 背景图生成（优化前景）

    https://huggingface.co/alimama-creative/EcomXL_controlnet_inpaint
    阿里妈妈优化的一个电商场景的模型，避免前景被意外填充

## 文字内容

    先生产文字，然后作图，最后基于canny or depth controlnet看看
    
