import os.path

import torch.backends.cudnn as cudnn

from work.cn_clip_train.dataset import get_dataset
from work.cn_clip_train.get_lossw import load_model, generate_scheduler_and_optimizer

dataset_path = ""
vision_model_config_file = f"./Chinese-CLIP/cn_clip/clip/model_configs/ViT-B-16.json"
text_model_config_file = f"./Chinese-CLIP/cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
resume_path = ""
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'valid')

wd = 0.001
lr = 5e-5
warmup = 100
total_steps = 10000
batch_size = 128
max_txt_length = 52
mask_ratio = 0.5
accum_freq = 8
epoch = 3

cudnn.benchmark = True
cudnn.deterministic = False

model, model_info = load_model(vision_model_config_file, text_model_config_file, resume_path)

# freeze vision
for k, v in model.visual.named_parameters():
    v.requires_grad = False

scheduler, optimizer = generate_scheduler_and_optimizer(model, lr, warmup, total_steps, wd)

# dataset
train_data_info = get_dataset(train_path, model_info["image_resolution"], batch_size, True,
                              max_txt_length=max_txt_length,
                              epoch_id=0)
val_data_info = get_dataset(val_path, model_info["image_resolution"], batch_size, False,
                            max_txt_length=max_txt_length, epoch_id=0)
