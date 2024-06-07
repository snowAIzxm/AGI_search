import os.path

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
from torch import nn

from work.cn_clip_train.dataset import get_dataset
from work.cn_clip_train.get_lossw import load_model, generate_scheduler_and_optimizer, get_loss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs/clip_v1')
cudnn.benchmark = True
cudnn.deterministic = False

dataset_path = "/data1/zhuxiaoming3/clip_train/2024-06-03/lmdb/"
vision_model_config_file = f"./Chinese-CLIP/cn_clip/clip/model_configs/ViT-B-16.json"
text_model_config_file = f"./Chinese-CLIP/cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
resume_path = "/data1/zhuxiaoming3/clip_train/muge/MUGE/clip_cn_vit-b-16.pt"
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'valid')
wd = 0.001
lr = 5e-5
warmup = 100
batch_size = 128
max_txt_length = 52
mask_ratio = 0.5
epoch = 3

model, model_info = load_model(vision_model_config_file, text_model_config_file, resume_path)
model = model.cuda()
# freeze vision
for k, v in model.visual.named_parameters():
    v.requires_grad = False

# dataset
train_data_info = get_dataset(train_path, model_info["image_resolution"], batch_size, True,
                              max_txt_length=max_txt_length,
                              epoch_id=0)
val_data_info = get_dataset(val_path, model_info["image_resolution"], batch_size, False,
                            max_txt_length=max_txt_length, epoch_id=0)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_img = loss_img.cuda()
loss_txt = loss_txt.cuda()

# train
dataloader = train_data_info.dataloader
dataloader_size = len(dataloader)
accum_freq = 4
total_steps = len(dataloader) * epoch / accum_freq
scheduler, optimizer = generate_scheduler_and_optimizer(model, lr, warmup, total_steps, wd)

dataloader.pin_memory = True
data_iter = iter(dataloader)

# first epoch
for step in range(dataloader_size // accum_freq):

    scheduler(step)
    optimizer.zero_grad()
    image_features_list = []
    text_features_list = []
    logit_scale_list = []

    for _ in range(accum_freq):
        batch = next(data_iter)
        images, texts, eos_indices = batch

        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        eos_indices = eos_indices.cuda(non_blocking=True)
        with autocast():
            image_features, text_features, logit_scale = model(images, texts, mask_ratio)
            image_features_list.append(image_features)
            text_features_list.append(text_features)
            logit_scale_list.append(logit_scale)
    image_feature = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    logit_scale = logit_scale_list[0]
    loss, acc = get_loss(image_feature, text_features, logit_scale, loss_img, loss_txt)
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss.item(), step)
    writer.add_scalar('Accuracy/train', acc, step)
