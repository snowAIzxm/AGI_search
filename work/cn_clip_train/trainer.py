import os.path
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
epoch = 20

model, model_info = load_model(vision_model_config_file, text_model_config_file, resume_path)
model = model.cuda()

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
accum_freq = 1
total_steps = len(dataloader) * epoch / accum_freq
scheduler, optimizer = generate_scheduler_and_optimizer(model, lr, warmup, total_steps, wd)

dataloader.pin_memory = True
data_iter = iter(dataloader)

# first freeze vision
# freeze vision
for k, v in model.visual.named_parameters():
    v.requires_grad = False
# second unfreeze vision
# freeze vision
for k, v in model.visual.named_parameters():
    v.requires_grad = True
# third mask ratio down
