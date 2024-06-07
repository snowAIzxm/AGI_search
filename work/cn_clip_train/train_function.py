import torch
from torch import autocast
from tqdm import tqdm

from work.cn_clip_train.get_lossw import get_loss


def train_model_dataloader(dataloader, model,
                           scheduler, optimizer,
                           start_step, accum_freq, mask_ratio,
                           loss_img, loss_txt, writer):
    amount = len(dataloader) // accum_freq
    data_iter = iter(dataloader)
    model.train()
    for step in tqdm(range(start_step, start_step + amount)):
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
            with autocast(device_type="cuda"):
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
        writer.add_scalar('Accuracy/train_i2t', acc["i2t"], step)
        writer.add_scalar('Accuracy/train_t2i', acc["t2i"], step)
    return start_step + amount


def get_test_dataloader_acc_and_loss(model, dataloader, loss_img, loss_txt):
    model.eval()
    total_loss = 0
    total_i2t_acc = 0
    total_t2i_acc = 0
    total_num = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts, eos_indices = batch
            images = images.cuda(non_blocking=True)
            texts = texts.cuda(non_blocking=True)
            eos_indices = eos_indices.cuda(non_blocking=True)
            image_features, text_features, logit_scale = model(images, texts, 0)
            loss, acc = get_loss(image_features, text_features, logit_scale, loss_img, loss_txt)
            total_loss += loss.item() * len(images)
            total_i2t_acc += acc["i2t"] * len(images)
            total_t2i_acc += acc["t2i"] * len(images)
            total_num += len(images)
    return total_loss / total_num, total_i2t_acc / total_num, total_t2i_acc / total_num
