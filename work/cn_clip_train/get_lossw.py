import json

import torch
from cn_clip.clip.model import CLIP, resize_pos_embed
from cn_clip.training.scheduler import cosine_lr
from torch import optim, autocast
from tqdm import tqdm


def get_loss(image_features, text_features, logit_scale, loss_img, loss_txt):
    logit_scale = logit_scale.mean()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(non_blocking=True)

    total_loss = (loss_img(logits_per_image, ground_truth)
                  + loss_txt(logits_per_text, ground_truth)
                  ) / 2

    i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
    t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
    acc = {"i2t": i2t_acc, "t2i": t2i_acc}
    return total_loss, acc


def load_model(vision_model_config_file, text_model_config_file, resume_path):
    # model
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v
    model_info['use_flash_attention'] = False
    model = CLIP(**model_info)

    # load weight
    checkpoint = torch.load(resume_path, map_location="cpu")
    sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
    resize_pos_embed(sd, model, prefix="module.")
    sd2 = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd2)
    return model, model_info


def generate_scheduler_and_optimizer(model, lr, warmup, total_steps, wd):
    # optimizer
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": wd},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    return scheduler, optimizer


