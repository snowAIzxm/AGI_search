import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
model = model.cuda()


def segment_car(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to processor for postprocessing
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
    predicted_instance_map = result["segmentation"]
    segment_info = result["segments_info"]
    label_2_ind_list = []
    for i in range(len(segment_info)):
        if segment_info[i]["label_id"] == 2:
            label_2_ind_list.append(segment_info[i]["id"])
    if len(label_2_ind_list) == 0:
        return np.zeros_like(predicted_instance_map)
    id_ = max(label_2_ind_list, key=lambda x: (predicted_instance_map == x).sum())

    return predicted_instance_map == id_
