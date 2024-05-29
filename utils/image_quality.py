import time

from PIL import Image
import numpy as np
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from tqdm import tqdm


class ImageQualityCalculator:
    def __init__(self):
        self.device = "cuda"
        self.model = pipeline(
            Tasks.image_quality_assessment_mos,
            'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC', device=self.device)
        self.batch_size = 16
        self.init_model()

    def init_model(self):
        start_time = time.time()
        self.model.model.training = False
        self.model.prepare_model()
        result = self.model(np.zeros((1000, 1000, 3), dtype=np.uint8))
        print(f"init model success, result is {result},cost time {time.time() - start_time}")

    def predict_image_list(self, image_path_list):
        image_tensor_list = []
        success_image_path_list = []
        for image_path in image_path_list:
            try:
                image = Image.open(image_path)
                image_tensor = self.model.preprocess(image)["input"]
                image_tensor_list.append(image_tensor)
                success_image_path_list.append(image_path)
            except Exception as e:
                print(f"error:{e}")
        image_tensor_list = torch.cat(image_tensor_list)
        with torch.no_grad():
            model_result = self.model.model(dict(input=image_tensor_list.to(self.device)))["output"].cpu()
        result = {}
        for image_path, score in zip(success_image_path_list, model_result):
            result[image_path] = score
        return result

    def get_image_quality(self, image_path):
        # i think >0.8 is can accept, 具体视情况而定
        image_quality = self.model(image_path)[OutputKeys.SCORE]
        return image_quality

    def get_image_name__list_score(self, image_path_list):
        result_list = []
        for i in tqdm(range(0, len(image_path_list), self.batch_size)):
            batch_image_path_list = image_path_list[i:i + self.batch_size]
            result_list.append(self.predict_image_list(batch_image_path_list))
        return result_list


self = ImageQualityCalculator()
