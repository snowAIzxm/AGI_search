import json
import os.path

import requests
from tqdm import tqdm


class TextGenerator:
    def __init__(self):
        self.model = self.init_mplug()
        self.save_path = "./clip_text.json"
        self.update_size = 10000

    def init_mplug(self):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model_id = 'damo/mplug_image-captioning_coco_base_zh'
        pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)
        return pipeline_caption

    def process_image_path_list(self, image_path_list):
        result = {}
        if os.path.exists(self.save_path):
            result = json.load(open(self.save_path))
        for image_path in tqdm(image_path_list):
            if image_path not in result:
                try:
                    cap = self.model(image_path)
                    result[image_path] = cap
                except Exception as e:
                    print(e)
                    result[image_path] = f"{e}"
            if len(result) % self.update_size == 0:
                with open(self.save_path, "w") as f:
                    json.dump(result, f)


self = TextGenerator()
