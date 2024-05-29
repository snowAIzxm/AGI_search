# first check image quality by alimodel or other
# second check image segment by sam or solov2
# third check image text by blip
class ControlDatasetProcess:
    def __init__(self, image_path_list, image_path_score_dict, image_path_mask_path_dict):
        self.image_score_limit = 0.8
        self.image_path_list = image_path_list
        self.image_path_score_dict = image_path_score_dict
        self.image_path_mask_path_dict = image_path_mask_path_dict

    def generate_dataset(self,):
