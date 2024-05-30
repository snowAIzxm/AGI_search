import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


class ControlImageGenerator:
    def __init__(self, model_w=800):
        self.model_input_w = model_w
        self.end_ratio = 0.1
        self.mid_w_ratio = 0.1
        self.text_pad_ratio = 0.1
        self.text_line_width = 4

    @staticmethod
    def get_car_mask_start_and_end_ratio(mask):
        h, w = mask.shape
        start_h = h
        end_h = 0
        start_w = w
        end_w = 0
        for i in range(h):
            if np.sum(mask[i]) > 0:
                start_h = min(start_h, i)
                end_h = max(end_h, i)
        for i in range(w):
            if np.sum(mask[:, i]) > 0:
                start_w = min(start_w, i)
                end_w = max(end_w, i)
        return start_h, end_h, start_w, end_w

    def process_car_image(self, car_image, car_mask):
        origin_image = np.concatenate([car_image, np.expand_dims(car_mask, 2)], axis=2)
        start_h, end_h, start_w, end_w = self.get_car_mask_start_and_end_ratio(car_mask)

        new_image = Image.fromarray(origin_image)
        new_image = new_image.crop((start_w, start_h, end_w, end_h))
        w, h = new_image.size
        pad_w1 = int(w / 0.8 * 0.1)

        new_image = ImageOps.expand(new_image, border=(pad_w1, pad_w1, pad_w1, pad_w1), fill=(0, 0, 0, 0))
        w, h = new_image.size
        new_image = new_image.resize((self.model_input_w, int(h / w * self.model_input_w)))
        return new_image

    def process_text_image(self, text_image):
        h, w = text_image.shape[:2]
        start_h, end_h, start_w, end_w = self.get_car_mask_start_and_end_ratio(text_image[..., 3])
        new_image = Image.fromarray(text_image)
        new_image = new_image.crop((0, int(start_h - w * self.text_pad_ratio), w, int(end_h + w * self.text_pad_ratio)))
        w, h = new_image.size
        new_image = new_image.resize((self.model_input_w, int(h / w * self.model_input_w),))
        return new_image

    @staticmethod
    def add_fg(full_img, fg_img, mask_img):
        full_img = np.array(full_img).astype(np.float32)
        fg_img = np.array(fg_img).astype(np.float32)
        mask_img = np.array(mask_img).astype(np.float32) / 255.
        full_img = full_img * mask_img[:, :, np.newaxis] + fg_img * (1 - mask_img[:, :, np.newaxis])
        return Image.fromarray(np.clip(full_img, 0, 255).astype(np.uint8))

    def get_text_image_region(self, text_mask):

        # 应用阈值以便识别字体区域
        _, thresh = cv2.threshold(text_mask, 127, 255, cv2.THRESH_BINARY_INV)

        # 腐蚀和膨胀操作
        kernel = np.ones((15, 60), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        # dilated = cv2.dilate(eroded, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(255 - eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取所有轮廓的四边形区域
        text_regions = []
        for contour in contours:
            # 使用多边形逼近将轮廓转换为四边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                # 如果逼近结果是四边形
                text_regions.append(approx)
            else:
                # 否则，计算最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                text_regions.append(box)
        edge_image = np.zeros_like(text_mask, dtype=np.uint8)
        for region in text_regions:
            cv2.drawContours(edge_image, [region], -1, (255), self.text_line_width)
        # 在原图上绘制四边形区域
        return edge_image

    @staticmethod
    def make_inpaint_condition(car_image, edges_image):
        init_image = np.array(car_image.convert("RGBA")).astype(np.float32) / 255.0
        mask_image = 1.0 - init_image[..., 3]
        init_image = init_image[..., :3]
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel

        zero_image = (edges_image > 0).astype(np.float16)[:, :, np.newaxis]
        zero_image[edges_image > 0.5] = 1.0
        zero_image[edges_image < 0.5] = -1.0

        zero_image = np.concatenate([zero_image, zero_image, zero_image], axis=2)
        new_zero_image = np.zeros((init_image.shape[0] + zero_image.shape[0], zero_image.shape[1], 3))
        new_zero_image[:zero_image.shape[0]] = zero_image
        new_zero_image[zero_image.shape[0]:] = init_image
        init_image = np.expand_dims(new_zero_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image

    def generate_result(self, car_image, car_mask, text_image):
        car_image = self.process_car_image(car_image, car_mask)
        car_h = car_image.size[1]
        text_image = self.process_text_image(text_image)
        text_h = text_image.size[1]
        edge_image = self.get_text_image_region(np.array(text_image)[..., 3])
        control_image = self.make_inpaint_condition(car_image, edge_image)

        return control_image, car_h, text_h, car_image, text_image


self = ControlImageGenerator()
