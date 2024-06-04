import json
import os.path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import requests
import asyncio
import aiohttp


class TextOptimizer:
    def __init__(self, url, future_size=16):
        self.url = url
        self.temperature = 0.001
        self.save_path = "./image_op_cap.json"
        self.executor = ThreadPoolExecutor(max_workers=future_size)

    async def fetch(self, session, data):
        async with session.post(self.url, json=data) as response:
            return await response.json()

    def request_data(self, prompt):
        return requests.post(self.url, json=prompt).json()

    def generate_prompt(self, car_title, cap):
        prompt = (f"你是优秀的文字工作者，组合以下内容，不超过30字;"
                  f"车的名称：{car_title};"
                  f"车的描述：{cap};").replace(" ", "")
        return {
            "model": "Qwen1.5-32B-Chat-GPTQ-Int4",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "stream": False
        }

    def clear_image_path_cap_dict(self, image_path_cap_dict: Dict[str, any], clue_id_title_dict: Dict[str, any]):
        result = {}
        for image_path, cap in image_path_cap_dict.items():
            if isinstance(cap, dict) and "caption" in cap:
                clue_id = image_path.split("/")[-2]
                if clue_id in clue_id_title_dict:
                    result[image_path] = dict(cap=cap["caption"], car_title=clue_id_title_dict[clue_id])
        return result

    async def process_image_path_cap_dict(
            self, image_path_cap_dict: Dict[str, any]):
        async with aiohttp.ClientSession() as session:
            result = {}
            for image_path, cap in image_path_cap_dict.items():
                data = self.generate_prompt(**cap)
                task = asyncio.create_task(self.fetch(session, data))
                result[image_path] = task
            real_results = await asyncio.gather(*result)
            return real_results

    def request_in_executor(self, image_path_cap_dict: Dict[str, any]):
        result = {}
        if os.path.exists(self.save_path):
            result = json.load(open(self.save_path))
        for image_path, cap in image_path_cap_dict.items():
            if image_path not in result:
                data = self.generate_prompt(**cap)
                result[image_path] = self.executor.submit(self.request_data, data)
        return result
