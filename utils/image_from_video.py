import subprocess
import re
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import ffmpeg
import numpy as np


@dataclass
class VideoFrameInfo:
    frame_number: int
    pts: int
    pts_time: float
    frame_type: str
    is_key_frame: bool


def check_video_frame_list_is_valid(video_frame_list, video_path):
    video_stream = [x for x in ffmpeg.probe(video_path)['streams'] if x['codec_type'] == 'video'][0]
    frame_count = int(video_stream['nb_frames'])
    if len(video_frame_list) != frame_count:
        return False
    for i in range(frame_count):
        if video_frame_list[i].frame_number != i:
            return False
    return True


def get_video_frame_info_list(video_path, is_debug: bool = False):
    # 构建 FFmpeg 命令
    cmd = ['ffmpeg', '-i', video_path, '-vf', 'showinfo', '-f', 'null', '-']

    # 运行 FFmpeg 命令并捕获输出
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # 用于匹配帧信息的正则表达式

    frame_info_regex = re.compile(
        r'n:\s*(\d+)\s+pts:\s*(\d+)\s+pts_time:(\S+)\s+pos:.*?i:(\S+)\s+iskey:(\d+)\s+type:(\S+)'
    )

    result = []
    # 逐行读取输出并解析
    while True:
        line = process.stderr.readline().decode('utf-8')
        if not line:
            break

        # 使用正则表达式匹配帧信息
        match = frame_info_regex.search(line)
        if match:
            frame_number = int(match.group(1))
            pts = int(match.group(2))
            pts_time = float(match.group(3))
            frame_type = match.group(6)
            is_key_frame = bool(int(match.group(5)))

            # 处理帧信息
            frame_info = VideoFrameInfo(frame_number, pts, pts_time, frame_type, is_key_frame)
            result.append(frame_info)

    # 确保 FFmpeg 进程已完成
    process.wait()
    return result


def get_video_frame_list(video_path) -> List[Tuple[np.ndarray, VideoFrameInfo]]:  # rgb
    video_capture = cv2.VideoCapture(video_path)
    frame_info_list = get_video_frame_info_list(video_path)
    frame_list = []
    for frame_info in frame_info_list:
        ret, frame = video_capture.read()
        if ret:
            frame_list.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_info))
        else:
            break
    video_capture.release()
    return frame_list
