import os
import shutil
from modelscope.utils.hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile

pretrained_model = 'damo/ofa_pretrain_base_zh'
pretrain_path = snapshot_download(pretrained_model, revision='v1.0.2')
task_model = 'damo/ofa_image-caption_muge_base_zh'
task_path = snapshot_download(task_model)

shutil.copy(os.path.join(task_path, ModelFile.CONFIGURATION), # 将任务的配置覆盖预训练模型的配置
            os.path.join(pretrain_path, ModelFile.CONFIGURATION))

ofa_pipe = pipeline(Tasks.image_captioning, model=pretrain_path)
result = ofa_pipe('http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/image-captioning/donuts.jpg')
print(result[OutputKeys.CAPTION])