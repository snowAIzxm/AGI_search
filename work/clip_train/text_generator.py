class TextGenerator:
    def __init__(self):
        pass

    def init_blip2(self):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model_id = 'damo/mplug_image-captioning_coco_base_zh'
        pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)
        return pipeline_caption

