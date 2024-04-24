from lib import *
from config_app.config import get_config



config_app = get_config()
weight_yolo = config_app['YOLO']['WEIGHT']



class detector:
    def __init__(self):
        self.model = YOLO(weight_yolo)

    # chuẩn bị các thông số cho deepsort
    def inference(self, frame_img):
        results = self.model(frame_img, verbose=False)[0]
        bboxes = results.boxes.xywh.cpu().numpy()
        # bboxes: x_center, y_center, width, height -> x_min, y_min, widht, height
        bboxes[:, :2] = bboxes[:, :2] - (bboxes[:, 2:] / 2) 
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        return bboxes, scores, class_ids
