from lib import *
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
from config_app.config import get_config

config_app = get_config()

model_path = config_app['DEEP_SORT']['MODEL_PATH']
max_cosine_distance = config_app['DEEP_SORT']['MAX_COSINE_DISTANCE']
classes = config_app['DEEP_SORT']['CLASSES']
nn_budget = None

class tracker:
    def __init__(self):
        # trích xuất đặc trưng từ bbox
        self.encoder = gdet.create_box_encoder(model_filename=model_path, batch_size=1)

        # độ đo 
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            'cosine', matching_threshold=max_cosine_distance,
            budget=nn_budget
        )

        # quản lí các track, thực hiện các bước liên kết và cập nhật
        self.tracker = Tracker(self.metric)

        key_list = []
        val_list = []
        for ID, class_name in enumerate(classes):
            key_list.append(ID)
            val_list.append(class_name)


        self.key_list = key_list
        self.val_list = val_list
    
    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)

        detections = [Detection(bbox, score, class_id, feature) 
                      for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)]
        
        self.tracker.predict() # dự đoán vị trí track 
        self.tracker.update(detections=detections) # phát hiện các track hoặc tạo các track mới

        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr() # top left bottom right
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(
                bbox.tolist() + [class_id, conf_score, tracking_id]
            )

        tracked_bboxes = np.array(tracked_bboxes)
        return tracked_bboxes
    
    




