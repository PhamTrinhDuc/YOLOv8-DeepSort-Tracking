from lib import *
from config_app.config import get_config

config_app = get_config()

model = YOLO(config_app['YOLO']["WEIGHT"])
dir_yaml = config_app['YOLO']["DIR_YAML"]
EPOCHS = config_app['YOLO']['EPOCHS']
BATCH_SIZE = config_app['YOLO']['BATCH_SIZE']
IMG_SIZE = config_app['YOLO']['IMAGE_SIZE']
PROJECT_NAME = "./Object_Detection/results_model/yolo"
NAME = "yolov8s_det"



def train():
    results = model.train(
        data=dir_yaml,
        epochs = EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project_name = PROJECT_NAME,
        name=NAME
    )

def eval():
    model_path = os.path.join(PROJECT_NAME, NAME, "weights/best.pt")
    model = YOLO(model_path)
    model.val(
        project=PROJECT_NAME,
        name="detect/val"
    )

def inference(path_img):
    
    results = model.predict(
        path_img,
        project=PROJECT_NAME,
        name="detect/predict",
        save=True
    )

    score = results.conf.cpu().numpy()
    class_id = results.boxes.cls.cpu().numpy()
    results.append({
        "path image": path_img,
        "score": score,
        "class_id": class_id
    })

    return results
    


