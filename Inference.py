from lib import *
import Tracker 
from config_app.config import get_config
import Detector
config_app = get_config()

# target: ve bounding va fill bounding cho anh va text, can bang 2 phan nay bang mask_alpha
 
def draw_detection(img, bboxes, scores, class_ids, ids, classes =['objects'], mask_alpha=0.7):
    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(classes), 3))


    mask_img = img.copy()
    det_img = img.copy()

    size = min([height, width]) * 0.0006
    text_thinkness = int(min([height, width]) * 0.001)

    # Draw bounding boxes and lables detections
    for bbox, score, class_id, id in zip(
        bboxes, scores, class_ids, ids
    ):
        color = colors[class_id]

        x1, y1, x2, y2 = bbox.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, pt1=(x1, y1), pt2=(x2, y2),
                      color=color, thickness=2)
        # Draw fill rectangle in mask image\
        cv2.rectangle(mask_img, pt1=(x1, y1), pt2=(x2, y2), 
                      color=color, thickness=-1)
        
        label = classes[class_id]
        caption = f"{label} {int(score * 100)}% ID: {id}"

        (t_width, t_heigth), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thinkness)
        
        t_heigth = int(t_heigth* 1.2)

        cv2.rectangle(det_img, pt1=(x1, y1), pt2=(x1+t_width, y1-t_heigth), color=color, thickness=2)
        cv2.rectangle(mask_img, pt1=(x1, y1), pt2=(x1+t_width, y1-t_heigth), color=color, thickness=-1)
        
        cv2.putText(img=det_img, text=caption, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=size, color=(255, 255, 255), thickness=text_thinkness, lineType=cv2.LINE_AA)
        
        cv2.putText(mask_img, text=caption, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=size, color=(255, 255, 255), thickness=text_thinkness, lineType=cv2.LINE_AA)
        
    return cv2.addWeighted(det_img, mask_alpha, mask_img, 1-mask_alpha, 0)


def video_tracking(video_path, detector, tracker, is_save_results, save_dir=""):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_save_results:
        # os.makedirs(save_dir, exist_ok=True)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_results_name = "output_video.avi"
        save_video_path = os.path.join(save_dir, save_results_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))


    tracking_ids = np.array([], dtype=np.int32)
    all_tracking_results = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
    
        bboxes, scores, class_ids = detector.inference(frame)
        tracked_pred = tracker.tracking(frame, bboxes, scores, class_ids)

        if tracked_pred.size > 0:
            bboxes = tracked_pred[:, :4]
            class_ids = tracked_pred[:, 4].astype(int)
            conf_scores = tracked_pred[:, 5]
            tracked_ids = tracked_pred[:, 6].astype(int)

            new_ids = np.setdiff1d(tracking_ids, tracked_ids)
            tracked_ids = np.concatenate((tracked_ids, new_ids))

            results_img = draw_detection(
                img=frame, bboxes=bboxes, scores=conf_scores, 
                class_ids=class_ids, ids=tracked_ids
            )
        else:
            results_img = frame
        
        all_tracking_results.append(tracked_pred)

        if is_save_results ==1:
            out.write(results_img)
        
        if cv2.waitKey(26) & 0xFF == ord('q'):
            break
    
    cap.release()
    if is_save_results:
        out.release()
    cv2.destroyAllWindows()

    return all_tracking_results, save_results_name

def run(path_video):
    model_path = config_app["YOLO"]["WEIGHT"]
    detector = Detector.detector()
    tracker = Tracker.tracker()

    video_test = path_video
    all_tracking_results, save_results_name = video_tracking(
        video_path=video_test,
        detector=detector,
        tracker=tracker,
        is_save_results=True
    )
    time.sleep(120)
    return save_results_name
    

if __name__ == "__main__":
    path_video = "data_test/CityRoam.mp4"
    save_results_name = run(path_video=path_video)
    print(save_results_name)