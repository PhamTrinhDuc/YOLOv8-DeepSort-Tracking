from lib import *


# tao format data cho YOLO tu dataset MOT17

class convert_data:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    # lay ra cac folder FRCNN va xoa cac folder khac
    def process_all_folder(self, mode):
        # root_folder = "./MOT17/train" | "./MOT17/val"

        for subfolder in tqdm(os.listdir(self.root_folder)):
            folder_path = os.path.join(self.root_folder, )
            if "FRCNN" not in subfolder:
                os.system(f'rm-rf {folder_path}')
                continue
            if os.path.isdir(folder_path):
                self.process_folder(folder_path, mode)
    

    # tạo 2 folder train, val. train và val chứa images và labels
    def creater_folder_for_yolo(self, folder_root):
        # folder_root = "dataset_FRCNN"
        modes = ['train', 'val']
        datas = ['images', 'labels']

        path_list = []
        for mode in modes:
            path_mode = os.path.join(folder_root, mode)
            for data in datas:
                path_data = os.path.join(path_mode, data)
                os.makedirs(path_data, exist_ok=True)
                path_list.append(path_data)

        return path_list




    # xử lí các folder FRCNN
    def process_folder(self, folder_path, mode):
        # folder_path = "./MOT17/train/...FRCNN" | "folder_path = "./MOT17/val/...FRCNN"

        # lấy ra các thông tin ảnh từ file seqinfor.ini
        file_ini = "seqinfo.ini"
        config = configparser.ConfigParser()
        config.read(os.path.join(folder_path, file_ini)) # read file .ini
        img_width = int(config["Sequence"]["imWidth"])
        img_height = int(config["Sequence"]["imgHeight"])

        # format gt: 'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility'
        gt_path = os.path.join(folder_path, "det/det.txt")
        gt_dataframe = pd.read_csv(
            gt_path,
            header=None,
            names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        )

        folder_target = os.path.join("./dataset_FRCNN", mode)

        label_folder_target = os.path.join(folder_target, 'labels')
        image_folder_target = os.path.join(folder_target, 'images')


        #  lấy ra các file ảnh và di chuyển đến folder target image
        dir_folder_image = os.path.join(folder_path, "img1")
        for file_img in os.listdir(dir_folder_image):
            path_image = os.path.join(dir_folder_image, file_img)
            shutil.move(path_image, image_folder_target)


        # xử lí bb và tao file txt, di chuyển đến folder target label
        for num_frame in gt_dataframe['frame'].unique():
            gt_frame = gt_dataframe[gt_dataframe['frame'] == num_frame]
            label_file = os.path.join(label_folder_target, f"{num_frame:06d}.txt")

            with open(label_file, mode='w') as file:
                for _, row in gt_frame.iterrows():
                    bbs = self.convert_gt(row, img_width, img_height)
                    file.write(f"0 {bbs[0]}, {bbs[1]}, {bbs[2]}, {bbs[3]}\n")

    
    # chuyển các thông số của bb về dạng chuẩn format của YOLO
    def convert_gt(row, img_width, img_height):
        x_center = (row["bb_left"] + row["bb_width"]) / 2
        y_center = (row["bb_top"] + row["bb_height"]) / 2

        x_center /= img_width
        y_center /= img_height

        bb_width_norm = row["bb_width"] / img_width
        bb_height_norm /= row["bb_height"] / img_height

        x_center = max(min(x_center, 1), 0)
        y_cneter = max(min(y_center, 1), 0)
        bb_width_norm = max(min(bb_width_norm, 1), 0)
        bb_height_norm = max(min(bb_height_norm, 1), 0)

        return x_center, y_center, bb_width_norm, bb_height_norm
    
    
    # tạo file yaml cho YOLO
    def create_file_yaml(self):
        num_classes = 1
        class_labels = ['Human']

        path_yaml = "./dataset_FRCNN/data.yml"
        data_yaml = {
            'path': "./dataset_FRCNN",
            'train': "train/images",
            'val': 'val/images',
            'nc': num_classes,
            'names': class_labels
        }

        with open(path_yaml, mode='w') as f:
            yaml.dump(data=data_yaml, default_flow_style=False)


    def process_dataset(self):
        self.creater_folder_for_yolo()
        self.process_all_folder()
        self.create_file_yaml()

