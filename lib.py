import os
import time
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import shutil
import configparser
from tqdm import tqdm
import shutil
from ultralytics import YOLO
from config_app.config import get_config

print(get_config()["YOLO"])