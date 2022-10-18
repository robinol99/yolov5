
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from PIL import Image, ImageDraw

detections_dir = "./runs/detect/yolo_coco_dataset_inference3"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))
plt.imshow(np.array(random_detection_image))