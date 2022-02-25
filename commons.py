import io
from PIL import Image
import torch
import numpy as np


def get_model():
    model = torch.hub.load('ultralytics/yolov5', "custom", path="models/best.pt")
    model.conf = 0.5
    model.eval()
    return model


def transform_image_yolo(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)




