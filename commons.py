import io
from PIL import Image
import torch
import numpy as np


def get_model(model_name,conf=None,iou=None):
    model = None
    if model_name[0] == "y":
        model = torch.hub.load('ultralytics/yolov5', "custom", path="models/par_modele/yolov5x/best.pt")
        if conf:
            model.conf = conf
        if iou:
            model.iou = iou
    model.eval()
    return model


def transform_image_yolo(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)




