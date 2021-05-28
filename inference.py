import json
from commons import get_model, transform_image_yolo

idshoes_class = json.load(open('idshoes_class.json'))

def get_prediction(image_bytes, model_ins, model_name, conf=None, iou=None):
    #try:
    model = model_ins
    image = transform_image_yolo(image_bytes=image_bytes)
    if conf:
        model.conf = float(conf)
    if iou:
        model.iou = float(iou)
    outputs = model(image)
    #except Exception:
    #    return 0, 'error'
    predicted_idx = None
    probability = None

    if outputs.xyxy[0].shape[0]!=0:
        max_conf_idx = int(outputs.xyxy[0][outputs.xyxy[0].max(0)[1][4].item()][5].item())
        predicted_idx = str(max_conf_idx)
        probability = outputs.xyxy[0].max(0)[0][4].item()
    else:
        probability = 0
        predicted_idx = str(25)
        
    return idshoes_class[predicted_idx], probability

def load_model(model_name="yolov5s"):
    return get_model(model_name)
