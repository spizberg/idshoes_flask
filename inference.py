# import json
from commons import get_model, transform_image_yolo

# idshoes_class = json.load(open('idshoes_class.json'))

def get_prediction(image_bytes, model_ins, conf=None, iou=None):
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
    class_name = None
    probability = None
    pandas_pred = outputs.pandas().xywhn[0]

    if pandas_pred.empty == False:
        probability, class_name = pandas_pred.loc[0, ["confidence", "name"]]
        class_name = "aucune" if class_name=="UNKNOWN" else class_name[:-2].lower()
    else:
        probability, class_name = 0, "aucune"
        
    return class_name, probability

