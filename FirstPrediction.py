import torch
import torchvision
from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "image.png"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)


# from imageai.Detection import ObjectDetection
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.loadModel()
# custom = detector.CustomObjects(mouse=True, person=True)
# detections = detector.detectObjectsFromImage(input_image="image.png", output_image_path="imagenew.jpg", minimum_percentage_probability=30)
#

from imageai.Detection import ObjectDetection
# from imageai.Detection.Custom import CustomObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()

# detector = CustomObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(os.path.join(execution_path, "tiny-yolov3.pt"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.png"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=1)
# custom = detector.CustomObjects(mouse=True, laptop=True, keyboard=True, phone=True)
# detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=os.path.join(execution_path, "image.png"), output_image_path=os.path.join(execution_path, "image3new-custom.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")