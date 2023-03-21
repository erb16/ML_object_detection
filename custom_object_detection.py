from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt")
detector.setJsonPath("hololens-yolo_yolov3_detection_config.json")
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")