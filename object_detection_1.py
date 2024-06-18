import cv2
import matplotlib.pyplot as plt

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean(127.5)
model.setInputSwapRB(True)


classes = []
with open("label.txt","rt") as fpt:
    classes = fpt.read().rstrip("\n").split("\n")
# print(classes)

img = cv2.imread("street.jpg")

classIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)

for classInd, conf, boxes in zip(classIndex, confidence, bbox):
    x,y = boxes[0],boxes[1]
    w,h = boxes[2],boxes[3]
    
    text = classes[classInd-1]
    cv2.putText(img, text,(x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 1)
    cv2.rectangle(img, (x,y),(x+w, y+h),(0,255,0), 4)
    
cv2.imshow("frame",img)
cv2.waitKey(0)
