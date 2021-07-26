import cv2
import numpy as np
import time
import math
from cvzone import *

zz=25

# Function from pysource.com------------------------------------------------------------
class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h, idd = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
           
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 30:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if(idd == 1):
                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
        return img

time.sleep(3)
    
total_filtered = []

# Create tracker object
tracker = EuclideanDistTracker()

# Load Yolo-Tiny ----------------------------------------------------------------------------------
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)   
      
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading video ------------------------------------------------------------------------
cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")
cap.set(3, 1280) # set video widht
cap.set(4, 720) # set video height

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
starting_time = time.time()
frame_id = 0
total = 0 
say=0

video_writer = cv2.VideoWriter('DRONE-SURVEILLANCE-CONTEST-VIDEO-PROCESSED.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1600,900))
while True:
    # Get frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (1600,900))
   
    
    frame_id += 1
    if frame_id % 1 != 0:
        continue
    
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00261, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(output_layers)
    
    result = []
    boxes = []
    liste = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                if frame_id < 20:
                    xx=240
                else:
                    xx=350
                yy=600
                
                if xx < center_y < yy and w > 20:
                    idd=1
                else:
                    idd=0
                    
                boxes.append([x, y, w, h, idd])
        
    boxes_ids = tracker.update(boxes)

    
    # Filtering out some issues -----------------------------------------------------------------------------
    added_ones = []
    filtered =[]
   
    for i in range(len(boxes_ids)):
        if boxes_ids[i][4] not in added_ones:
            added_ones.append(boxes_ids[i][4])
            filtered.append(boxes_ids[i])
            
    for box_id in filtered:
            x, y, w, h, id = box_id
            bboxes = [x,y,w,h]
           
            if id > 99:
                zz=33
                
            cv2.rectangle(frame, (x,y+h-20), (x+zz,y+h), (0,0,255),-1)
            cv2.putText(frame, str(id), (x+3 , y + h-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            frame = cornerRect(frame,bboxes,rt=0,colorC=(0,0,255))
   
          
    total_filtered = [*total_filtered, *filtered]

    # Total car count -------------------------------------------------------------------------------------------------------
    try:
        _,_,_,_,value = max(filtered, key=lambda item: item[4])
        if value > total:
            total = value
    except:
        pass
     
  
    # Name and no. of cars -------------------------------------------------------------------------------------------------------------
    cv2.putText(frame, "Utkarsh K", (720,80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(frame, str(total), (1465, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    
    # Video ------------------------------------------------------------------------------------------------------------------
    cv2.imshow("Awesome Car Counter for competition held by CVZONE", frame)
    video_writer.write(frame)
    
    k = cv2.waitKey(1) & 0xff

cap.release()
result.release()
cv2.destroyAllWindows()
