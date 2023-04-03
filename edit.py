import cv2
import torch
import numpy as np
import math
import time
kitlenme_baslangic = None
kitlenme_suresi = 4
baudrate = 9600
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
x_target = 320
y_target = 240
Max_Satu_Point = 30
Min_Satu_Point = -30
start_time = time.monotonic()
delay = 0.2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    tensor = results.xyxy[0][0]
    if tensor is not None:
        x1, y1, x2, y2, conf, cls = tensor.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        width = x2 - x1
        height = y2 - y1
        x_left = x1
        y_top = y1
        center_x = x_left + int(width / 2)
        center_y = y_top + int(height / 2)
        x_distance = center_x - x_target
        y_distance = center_y - y_target
        if center_x < x_target:
            x_direction = "right"
            x_angle = min(x_distance * 0.06, 30)
        else:
            x_direction = "left"
            x_angle = max(x_distance * 0.06, -30)
        if center_y < y_target:
            y_direction = "down"
            y_angle = min(y_distance * 0.06, 30)
        else:
            y_direction = "up"
            y_angle = max(y_distance * 0.06, -30)
        x_angle = max(min(x_angle, Max_Satu_Point), Min_Satu_Point)
        y_angle = max(min(y_angle, Max_Satu_Point), Min_Satu_Point)
        data = "{},{};".format(x_angle, y_angle)
        if center_x >= 160 and center_x <= 480 and center_y >= 48 and center_y <= 432:
            x_uzunluk = x1 - x2
            y_uzunluk = y1 - y2
            if x_uzunluk > 16 and y_uzunluk > 19.2:
                if kitlenme_baslangic is None:
                    kitlenme_baslangic = time.monotonic()
                elif time.monotonic() - kitlenme_baslangic >= kitlenme_suresi:
                    print("Hedef kitlendi")
                    cv2.putText(frame, "Kitlendi", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                kitlenme_baslangic = None
    current_time = time.monotonic()
    cv2.putText(frame, f"Aileron: {y_angle:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Elevator: {x_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (160, 48), (480, 432), (0, 255, 0), 3)
    cv2.imshow("yolo", frame)
    if current_time - start_time >= delay:
        data = "{},{};".format(x_angle, y_angle)
        print("data:", data)
        start_time = time.monotonic()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
