import cv2
from ultralytics import YOLO


model = YOLO("my_model/train/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Draw boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Pen Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()