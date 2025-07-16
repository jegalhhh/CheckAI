from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. 모델 로드
model = YOLO("runs/detect/train4/weights/best.pt")



# 2. 이미지 경로 지정
img_path = "1826581.jpgcn"  # 테스트할 이미지 경로

# 3. 추론 실행
results = model(img_path)

# 4. 결과에서 첫 번째 이미지의 바운딩 박스 추출
boxes = results[0].boxes
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR이므로 RGB로 변환

for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
    conf = box.conf[0].item()                    # confidence
    cls = int(box.cls[0].item())                 # class index
    label = model.names[cls]                     # class name
    
    # 바운딩 박스 그리기
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(0, 255, 0), thickness=2)
    cv2.putText(img, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# 5. 시각화
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("YOLOv8 Detection Result")
plt.show()
