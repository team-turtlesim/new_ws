# yolo/models

`yolo_node` 가 로드하는 ONNX 모델과 클래스 라벨을 두는 곳.

## 파일
- `labels.txt` — 한 줄에 클래스 하나(class_id 0부터). `#` 주석/빈 줄 무시. 기본은 COCO 80.
- `best.onnx` — (git 미포함) 학습된 YOLO 모델. 기본 `model_path` 가 이 파일을 찾는다.

## 모델 준비 (yolo26n 파인튜닝 → ONNX)
학습/export 는 **PC(가능하면 GPU)** 에서 하고, 산출물만 보드로 복사하는 걸 권장한다
(보드는 aarch64 CPU 전용 — [platform 제약]).

```bash
# PC 에서 (ultralytics 설치 필요)
yolo detect train model=yolo26n.pt data=my_dataset.yaml imgsz=640 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=19 simplify=True
# → best.onnx 생성. 이 파일과 (커스텀) labels.txt 를 보드의 이 폴더로 복사.
scp best.onnx labels.txt topst@<board>:~/new_ws/src/yolo/models/
```

보드에서 스톡 모델로 바로 시험하려면(커스텀 학습 전 파이프라인 검증용):
```bash
pip install ultralytics        # 무겁다(torch 동반). PC 에서 export 권장.
yolo export model=yolo26n.pt format=onnx imgsz=640 opset=19 simplify=True
mv yolo26n.onnx ~/new_ws/src/yolo/models/best.onnx
```

## 실행
```bash
pip install onnxruntime==1.18.1                      # 최초 1회 (CPU)
colcon build --packages-select yolo && source install/setup.bash
ros2 run yolo yolo_node                              # 기본 model_path = 이 폴더의 best.onnx
# 모델 경로/입력크기/임계값 오버라이드:
ros2 run yolo yolo_node --ros-args \
  -p model_path:=/path/best.onnx -p labels_path:=/path/labels.txt \
  -p conf_threshold:=0.35 -p input_size:=640
```

모델 파일이 없으면 노드는 죽지 않고 빈 검출(`DetectionArray`)만 발행하며 경고를 남긴다.
