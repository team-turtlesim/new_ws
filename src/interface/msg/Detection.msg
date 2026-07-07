# 단일 객체 검출 (YOLO). 좌표는 추론에 쓴 카메라 원본 해상도(예: 640x480) 픽셀 기준.
string  label        # 클래스 이름 (labels.txt 매핑; 없으면 "id:<class_id>")
int32   class_id     # 클래스 인덱스
float32 confidence   # 검출 신뢰도 0.0~1.0

# 바운딩 박스 (px). (x, y) = 좌상단 모서리, (width, height) = 박스 크기.
float32 x
float32 y
float32 width
float32 height
