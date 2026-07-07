# 한 프레임의 YOLO 검출 결과 묶음.
# yolo_node 가 /camera/image/compressed 원본을 추론해 매 프레임(추론 성공 시) 발행한다.
# interpret 등 소비자가 구독해 정지/감속 판단에 사용한다.
# 검출이 하나도 없어도(빈 프레임) detections 를 빈 배열로 채워 발행한다 → 소비자가
# "검출 없음"과 "노드 정지(stale)"를 구분할 수 있다.
std_msgs/Header header

uint32 image_width       # 추론에 쓴 원본 영상 폭 (px) — 박스 좌표 기준계
uint32 image_height      # 추론에 쓴 원본 영상 높이 (px)

Detection[] detections   # 검출 목록 (없으면 빈 배열)
