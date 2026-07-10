# 노란 램프(진입로) 인지 결과 — 판단/시간필터 전의 '그 순간' 기하값.
# ramp_detection 노드가 /opencv/image/yellow 를 구독해 매 프레임 발행한다.
# interpret 가 구독해 진입 판단(감속/커밋/후퇴)에 쓴다.
#
# 이 메시지는 '노란 것에 대해 무엇이 보이는가'만 말한다. 진입할지 말지는 말하지 않는다.
std_msgs/Header header

uint32 image_width       # 입력 영상 폭 (px)
uint32 image_height      # 입력 영상 높이 (px)
float32 center_x         # 이미지 중심 x (px)

# --- 근거리 ROI: 조향에 쓸 수 있는 목표 ---
float32 ramp_center_px   # 노란 램프 차로 중심 x (px); 선이 없으면 -1.0
# 정규화 횡오차 [-1.0 ~ 1.0]. 양수(+) = 램프 중심이 이미지 오른쪽 -> 우조향.
# lane_detection 의 raw_offset 과 같은 부호 규약. 시간필터 없음.
float32 raw_offset
float32 confidence       # 0.0~1.0. 앵커 선의 스캔행 커버리지 기반. 선 없으면 0.0
bool both_boundaries     # 두 경계 사이 중앙잡기인가(false = 한 선 앵커 + 반 차선폭)
uint8 line_count         # 근거리 ROI 에서 분리된 노란 선 개수(잡음 필터 통과분)
float32 anchor_coverage  # 앵커 선이 걸린 스캔행 비율 0.0~1.0 (실선일수록 1 에 가까움)
float32 near_yellow_frac # 근거리 ROI 의 노란 픽셀 비율 0.0~1.0

# 조감도(IPM)에서 잰 앵커 경계선의 진행방향 각도 [rad]. 지면 실각도라 믿을 수 있다.
# 음수(-) = 선이 왼쪽으로 기울어짐 -> 좌조향 필요. 양수(+) = 오른쪽.
# raw_offset 과 같은 부호 규약(음수=좌). 선이 없으면 0.0.
# 원본 화면에서 잰 heading 은 원근 왜곡으로 못 쓴다(2026-07-06 제거) — 이건 다르다.
float32 bev_heading

# --- 원거리 밴드: 이벤트 감지 전용. 조향에 절대 쓰지 않는다 ---
# 원근 왜곡이 큰 상단 밴드라 기하값은 못 믿는다. '저 앞에 노랑이 있나'만 본다.
float32 lookahead_yellow_frac  # 상단 밴드의 노란 픽셀 비율 0.0~1.0
# 상단 밴드 노란 픽셀의 무게중심 x, [-1.0 ~ 1.0] 정규화. 노랑 없으면 0.0
# (lookahead_yellow_frac 이 0 에 가까우면 이 값은 의미 없음 — 반드시 함께 볼 것)
float32 lookahead_centroid_x
