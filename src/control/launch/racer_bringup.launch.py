"""One-shot bringup for the lane-following racer.

Default (safe) launch starts the perception + judgment + web-dashboard stack:
    camera -> opencv(edge) -> lane_detection(/lane/detection)
           -> interpret(judgment + control law) -> /lane_info + /control
    battery, monitor(web dashboard on :5000)
interpret 는 프레임 도착마다 판단(시간필터)과 제어결정(offset PID)을 한 콜백에서
수행해 /control 을 발행한다(이벤트구동 -> 예전 lane_follow 의 고정 20Hz 타이머
위상지연 + /lane_info 홉 제거). 하드웨어(액추에이터)는 control_node 만 만진다.
The monitor "edge" pane is pointed at the lane-detection debug overlay so the
dashboard shows ROI line + fitted lanes + centre.

Add `drive:=true` to ALSO start the actuator driver (control_node). Even then the
car does NOT move until cruise_throttle is raised (defaults to 0.0) — interpret
always publishes /control so you can validate steering on the dashboard first,
then:
    ros2 param set /interpret_node cruise_throttle 0.17

control_node 에는 stale 워치독이 있어, interpret(이벤트구동)이 프레임 끊김으로
발행을 멈추면 자동으로 조향 중립 + 정지한다.

Add `yolo:=true` to ALSO start yolo_node (객체검출). interpret 는 yolo_enabled=true 가
기본이라 검출이 곧바로 제어에 물린다: 신호등 정지/출발 + 방향표지 갈래선택(바이어스).
표시만 하고 제어에서 떼려면 `ros2 param set /interpret_node yolo_enabled false`.

Add `aruco:=true` 도 마찬가지로 aruco_enabled=true 가 기본이라 /aruco_stop 이 곧바로
정지에 물린다. 떼려면 `ros2 param set /interpret_node aruco_enabled false`.

기본값은 '흰 아웃라인 주행'에 맞춰져 있다: ramp/aruco 는 기본 OFF, 초록불 출발
게이트(green_start)도 기본 OFF 라 throttle 만으로 출발한다. cruise 는 throttle 인자
(기본 0.0=정지)로 런치 때 지정하거나 실행 중 param 으로 올린다.

Usage:
    # 오늘(흰라인 반복주행): 차는 정지로 뜨고, throttle 로 직접 출발/속도 지정.
    ros2 launch control racer_bringup.launch.py drive:=true yolo:=false
    ros2 param set /interpret_node cruise_throttle 0.15   # 실행 중 라이브로 출발/속도
    #  또는 런치 때 바로:  ros2 launch ... drive:=true yolo:=false throttle:=0.15
    ros2 launch control racer_bringup.launch.py drive:=true green_start:=true  # 초록불 출발 유지
    ros2 launch control racer_bringup.launch.py ramp:=true ramp_start:=RAMP    # 원(링) 주행 복귀
    ros2 launch control racer_bringup.launch.py debug_overlay:=false           # raw edge pane
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


# Tuned judgment + lateral-control params (interpret 노드가 소비).
# 이력: 2026-07-04 live-drive 게인 -> 07-05 heading 편향 정리 -> 07-06 이벤트구동
# 전환 후 직진 뱀주행(리밋사이클 ~0.5Hz) 실측 튜닝, 그리고 heading 신뢰불가
# 판명으로 offset 전용 전환.
# 07-06 측정: 적분(ki)이 저속 위치루프 위상지연을 키워 리밋사이클 유발
# (ki=0.2/0.05 offset std~0.11, ki=0 은 0.06). kp 과다는 2차 요인.
#   원본(kp0.45/kd0.12/ki0.2) std 0.110 -> 최종(kp0.25/kd0.16/ki0) std 0.060(-45%).
# 곡선: |offset| 스케줄로 kp 부스트 + 곡선감속(반응형). heading 은 미사용(제거됨).
INTERPRET_PARAMS = {
    'debug_log': True,
    'kp_offset': 0.25,       # 직진 kp: 0.45 -> 0.25 루프게인 축소(리밋사이클 억제)
    'kd_offset': 0.16,       # 0.12 -> 0.16 감쇠 강화(offset 깨끗해 여유 있음)
    'ki_offset': 0.0,        # 0.2 -> 0.0 적분 위상지연이 뱀주행 유발 -> 비활성
    'steer_smooth_alpha': 0.30,  # 0.35 -> 0.30 출력 평활 강화
    'd_offset_limit': 2.0,
    # 게인 스케줄링(직진<->곡선, |offset| 기준). 곡선에서 offset 이 벌어지면 kp↑.
    # 07-08: 속도↑ 시 곡선 언더스티어(못 돎/이탈) -> 곡선 반응을 더 일찍(lo/hi↓) +
    # 조향여유↑(steer_limit 0.7->0.8) 로 튜닝. (라이브 실측으로 추가 조정 여지 있음)
    'kp_offset_curve': 0.45,  # 곡선 kp(offset 교정 강화)
    'sched_offset_lo': 0.15,  # 0.3 -> 0.15 곡선 반응 더 일찍 시작(언더스티어 대응)
    'sched_offset_hi': 0.30,  # 0.6 -> 0.30 곡선 게인 더 빨리 최대
    'steer_limit': 0.8,       # 0.7 -> 0.8 곡선 조향 범위 확대
    # 곡선 감속("코너 브레이크"): w↑ 에서 throttle 을 이 비율로 낮춰 라인 유지.
    'curve_throttle_scale': 0.95,  # 0.85 -> 0.95 링 곡선 감속 완화(속도 유지, 사용자 지정)
    # cruise_throttle / require_green_start / ramp_enabled 는 런치 인자(throttle,
    # green_start, ramp)로 주입한다 — 아래 interpret Node 참고(여기 static 값 없음).
    'ramp_throttle_scale': 1.0,  # RAMP 모드 실효속도 = cruise×1.0 (ramp 켤 때만 의미)
    # 초록불 출발 확정 프레임: 10 -> 2 (2026-07-14 사용자 요청, 빠른 출발).
    # ⚠️ 주의: 오검출(실측 최대 4연속)을 다 못 걸러 헛출발 위험 — conf 0.60 게이트에
    # 의존한다. 트랙에서 초록 여러 번 보여줘 헛출발 안 나는지 꼭 확인할 것.
    'green_start_frames': 2,
}


# ramp_node(노란 차선/링 인지) 튜닝 파라미터. 노드 declare_parameter 기본값을 런치에서
# 한눈에 관리·오버라이드한다. 값은 노드 기본값과 동일(동작 불변) — 여기서 바꾸면 적용된다.
# 라이브 튜닝은 여전히 `ros2 param set /ramp_detection_node <이름> <값>` 으로 가능.
RAMP_PARAMS = {
    # --- 12시 마커 검출/카운트 ---
    'marker_enabled': True,
    'marker_run_px': 120,        # 한 행 연속 노란 런이 이 px 이상이면 가로 마커 후보
    'marker_min_rows': 3,        # run≥문턱 행이 이만큼 이상이어야 진짜 마커(오검출 배제)
    'marker_cooldown_frames': 90,  # 마커 한 번 센 뒤 이 프레임 동안 재카운트 금지(≈3초)
    # --- 점선 추종(마커 구간, mcnt==1) ---
    'marker_follow_side': 'dashed',  # 실선 제외·점선 중 맨 오른쪽
    'solid_row_min': 6,          # 행수<이 값=점선, ≥=실선. 낮출수록 실선→점선 오인 감소
    'dash_aim_px': 45,           # 점선을 오른쪽 경계로 보고 차로 중심을 그 왼쪽 이 px 로
    'ring_multiline': True,      # 마커 구간 다중선 검출(좌/우 2트랙 한계 해제)
    'debug_log': False,          # per-frame 링 판단 로그(마커/점선 선택) — 필요 시 true
}


# bluesign_node(파란 표지판 트리거) 튜닝 파라미터. opencv_node 가 발행한 파란 마스크
# (/opencv/image/blue)의 상단 ROI 파란비율로 /sign/near 를 낸다. 노드 기본값과 동일(동작
# 불변) — 여기서 바꾸면 적용된다. 라이브 튜닝: ros2 param set /bluesign_node <이름> <값>.
# 트리거는 '켜기'만 하므로(오검출=CPU만 잠깐 낭비, 놓침이 더 위험) 민감하게 두는 게 안전.
BLUESIGN_PARAMS = {
    # 상단 ROI(프레임 대비 비율). 표지판은 화면 위쪽에 먼저 나타난다.
    'roi_top_frac': 0.0,
    'roi_bottom_frac': 0.5,   # 상단 절반만 본다(하단은 차선이라 무시)
    'roi_left_frac': 0.0,
    'roi_right_frac': 1.0,
    # 트리거 임계 + 디바운스/히스테리시스.
    'blue_frac_on': 0.02,     # ROI 파란비율 2% 이상 -> 후보
    'blue_frac_off': 0.01,    # 1% 미만 -> 해제(히스테리시스)
    'on_frames': 2,           # 2프레임 연속 -> near=True (민감)
    'off_frames': 8,          # 8프레임 연속 미만 -> near=False (깜빡임 억제)
    'debug_log': False,       # per-frame 파란비율/near 로그 — 튜닝 시 true
}


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    cfg = {'vehicle_config_file': vehicle_config_path}

    drive = LaunchConfiguration('drive')
    debug_overlay = LaunchConfiguration('debug_overlay')
    yolo = LaunchConfiguration('yolo')
    aruco = LaunchConfiguration('aruco')
    ramp = LaunchConfiguration('ramp')
    bluesign = LaunchConfiguration('bluesign')
    ramp_start = LaunchConfiguration('ramp_start')
    throttle = LaunchConfiguration('throttle')
    green_start = LaunchConfiguration('green_start')
    exposure = LaunchConfiguration('exposure')
    gain = LaunchConfiguration('gain')

    # monitor "edge" pane topic: lane overlay when debug_overlay, else raw edge.
    edge_topic = PythonExpression([
        "'/lane_detection/image/debug' if '", debug_overlay,
        "' == 'true' else '/opencv/image/edge'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'drive', default_value='false',
            description='Also start control_node (actuator driver).',
        ),
        DeclareLaunchArgument(
            'debug_overlay', default_value='true',
            description='Point the monitor edge pane at the lane debug overlay.',
        ),
        DeclareLaunchArgument(
            'yolo', default_value='false',
            description='Also start yolo_node (object detection) + show its overlay '
                        'pane on the dashboard. 기본 OFF (흰라인 수동주행). yolo:=true 면 '
                        'interpret 가 yolo_enabled=true 기본이라 검출이 제어에 물린다 '
                        '(신호등 정지/출발, 방향표지 바이어스).',
        ),
        DeclareLaunchArgument(
            'aruco', default_value='false',
            description='Also start aruco_node (ArUco 마커 검출) + show its overlay '
                        'pane on the dashboard. 기본 OFF (오늘 흰라인 주행). aruco:=true '
                        '면 /detected_marker_id + /aruco_stop 를 발행하고 interpret 가 '
                        'aruco_enabled=true 기본이라 정지에 물린다.',
        ),
        DeclareLaunchArgument(
            'ramp', default_value='false',
            description='Start ramp_node (노란 진입로/링 인지). 기본 OFF (오늘 흰라인 '
                        '주행). ramp:=true 면 /yellow/lane 을 발행하고 interpret 의 '
                        'ramp_enabled 이 함께 켜져(ramp 인자와 연동) 노란 차선 추종한다 '
                        '(원 주행). ramp:=false 면 노드 자체를 안 띄우고 판단쪽도 정지.',
        ),
        DeclareLaunchArgument(
            'ramp_start', default_value='WAIT',
            description='interpret 의 램프 시작 상태. WAIT(기본): 흰차선부터 시작해 '
                        '노랑 커밋시 RAMP 전환. RAMP: 링 위에서 출발 — arm_delay/커밋 '
                        '가드를 건너뛰고 노란차선을 첫 프레임부터 추종(링 출발 테스트).',
        ),
        DeclareLaunchArgument(
            'throttle', default_value='0.0',
            description='interpret 의 cruise_throttle(순항 스로틀). 기본 0.0 = 정지'
                        '(안전). 출발/속도는 여기서 지정하거나 실행 중 라이브로: '
                        'ros2 param set /interpret_node cruise_throttle 0.15',
        ),
        DeclareLaunchArgument(
            'green_start', default_value='false',
            description='초록불 출발 게이트(require_green_start). false(기본): 초록불 없이 '
                        'throttle 만으로 출발(흰라인 수동주행 — throttle 을 직접 조절). '
                        'true: 런치 직후 정지 대기하다 첫 초록불을 봐야 출발(대회 실전 '
                        '출발선 + 초록시 YOLO 자동 off). yolo:=true 일 때만 의미.',
        ),
        DeclareLaunchArgument(
            'exposure', default_value='156',
            description='USB 카메라 수동 노출(단위 100us). 저조도에서 오토 익스포저가 '
                        'fps 를 반토막 내는 걸 막아 fps 유지. ≤333 이면 30fps 유지. '
                        '0 이면 오토(기존 동작). (2026-07-14 대회장 확정값 156)',
        ),
        DeclareLaunchArgument(
            'gain', default_value='20',
            description='USB 카메라 게인(0~255). 밝기 조절의 주 레버(fps 무관, 노이즈↑). '
                        '너무 밝아 옆 조명이 잡히면 낮추고, 흰선 깜빡이면 올린다. '
                        '(2026-07-14 대회장 확정값 20). 라이브 튜닝: python3 ~/cam.py <exp> <gain>',
        ),
        DeclareLaunchArgument(
            'bluesign', default_value='false',
            description='Start bluesign_node (파란 표지판 트리거) + 대시보드 BlueSign 패널. '
                        '기본 OFF. bluesign:=true 면 opencv 의 파란 마스크(/opencv/image/blue)'
                        '상단 ROI 파란비율로 /sign/near 를 내고, interpret 이 이를 받아 갈림길 '
                        '근처에서 YOLO 전원(/yolo/active)을 켠다(ArUco 마커 대신 색 트리거). '
                        'yolo_power_gate=true(기본)일 때만 실제로 YOLO 를 깨운다.',
        ),

        # --- Perception + judgment/control-law + web (always) ---
        Node(package='camera', executable='camera_node', name='camera_node',
             output='screen',
             parameters=[cfg, {
                 'exposure_absolute': ParameterValue(exposure, value_type=int),
                 'camera_gain': ParameterValue(gain, value_type=int),
             }]),
        Node(package='opencv', executable='opencv_node', name='opencv_node',
             output='screen'),
        Node(package='lane_detection', executable='lane_node',
             name='lane_detection_node', output='screen', parameters=[cfg]),
        # interpret: LaneDetection(인지) -> 시간필터/판단 + offset PID(제어결정)
        #            -> LaneInfo(디버그) + Control(/control). 이벤트구동.
        Node(package='interpret', executable='interpret_node',
             name='interpret_node', output='screen',
             parameters=[cfg, INTERPRET_PARAMS, {
                 'ramp_start_state': ramp_start,
                 # 오늘(흰라인 반복주행): throttle 을 런치인자/param 으로 직접 지정,
                 # green_start 로 초록불 게이트 on/off, ramp_enabled 는 ramp 노드와 연동
                 # (ramp:=false 면 판단쪽 램프 로직도 완전 정지).
                 'cruise_throttle': ParameterValue(throttle, value_type=float),
                 'require_green_start': ParameterValue(green_start, value_type=bool),
                 'ramp_enabled': ParameterValue(ramp, value_type=bool),
             }]),
        Node(package='battery', executable='battery_node', name='battery_node',
             output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node',
             output='screen',
             parameters=[cfg, {
                 'opencv_edge_topic': edge_topic,
                 # yolo/aruco/ramp:=true 일 때만 대시보드에 해당 오버레이 패널 표시.
                 # edge 패널은 계속 lane_detection 오버레이(흰 차선)를 보여준다 —
                 # ramp 는 자기 패널을 따로 갖는다.
                 'yolo_debug': ParameterValue(yolo, value_type=bool),
                 'aruco_debug': ParameterValue(aruco, value_type=bool),
                 'ramp_debug': ParameterValue(ramp, value_type=bool),
                 # Ramp 패널 = 노란 마스크로 돌린 lane_node 의 오버레이.
                 'ramp_debug_topic': '/yellow_lane/image/debug',
                 # BlueSign 패널 = bluesign_node 의 파란 마스크 오버레이(ROI+파란비율+상태).
                 'bluesign_debug': ParameterValue(bluesign, value_type=bool),
                 # 대시보드 이미지 갱신 주기(ms). 낮출수록 화면 부드럽지만 CPU↑ (실제
                 # 파이프라인 fps 를 갉아먹을 수 있음). 100=~10fps(기본), 50=~20fps.
                 'image_refresh_interval_ms': 50,
             }]),

        # --- Ramp (yellow) perception (only with ramp:=true) ---
        # ramp_node 는 lane_node 의 사본이다(알고리즘 동일: 행별 클러스터 추적 + 단독선일
        # 때 학습한 차선폭으로 반대편 추정). 입력만 노란 마스크.
        # 별도 노드로 둔 이유: 앞으로 여기에 링 주행·탈출 알고리즘(12시 마커 카운트,
        # 안쪽/바깥쪽 경계 선택 등)이 들어간다. 같은 노드를 두 인스턴스로 쓰면 그 변경이
        # 트랙 검증이 끝난 흰 차선 주행까지 건드린다. 지금은 같지만 앞으로 갈라진다.
        Node(package='ramp_detection', executable='ramp_node',
             name='ramp_detection_node', output='screen',
             parameters=[cfg, RAMP_PARAMS],
             condition=IfCondition(ramp)),

        # --- Object detection (only with yolo:=true) ---
        # 카메라 원본(/camera/image/compressed)을 직접 구독하는 독립 인지 브랜치.
        # 검출을 /yolo/detections + /yolo/image/debug 로 발행. 모델이 없으면 빈 검출만
        # 내보내며 죽지 않는다(models/README.md 참고).
        Node(package='yolo', executable='yolo_node', name='yolo_node',
             output='screen', condition=IfCondition(yolo)),

        # --- ArUco marker detection (only with aruco:=true) ---
        # 카메라 원본을 직접 구독하는 독립 인지 브랜치. /detected_marker_id + /aruco_stop
        # + /aruco/image/debug 발행. interpret 가 /aruco_stop 을 구독해 정지에 반영한다.
        Node(package='aruco', executable='aruco_node', name='aruco_node',
             output='screen', condition=IfCondition(aruco)),

        # --- Blue-sign trigger (only with bluesign:=true) ---
        # opencv 의 파란 마스크(/opencv/image/blue)를 구독하는 값싼 캐스케이드 게이트.
        # 상단 ROI 파란비율을 디바운스해 /sign/near(Bool) 발행 + /bluesign/image/debug 오버레이.
        # interpret 이 /sign/near 를 받아 갈림길 근처에서 YOLO 전원을 켠다(색 기반 YOLO 깨우기).
        Node(package='bluesign', executable='bluesign_node', name='bluesign_node',
             output='screen', parameters=[BLUESIGN_PARAMS],
             condition=IfCondition(bluesign)),

        # --- Actuator driver (only with drive:=true) ---
        Node(package='control', executable='control_node', name='control_node',
             output='screen', parameters=[cfg],
             condition=IfCondition(drive)),
    ])
