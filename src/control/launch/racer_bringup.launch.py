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

Usage:
    ros2 launch control racer_bringup.launch.py                 # web + perception + judgment
    ros2 launch control racer_bringup.launch.py drive:=true     # + actuator
    ros2 launch control racer_bringup.launch.py yolo:=true      # + object detection (제어 연동)
    ros2 launch control racer_bringup.launch.py aruco:=true     # + ArUco 마커 검출 (제어 연동)
    ros2 launch control racer_bringup.launch.py debug_overlay:=false  # raw edge pane
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
    # 원 주행(ramp) 테스트: ramp 기본 ON + 출발속도 0.19.
    # ramp_throttle_scale 0.6->1.0 이라 RAMP 실효속도 = cruise 그대로 0.19
    # (07-10 링 실험이 0.17~0.19 실효였고, 사용자가 0.19 실효로 시작 지정).
    'ramp_enabled': True,
    'ramp_throttle_scale': 1.0,
    'cruise_throttle': 0.19,  # 0.0 -> 0.19 (원 주행 출발속도, 사용자 지정)
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


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    cfg = {'vehicle_config_file': vehicle_config_path}

    drive = LaunchConfiguration('drive')
    debug_overlay = LaunchConfiguration('debug_overlay')
    yolo = LaunchConfiguration('yolo')
    aruco = LaunchConfiguration('aruco')
    ramp = LaunchConfiguration('ramp')
    ramp_start = LaunchConfiguration('ramp_start')

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
                        'pane on the dashboard. interpret 가 yolo_enabled=true 기본이라 '
                        '검출이 제어에 물린다(신호등 정지/출발, 방향표지 바이어스) — 표시만 '
                        '하려면 ros2 param set /interpret_node yolo_enabled false 로 끈다.',
        ),
        DeclareLaunchArgument(
            'aruco', default_value='false',
            description='Also start aruco_node (ArUco 마커 검출) + show its overlay '
                        'pane on the dashboard. /detected_marker_id + /aruco_stop 를 '
                        '발행하고, interpret 가 aruco_enabled=true 기본이라 정지에 물린다 '
                        '— 떼려면 ros2 param set /interpret_node aruco_enabled false.',
        ),
        DeclareLaunchArgument(
            'ramp', default_value='true',
            description='Start ramp_node (노란 진입로/링 인지). /yellow/lane 을 발행하고 '
                        'interpret 가 ramp_enabled=true(기본)로 커밋 후 노란 차선 추종한다 '
                        '(원 주행). 인지만 보고 주행에서 떼려면 '
                        'ros2 param set /interpret_node ramp_enabled false, '
                        'ramp:=false 면 노드 자체를 안 띄운다.',
        ),
        DeclareLaunchArgument(
            'ramp_start', default_value='WAIT',
            description='interpret 의 램프 시작 상태. WAIT(기본): 흰차선부터 시작해 '
                        '노랑 커밋시 RAMP 전환. RAMP: 링 위에서 출발 — arm_delay/커밋 '
                        '가드를 건너뛰고 노란차선을 첫 프레임부터 추종(링 출발 테스트).',
        ),

        # --- Perception + judgment/control-law + web (always) ---
        Node(package='camera', executable='camera_node', name='camera_node',
             output='screen', parameters=[cfg]),
        Node(package='opencv', executable='opencv_node', name='opencv_node',
             output='screen'),
        Node(package='lane_detection', executable='lane_node',
             name='lane_detection_node', output='screen', parameters=[cfg]),
        # interpret: LaneDetection(인지) -> 시간필터/판단 + offset PID(제어결정)
        #            -> LaneInfo(디버그) + Control(/control). 이벤트구동.
        Node(package='interpret', executable='interpret_node',
             name='interpret_node', output='screen',
             parameters=[cfg, INTERPRET_PARAMS, {'ramp_start_state': ramp_start}]),
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

        # --- Actuator driver (only with drive:=true) ---
        Node(package='control', executable='control_node', name='control_node',
             output='screen', parameters=[cfg],
             condition=IfCondition(drive)),
    ])
