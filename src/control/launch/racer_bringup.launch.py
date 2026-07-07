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

Add `yolo:=true` to ALSO start yolo_node (객체검출). 검출은 대시보드 YOLO 패널에
표시되지만, interpret 는 yolo_enabled=false 라 제어(정지/감속)엔 관여하지 않는다.
검증 후 `ros2 param set /interpret_node yolo_enabled true` 로 제어 연동을 켠다.

Usage:
    ros2 launch control racer_bringup.launch.py                 # web + perception + judgment
    ros2 launch control racer_bringup.launch.py drive:=true     # + actuator
    ros2 launch control racer_bringup.launch.py yolo:=true      # + object detection (표시만)
    ros2 launch control racer_bringup.launch.py aruco:=true     # + ArUco 마커 검출 (표시만)
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
    'kp_offset_curve': 0.45,  # 곡선 kp(offset 교정 강화)
    'sched_offset_lo': 0.3,
    'sched_offset_hi': 0.6,
    # 곡선 감속("코너 브레이크"): w↑ 에서 throttle 을 이 비율로 낮춰 라인 유지.
    'curve_throttle_scale': 0.9,  # 곡선 감속 비율(cruise 0.19 x 0.9 = 0.17)
    'cruise_throttle': 0.0,  # SAFE: no motion until raised via param
}


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    cfg = {'vehicle_config_file': vehicle_config_path}

    drive = LaunchConfiguration('drive')
    debug_overlay = LaunchConfiguration('debug_overlay')
    yolo = LaunchConfiguration('yolo')
    aruco = LaunchConfiguration('aruco')

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
                        'pane on the dashboard. interpret 는 yolo_enabled=false 라 '
                        '검출을 표시만 하고 제어(정지/감속)엔 관여하지 않는다 — 검증 후 '
                        'ros2 param set /interpret_node yolo_enabled true 로 켠다.',
        ),
        DeclareLaunchArgument(
            'aruco', default_value='false',
            description='Also start aruco_node (ArUco 마커 검출) + show its overlay '
                        'pane on the dashboard. /detected_marker_id + /aruco_stop 를 '
                        '발행하지만 지금은 어떤 노드도 구독하지 않아 주행엔 영향 없다 '
                        '(interpret 연동은 추후).',
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
             parameters=[cfg, INTERPRET_PARAMS]),
        Node(package='battery', executable='battery_node', name='battery_node',
             output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node',
             output='screen',
             parameters=[cfg, {
                 'opencv_edge_topic': edge_topic,
                 # yolo:=true / aruco:=true 일 때만 대시보드에 해당 오버레이 패널 표시.
                 'yolo_debug': ParameterValue(yolo, value_type=bool),
                 'aruco_debug': ParameterValue(aruco, value_type=bool),
             }]),

        # --- Object detection (only with yolo:=true) ---
        # 카메라 원본(/camera/image/compressed)을 직접 구독하는 독립 인지 브랜치.
        # 검출을 /yolo/detections + /yolo/image/debug 로 발행. 모델이 없으면 빈 검출만
        # 내보내며 죽지 않는다(models/README.md 참고).
        Node(package='yolo', executable='yolo_node', name='yolo_node',
             output='screen', condition=IfCondition(yolo)),

        # --- ArUco marker detection (only with aruco:=true) ---
        # 카메라 원본을 직접 구독하는 독립 인지 브랜치. /detected_marker_id + /aruco_stop
        # + /aruco/image/debug 발행. 현재 구독자 없어 주행 무영향(추후 interpret 연동).
        Node(package='aruco', executable='aruco_node', name='aruco_node',
             output='screen', condition=IfCondition(aruco)),

        # --- Actuator driver (only with drive:=true) ---
        Node(package='control', executable='control_node', name='control_node',
             output='screen', parameters=[cfg],
             condition=IfCondition(drive)),
    ])
