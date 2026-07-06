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

Usage:
    ros2 launch control racer_bringup.launch.py                 # web + perception + judgment
    ros2 launch control racer_bringup.launch.py drive:=true     # + actuator
    ros2 launch control racer_bringup.launch.py debug_overlay:=false  # raw edge pane
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


# Tuned judgment + lateral-control params (interpret 노드가 소비).
# 이력: 2026-07-04 live-drive 게인 -> 07-05 heading 포화/편향 정리(heading 은
# 양쪽차선일 때만 신뢰, k_heading 0) -> 07-06 이벤트구동 전환 후 남은 직진
# 뱀주행(리밋사이클 ~0.5Hz)을 실측 튜닝으로 확정.
# 07-06 측정: 적분(ki)이 저속 위치루프 위상지연을 키워 리밋사이클을 유발
# (ki=0.2/0.05 offset std~0.11, ki=0 은 0.06). kp 과다는 2차 요인.
#   원본(kp0.45/kd0.12/ki0.2) std 0.110 -> 최종(kp0.25/kd0.16/ki0) std 0.060(-45%).
# 곡선은 curve_bias 로 안쪽 조준; 크로싱 심하면 0.4, 뱀주행 재발하면 0.2 로.
INTERPRET_PARAMS = {
    'debug_log': True,
    'kp_offset': 0.25,       # 직진 kp: 0.45 -> 0.25 루프게인 축소(리밋사이클 억제)
    'kd_offset': 0.16,       # 0.12 -> 0.16 감쇠 강화(offset 깨끗해 여유 있음)
    'ki_offset': 0.0,        # 0.2 -> 0.0 적분 위상지연이 뱀주행 유발 -> 비활성
    'k_heading': 0.0,        # 직진 heading 항 0(뱀주행 방지)
    'steer_smooth_alpha': 0.30,  # 0.35 -> 0.30 출력 평활 강화
    'd_offset_limit': 2.0,
    'curve_bias': 0.0,        # heading 미신뢰 -> 곡선 안쪽조준 비활성(순수 offset=0 추종)
    # 게인 스케줄링(직진<->곡선 연속 블렌딩, |heading| 기준). 07-06 곡선측정에서
    # 직진튜닝으로는 코너에서 바깥 밀림+차선이탈 확인 -> 곡선에서만 kp↑+선행조향.
    'kp_offset_curve': 0.45,  # 곡선 kp(시작값, 곡선주행으로 재튜닝)
    'k_heading_curve': 0.0,   # heading 미신뢰 -> 선행조향 비활성
    'heading_ema_alpha': 0.15,  # heading 지속성↑(단일차선 순간 토글 억제)
    'sched_heading_lo': 0.15,
    'sched_heading_hi': 0.35,
    # offset 기반 복구게인(안전망): 단일차선 곡선은 heading=0 이라 heading 스케줄이
    # 안 걸림 -> offset 이 크게 벌어지면 kp 를 올려 이탈 복구. 07-06 단일차선 곡선
    # 이탈(off -0.9, w=0, kp0.25) 대응.
    'sched_offset_lo': 0.3,
    'sched_offset_hi': 0.6,
    # 곡선 감속("코너 브레이크"): w↑ 에서 throttle 을 이 비율로 낮춰 라인 유지.
    # 07-06: kp0.45/steer0.66 로도 0.18 속도로는 급곡선 못 버팀 -> 감속 필요.
    'curve_throttle_scale': 0.9,  # 곡선 감속 비율(cruise 0.19 x 0.9 = 0.17)
    'cruise_throttle': 0.0,  # SAFE: no motion until raised via param
}


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    cfg = {'vehicle_config_file': vehicle_config_path}

    drive = LaunchConfiguration('drive')
    debug_overlay = LaunchConfiguration('debug_overlay')

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
             parameters=[cfg, {'opencv_edge_topic': edge_topic}]),

        # --- Actuator driver (only with drive:=true) ---
        Node(package='control', executable='control_node', name='control_node',
             output='screen', parameters=[cfg],
             condition=IfCondition(drive)),
    ])
