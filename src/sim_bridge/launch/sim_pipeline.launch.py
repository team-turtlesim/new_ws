"""시뮬 주행 파이프라인 런치(new_ws 쪽).

가제보(pinky)는 별도 터미널에서 이미 돌고 있다고 가정한다. 이 런치는 실차 팀
차선주행 노드 체인 + 브리지(시뮬 전용) + 아루코 정지(공용)를 띄운다:

  /camera/image_raw ─(image_bridge)→ camera/image/compressed
     → opencv_node → /opencv/image/edge
     → lane_node   → /lane/detection
     → interpret   → /lane_info(디버그) + /control(제어)
                        │
  /camera/image_raw ─(aruco_detector)→ /aruco_stop
                        ↓            ↓
                  (mission_override) → /control_safe
                        └(cmd_bridge)→ /cmd_vel  → 가제보 로봇

2026-07-07 실차팀 코드 개편: lane_follow_node 폐지, PID(제어결정)가 interpret 로
합쳐졌다. 이제 interpret 가 한 콜백에서 판단(/lane_info)과 제어(/control)를 모두
발행한다(이벤트구동). 그래서 cruise_throttle 은 interpret_node 로 넘긴다.

- 실차 노드(opencv/lane/interpret)는 손대지 않는다.
- 아루코(감지+정지)는 sim_bridge 밖 'aruco' 패키지 → 시뮬/실차 공용. 브리지를 빼도 남는다.
- mission_override 는 항상 돈다(/control -> /control_safe 통로). use_aruco=false 면 정지
  로직 없이 통과만 하고 detector 는 안 띄운다.
- 안전을 위해 cruise_throttle 은 기본 0.0(조향 검증 후 올린다).
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path():
    # sim_bridge/launch/../../../../src/config/vehicle_config.yaml 형태를 찾는다.
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return ''


def generate_launch_description():
    vehicle_config = get_vehicle_config_path()

    cruise_throttle = LaunchConfiguration('cruise_throttle')
    steer_sign = LaunchConfiguration('steer_sign')

    common = [{'vehicle_config_file': vehicle_config}] if vehicle_config else []

    return LaunchDescription([
        DeclareLaunchArgument(
            'cruise_throttle', default_value='0.0',
            description='interpret 순항 스로틀. 조향 검증 전엔 0.0 유지.',
        ),
        DeclareLaunchArgument(
            'steer_sign', default_value='1.0',
            description='cmd_bridge 조향 부호. 시뮬에서 반대로 돌면 -1.0 로.',
        ),
        DeclareLaunchArgument(
            'viewer', default_value='false',
            description='cv2 디버그 창(debug_viewer) 같이 띄울지. 헤드리스면 false 유지.',
        ),
        DeclareLaunchArgument(
            'use_aruco', default_value='true',
            description='아루코 장애물 정지 사용. true 면 detector 띄우고 마커 보이면 정지.',
        ),
        DeclareLaunchArgument(
            'use_sign', default_value='true',
            description='YOLO 신호등 정지 사용. true 면 sign_detector 띄우고 red_light 면 정지(green 까지).',
        ),
        DeclareLaunchArgument(
            'use_arrow', default_value='true',
            description='화살표 갈림길 방향 사용. true 면 left/right_sign 보이는 동안 steering offset(전진 조향).',
        ),

        # 입력 브리지: 가제보 raw -> 압축(JPEG)
        Node(
            package='sim_bridge', executable='image_bridge', name='sim_image_bridge',
            output='screen',
        ),

        # 실차 전처리: 압축영상 -> 차선 이진영상(color mask)
        Node(
            package='opencv', executable='opencv_node', name='opencv_node',
            output='screen', parameters=common,
        ),
        # 실차 차선검출
        Node(
            package='lane_detection', executable='lane_node', name='lane_node',
            output='screen', parameters=common,
        ),
        # 실차 판단(EMA/필터) + 제어결정(offset PID) -> /lane_info + /control
        # 2026-07-07 개편: 구 lane_follow_node 가 하던 PID 가 여기로 합쳐졌다.
        # cruise_throttle 은 여기로 넘긴다(조향 검증 전엔 0.0). interpret 자체
        # steer_sign 은 실차 서보 극성(-1.0) 기본값 사용 — 시뮬 극성은 cmd_bridge 담당.
        Node(
            package='interpret', executable='interpret_node', name='interpret_node',
            output='screen',
            parameters=common + [{'cruise_throttle': cruise_throttle}],
        ),

        # 아루코 감지(공용 패키지): /camera/image_raw -> /aruco_stop. use_aruco 일 때만.
        Node(
            package='aruco', executable='aruco_detector_node', name='aruco_detector_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_aruco')),
        ),
        # YOLO 표지판/신호등 인지: /camera/image_raw -> /detected_sign. use_sign 일 때만.
        Node(
            package='sign', executable='sign_detector_node', name='sign_detector_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_sign')),
        ),
        # 미션 레벨 안전 오버라이드(공용): /control(+/aruco_stop +/detected_sign) -> /control_safe.
        # 항상 실행. use_aruco/use_sign=false 면 해당 미션 무시. red_light 면 정지(green 까지 래치).
        Node(
            package='aruco', executable='mission_override', name='mission_override',
            output='screen',
            parameters=[{
                'use_aruco': LaunchConfiguration('use_aruco'),
                'use_sign': LaunchConfiguration('use_sign'),
                'use_arrow': LaunchConfiguration('use_arrow'),
            }],
        ),

        # 출력 브리지: /control_safe -> /cmd_vel (control_node 하드웨어 대체)
        Node(
            package='sim_bridge', executable='cmd_bridge', name='sim_cmd_bridge',
            output='screen',
            parameters=[{'steer_sign': steer_sign}],
        ),

        # (선택) cv2 디버그 뷰어: viewer:=true 일 때만. 기본 off 로 헤드리스 유지.
        Node(
            package='sim_bridge', executable='debug_viewer', name='debug_viewer',
            output='screen',
            condition=IfCondition(LaunchConfiguration('viewer')),
        ),
    ])
