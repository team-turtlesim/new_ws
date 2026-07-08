"""mission_override: 미션 레벨 안전 오버라이드 미들웨어 (시뮬/실차 공용).

차선주행 노드의 주행명령(/control)과 미션 감지 신호를 받아, 정지가 필요하면
주행명령을 눌러 최종 주행명령(/control_safe)을 발행한다.

이 노드는 특정 감지기에 종속되지 않는 "미션 레벨" 계층이다. 현재 병합하는 미션:
  - 아루코 동적 장애물 정지: /aruco_stop(Bool) — 보이는 동안 정지.
  - YOLO 신호등 정지(2026-07-08 추가): /detected_sign(String) — red_light 정지
    래치, green_light 해제. 검출은 프레임마다 끊기므로 마지막 신호 상태를 유지.
  - 화살표 갈림길 방향(2026-07-08 추가): /detected_sign(String) — left/right_sign 이
    보이는 '동안만' steering 을 그 방향으로 offset. 갈림길이 잠깐 갈렸다 합쳐지는
    구조라 살짝 치우치기만 하면 됨. throttle 은 안 건드림(전진하며 조향 = 제자리
    회전 금지, 실차 Ackermann 특성). 안 보이면(타임아웃) 자동으로 차선주행 복귀.
앞으로 다른 미션 신호도 같은 방식으로 이 노드에서 병합한다. 그래서 이름을
aruco_override 가 아니라 mission_override 로 둔다.
(/control 발행 주체는 2026-07-07 개편으로 lane_follow_node -> interpret_node 로
바뀌었으나, 토픽/타입이 같아 이 노드는 무관하게 동작한다.)

우선순위(기존 정지 로직을 절대 방해하지 않게):
  1순위 정지(최우선): 아루코 OR red_light -> throttle=0. 화살표 무시.
  2순위 방향: 정지 아닐 때만, 화살표 보이면 steering 에 offset(throttle 유지).
  3순위 평소: 미션 없음 -> /control 그대로 통과.

설계 원칙:
  * 실차 코드(interpret 등)는 안 건드린다. 주행명령 발행노드(interpret)와 출력단
    (시뮬 cmd_bridge / 실차 control_node) 사이에 끼우는 미들웨어.
  * 브리지(시뮬 전용)와 분리 -> 시뮬/실차 양쪽에서 재사용. 실차 전환 시 이 노드는
    그대로 두고 출력 토픽만 remap(/control_safe -> /control)하면 실차 control_node 가
    바로 받는다.
  * 우선순위: 미션 정지 > 주행.
  * use_aruco=false 면 아루코 무시하고 항상 통과(패스스루)만 한다.
  * 감지 노드가 없으면 /aruco_stop 이 안 와서 stop=False 유지 -> 정상 주행(안전 기본).
  * /control 을 받을 때마다 그 값을 (정지/통과 처리해) 재발행하므로 /control_safe 는
    /control 과 같은 주기로 나간다.
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from interface.msg import Control


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


class MissionOverride(Node):
    def __init__(self):
        super().__init__('mission_override')

        self.declare_parameter('control_in_topic', '/control')
        self.declare_parameter('control_out_topic', '/control_safe')
        self.declare_parameter('aruco_stop_topic', '/aruco_stop')
        self.declare_parameter('detected_sign_topic', '/detected_sign')
        # false 면 해당 미션 무시(런치 토글용).
        self.declare_parameter('use_aruco', True)
        self.declare_parameter('use_sign', True)     # 신호등 정지(red/green)
        self.declare_parameter('use_arrow', True)    # 화살표 방향 offset
        # --- 화살표 방향(갈림길) 파라미터 (실시간 튜닝) ---
        # arrow_offset: steering 에 더할 치우침 크기. 작게 시작(급조향 금지, 살짝이면 됨).
        self.declare_parameter('arrow_offset', 0.15)
        # arrow_timeout: 이 시간(초) 신호 안 오면 화살표 사라진 것 -> 즉시 차선주행 복귀.
        # 짧을수록 표지판 지나자마자 바로 순수 차선주행(interpret) 복귀. 너무 짧으면
        # 검출 한두프레임 끊길 때 offset 깜빡임 -> 0.2 정도(카메라 ~20Hz면 3~4프레임).
        self.declare_parameter('arrow_timeout', 0.2)
        # arrow_steer_sign: 좌/우 방향 부호. 2026-07-08 시뮬 실측으로 -1.0 확정
        # (1.0이면 left_sign이 우회전하는 반대 극성이었음; -1.0에서 left→좌/right→우 검증).
        self.declare_parameter('arrow_steer_sign', -1.0)
        self.declare_parameter('debug_log', False)

        control_in = str(self.get_parameter('control_in_topic').value)
        control_out = str(self.get_parameter('control_out_topic').value)
        aruco_stop_topic = str(self.get_parameter('aruco_stop_topic').value)
        detected_sign_topic = str(self.get_parameter('detected_sign_topic').value)
        self.use_aruco = bool(self.get_parameter('use_aruco').value)
        self.use_sign = bool(self.get_parameter('use_sign').value)
        self.use_arrow = bool(self.get_parameter('use_arrow').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)

        # 아루코 정지 신호(장애물 보이면 True). 노드 없으면 계속 False -> 정상 주행.
        self.aruco_stop = False
        self.prev_aruco = False
        # 신호등 정지 래치(red 보면 True, green 보면 False). 검출 끊겨도 유지.
        self.light_stop = False
        # 화살표 방향: 신호등과 '분리된' 변수. 래치 아님 — 보이는 동안만(타임아웃) 유지.
        self.arrow = None          # None / 'left_sign' / 'right_sign'
        self.arrow_time_ns = 0     # 마지막 화살표 수신 시각(ns) — 타임아웃 판정용

        self.pub = self.create_publisher(Control, control_out, 10)
        self.create_subscription(Control, control_in, self.control_callback, 10)
        self.create_subscription(Bool, aruco_stop_topic, self.aruco_stop_callback, 10)
        self.create_subscription(String, detected_sign_topic, self.detected_sign_callback, 10)

        self.get_logger().info(
            'mission_override started:\n'
            f'  {control_in} (+ {aruco_stop_topic} + {detected_sign_topic}) -> {control_out}\n'
            f'  use_aruco={self.use_aruco} use_sign={self.use_sign} use_arrow={self.use_arrow}\n'
            '  정지(아루코/red)=throttle 0 최우선 / 화살표=steering offset(전진유지)'
        )

    def aruco_stop_callback(self, msg: Bool):
        self.aruco_stop = bool(msg.data)
        # 상태 전이 로그(정지<->출발) — 눈으로 동작 확인.
        if self.aruco_stop != self.prev_aruco:
            if self.aruco_stop:
                self.get_logger().warning('MISSION: obstacle detected (aruco) -> STOP')
            else:
                self.get_logger().info('MISSION: aruco clear -> GO')
            self.prev_aruco = self.aruco_stop

    def detected_sign_callback(self, msg: String):
        """YOLO /detected_sign 반응. 신호등과 화살표를 '분리된' 상태로 처리한다.
        /detected_sign 은 '검출됐을 때만' 오므로:
          - 신호등: red=정지 래치 / green=해제 (검출 끊겨도 마지막 상태 유지).
          - 화살표: 보이는 '동안만' 유지 -> 여기선 마지막 방향+시각만 기록하고,
            사라짐(타임아웃) 판정은 control_callback 이 한다."""
        s = str(msg.data)
        # --- 신호등(래치) ---
        if s == 'red_light' and not self.light_stop:
            self.light_stop = True
            self.get_logger().warning('MISSION: red_light -> STOP (초록불까지 정지)')
        elif s == 'green_light' and self.light_stop:
            self.light_stop = False
            self.get_logger().info('MISSION: green_light -> GO')
        # --- 화살표(보이는 동안만) ---
        elif s in ('left_sign', 'right_sign'):
            if self.arrow != s:
                self.get_logger().info(
                    f'MISSION: {s} -> steer {"LEFT" if s == "left_sign" else "RIGHT"} '
                    '(offset, 전진유지)')
            self.arrow = s
            self.arrow_time_ns = self.get_clock().now().nanoseconds

    def control_callback(self, msg: Control):
        out = Control()
        out.header = msg.header
        steering = float(msg.steering)
        throttle = float(msg.throttle)

        # === 1순위 정지(최우선): 아루코 OR 신호등 red. 화살표보다 우선. ===
        stop = (self.use_aruco and self.aruco_stop) or (self.use_sign and self.light_stop)

        # 화살표 타임아웃: 이 시간 신호 안 오면 사라진 것 -> 방향 해제(차선주행 복귀).
        if self.arrow is not None:
            age = (self.get_clock().now().nanoseconds - self.arrow_time_ns) * 1e-9
            if age > float(self.get_parameter('arrow_timeout').value):
                self.get_logger().info('MISSION: arrow gone -> lane follow')
                self.arrow = None

        if stop:
            # 정지: throttle 0, steering 은 유지(바퀴 정렬 보존). 화살표 무시.
            throttle = 0.0
        elif self.use_arrow and self.arrow is not None:
            # === 2순위 방향: 정지 아닐 때만. steering 에 offset(throttle 유지 = 전진). ===
            off = (float(self.get_parameter('arrow_offset').value)
                   * float(self.get_parameter('arrow_steer_sign').value))
            if self.arrow == 'left_sign':
                steering = clip(steering - off, -1.0, 1.0)   # 왼쪽(부호 시뮬 검증)
            elif self.arrow == 'right_sign':
                steering = clip(steering + off, -1.0, 1.0)   # 오른쪽
        # === 3순위 평소: 아무것도 안 함 -> /control 그대로 통과 ===

        out.steering = steering
        out.throttle = throttle
        self.pub.publish(out)

        if self.debug_log:
            self.get_logger().info(
                f'stop={stop}(aruco={self.aruco_stop},light={self.light_stop}) '
                f'arrow={self.arrow} steer={out.steering:+.3f} '
                f'thr_in={msg.throttle:.3f} -> thr_out={out.throttle:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MissionOverride()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
