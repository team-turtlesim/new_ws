"""cmd_bridge: 실차 /control -> 가제보 /cmd_vel.

실차에서는 control_node 가 /control(interface/Control)을 받아 D3Racer(pca9685)
서보/모터를 구동한다. 시뮬엔 그 하드웨어가 없으므로, 이 노드가 control_node 를
대체해 /control 을 geometry_msgs/Twist(/cmd_vel)로 바꿔 가제보 로봇을 굴린다.

매핑:
  angular.z = steer_sign * angular_gain * (steering - steer_center)   [-max_angular, max_angular]
  linear.x  = linear_gain * throttle                                  [0, max_linear]

입력 토픽:
  * 기본 구독은 /control_safe (mission_override 를 거친 최종 주행명령).
    미션 정지 로직은 이 노드가 아니라 mission_override 가 담당한다(브리지는 순수
    형식변환만). mission_override 가 없으면 /control 을 직접 구독하도록 파라미터로 바꿀 것.

주의:
  * steer_center: Control.steering 의 중립값. 실차 STEER_TRIM(=0.1)이 섞여 있어
    이를 빼야 시뮬에서 직진이 angular.z=0 이 된다.
  * steer_sign: 실차 서보 극성(-1.0)과 시뮬 /cmd_vel 조향 방향은 다를 수 있다.
    시뮬에서 조향이 반대로 가면 이 값을 뒤집는다(1.0 <-> -1.0). 반드시 검증.
  * 하드웨어(topst_utils/d3racer/pca9685/ina219)는 절대 import 하지 않는다.
  * 워치독: 입력이 끊기면 로봇을 정지시킨다(마지막 명령 유지 방지).
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from interface.msg import Control


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


class CmdBridge(Node):
    def __init__(self):
        super().__init__('sim_cmd_bridge')

        # 기본 입력은 /control_safe (mission_override 출력). 미사용 시 /control 로 바꿔도 됨.
        self.declare_parameter('control_topic', '/control_safe')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # 조향 매핑
        self.declare_parameter('steer_center', 0.10)   # == 실차 STEER_TRIM
        self.declare_parameter('steer_sign', 1.0)      # 반대로 돌면 -1.0
        self.declare_parameter('angular_gain', 2.0)    # rad/s per unit steer
        self.declare_parameter('max_angular', 2.0)     # rad/s clamp

        # 스로틀 매핑
        self.declare_parameter('linear_gain', 1.0)     # m/s per throttle unit
        self.declare_parameter('max_linear', 0.5)      # m/s clamp

        # 워치독 / 발행 주기
        self.declare_parameter('watchdog_sec', 0.5)    # /control 끊기면 정지
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('debug_log', False)

        self.control_topic = str(self.get_parameter('control_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.watchdog_sec = float(self.get_parameter('watchdog_sec').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')
        self.debug_log = bool(self.get_parameter('debug_log').value)

        self.last_twist = Twist()
        self.last_msg_time = None

        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(
            Control, self.control_topic, self.control_callback, 10,
        )
        # 고정 주기 발행 루프(워치독 포함): /control 수신과 무관하게 /cmd_vel 을
        # 꾸준히 내보내 가제보 구동을 매끄럽게, 끊기면 즉시 정지.
        self.timer = self.create_timer(1.0 / publish_hz, self.publish_loop)

        self.get_logger().info(
            'cmd_bridge started:\n'
            f'  control_topic={self.control_topic} -> cmd_vel_topic={self.cmd_vel_topic}\n'
            f'  steer_center={float(self.get_parameter("steer_center").value)} '
            f'steer_sign={float(self.get_parameter("steer_sign").value)} '
            f'angular_gain={float(self.get_parameter("angular_gain").value)} '
            f'max_angular={float(self.get_parameter("max_angular").value)}\n'
            f'  linear_gain={float(self.get_parameter("linear_gain").value)} '
            f'max_linear={float(self.get_parameter("max_linear").value)}\n'
            f'  watchdog_sec={self.watchdog_sec} publish_hz={publish_hz}\n'
            '  NOTE: 시뮬에서 조향 방향이 반대면 steer_sign 을 뒤집으세요.'
        )

    def control_callback(self, msg: Control):
        steer_center = float(self.get_parameter('steer_center').value)
        steer_sign = float(self.get_parameter('steer_sign').value)
        angular_gain = float(self.get_parameter('angular_gain').value)
        max_angular = float(self.get_parameter('max_angular').value)
        linear_gain = float(self.get_parameter('linear_gain').value)
        max_linear = float(self.get_parameter('max_linear').value)

        steer = float(msg.steering) - steer_center
        angular = clip(steer_sign * angular_gain * steer, -max_angular, max_angular)
        linear = clip(linear_gain * float(msg.throttle), 0.0, max_linear)

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.last_twist = twist
        self.last_msg_time = self.get_clock().now()

        if self.debug_log:
            self.get_logger().info(
                f'steer={msg.steering:+.3f} thr={msg.throttle:.3f} '
                f'-> lin.x={linear:.3f} ang.z={angular:+.3f}'
            )

    def publish_loop(self):
        # 워치독: /control 이 watchdog_sec 이상 끊기면 정지 Twist 발행.
        if self.last_msg_time is None:
            self.pub.publish(Twist())
            return
        age = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if age > self.watchdog_sec:
            self.last_twist = Twist()
            self.pub.publish(Twist())
            if self.debug_log:
                self.get_logger().warning(
                    f'/control stale ({age:.2f}s) -> stop', throttle_duration_sec=1.0,
                )
            return
        self.pub.publish(self.last_twist)


def main(args=None):
    rclpy.init(args=args)
    node = CmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
