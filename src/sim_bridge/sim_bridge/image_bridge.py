"""image_bridge: 가제보 raw 영상 -> 실차 파이프라인용 압축(JPEG) 영상.

가제보 카메라는 sensor_msgs/Image(raw, 보통 rgb8/R8G8B8)를 /camera/image_raw 로
발행하지만, 실차 팀 opencv_node 는 sensor_msgs/CompressedImage(JPEG)를
camera/image/compressed 로 구독한다. 이 노드가 그 사이를 통역한다.

실차 코드는 건드리지 않는다. 토픽 이름/메시지 형식 차이만 여기서 흡수한다.

cv_bridge 없이 numpy + cv2 로 직접 변환한다(의존성 최소화 + 인코딩 방어적 처리).
지원 encoding: rgb8, bgr8, rgba8, bgra8, mono8. (가제보 기본은 rgb8)
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CompressedImage


class ImageBridge(Node):
    def __init__(self):
        # 노드명은 sim_image_bridge: pinky 가제보 launch 가 ros_gz_image 노드를
        # 'image_bridge' 라는 이름으로 이미 띄우므로 이름 충돌을 피한다.
        super().__init__('sim_image_bridge')

        # 입력: 가제보가 발행하는 raw Image 토픽.
        self.declare_parameter('input_topic', '/camera/image_raw')
        # 출력: opencv_node 가 구독하는 압축 토픽.
        # (camera_node 의 기본 publish_topic 과 동일 -> 기본 네임스페이스에서
        #  /camera/image/compressed 로 해석됨)
        self.declare_parameter('output_topic', 'camera/image/compressed')
        self.declare_parameter('jpeg_quality', 90)
        # 입력 QoS reliability: 가제보/ros_gz 브리지가 best_effort 로 낼 수도 있어
        # 기본을 best_effort 로 둔다(reliable/best_effort publisher 모두와 호환).
        self.declare_parameter('input_reliability', 'best_effort')
        self.declare_parameter('debug_log', False)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)
        input_reliability = str(self.get_parameter('input_reliability').value).lower()

        in_rel = (
            ReliabilityPolicy.RELIABLE
            if input_reliability == 'reliable'
            else ReliabilityPolicy.BEST_EFFORT
        )
        in_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=in_rel,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 출력 QoS 는 opencv_node 구독 QoS(RELIABLE/KEEP_LAST 10/VOLATILE)와 일치시킨다.
        out_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(CompressedImage, self.output_topic, out_qos)
        self.sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, in_qos,
        )

        self._warned_encoding = False
        self.get_logger().info(
            'image_bridge started:\n'
            f'  input_topic={self.input_topic} (Image, {input_reliability})\n'
            f'  output_topic={self.output_topic} (CompressedImage jpeg)\n'
            f'  jpeg_quality={self.jpeg_quality}'
        )

    def to_bgr(self, msg: Image):
        """sensor_msgs/Image -> cv2 BGR (imencode 입력용)."""
        enc = (msg.encoding or '').lower()
        h, w = msg.height, msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            ch = 4 if enc.endswith('a8') else 3
            img = buf.reshape(h, w, ch)
            if enc == 'rgb8':
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if enc == 'rgba8':
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            if enc == 'bgra8':
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img  # bgr8
        if enc in ('mono8', '8uc1'):
            gray = buf.reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 알 수 없는 인코딩: 3채널 가정으로 최선의 시도(경고는 1회만).
        if not self._warned_encoding:
            self.get_logger().warning(
                f"Unhandled image encoding '{msg.encoding}', assuming rgb8-like."
            )
            self._warned_encoding = True
        try:
            img = buf.reshape(h, w, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except ValueError:
            return None

    def image_callback(self, msg: Image):
        bgr = self.to_bgr(msg)
        if bgr is None:
            self.get_logger().warning('Failed to convert incoming Image; skipping frame')
            return

        ok, encoded = cv2.imencode(
            '.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning('Failed to JPEG-encode frame')
            return

        out = CompressedImage()
        # 원본 타임스탬프 보존(파이프라인 프레임 정렬용).
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or 'camera'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.pub.publish(out)

        if self.debug_log:
            self.get_logger().info(
                f'compressed {msg.width}x{msg.height} {msg.encoding} '
                f'-> {len(out.data)} bytes'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ImageBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
