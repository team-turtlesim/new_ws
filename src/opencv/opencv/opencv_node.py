import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class OpenCvNode(Node):
    def __init__(self):
        super().__init__('opencv_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_log', True)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)

        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            subscribe_topic,
            self.image_callback,
            image_qos,
        )

        self.gray_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/grayscale',
            image_qos,
        )
        self.blur_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/blur',
            image_qos,
        )
        self.edge_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/edge',
            image_qos,
        )

        self.get_logger().info(
            f'OpenCV node started: subscribe_topic={subscribe_topic}, jpeg_quality={self.jpeg_quality}'
        )

    def to_compressed_msg(self, image, source_msg: CompressedImage, frame_id: str):
        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning(f'Failed to encode image for frame_id={frame_id}')
            return None

        out_msg = CompressedImage()
        out_msg.header.stamp = source_msg.header.stamp
        out_msg.header.frame_id = frame_id
        out_msg.format = 'jpeg'
        out_msg.data = encoded.tobytes()
        return out_msg


    def image_callback(self, msg: CompressedImage):
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        np_arr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        if np_arr is None:
            self.get_logger().warning('Failed to decode compressed image')
            return

        gray = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, 50, 150)

        gray_msg = self.to_compressed_msg(gray, msg, frame_id='opencv_grayscale')
        blur_msg = self.to_compressed_msg(blur, msg, frame_id='opencv_blur')
        edge_msg = self.to_compressed_msg(edge, msg, frame_id='opencv_edge')
        if gray_msg is None or blur_msg is None or edge_msg is None:
            return

        self.gray_pub.publish(gray_msg)
        self.blur_pub.publish(blur_msg)
        self.edge_pub.publish(edge_msg)

        if self.debug_log:
            self.get_logger().info('Published grayscale/blur/edge frames')

    

def main(args=None):
    rclpy.init(args=args)
    node = OpenCvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
