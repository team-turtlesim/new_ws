import os
from pathlib import Path
import shutil

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from interface.msg import Battery
from interface.msg import Control
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import yaml

from .flask_app_factory import FLASK_IMPORT_ERROR, FlaskServerThread, create_app
from .graph_utils import build_graph_snapshot
from .image_utils import extract_jpeg_dimensions
from .monitor_state import MonitorState

PACKAGE_ROOT = Path(__file__).resolve().parent


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def resolve_resource_path(filename):
    candidates = []

    try:
        share_dir = Path(get_package_share_directory('monitor'))
        candidates.append(share_dir / 'resource' / filename)
    except PackageNotFoundError:
        pass

    candidates.append(PACKAGE_ROOT.parent / 'resource' / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'Unable to find resource file: {filename}')

class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor_node')

        if FLASK_IMPORT_ERROR is not None:
            raise RuntimeError(
                'Flask is not installed. Install "python3-flask" and try again.'
            ) from FLASK_IMPORT_ERROR

        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('battery_topic', 'battery_status')
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('debug_image',True)
        self.declare_parameter('opencv_grayscale_topic', '/opencv/image/grayscale')
        self.declare_parameter('opencv_blur_topic', '/opencv/image/blur')
        self.declare_parameter('opencv_edge_topic', '/opencv/image/edge')
        self.declare_parameter('yolo_debug_topic', '/yolo/image/debug')
        self.declare_parameter('yolo_debug', False)  # YOLO 오버레이 패널 표시 여부
        self.declare_parameter('aruco_debug_topic', '/aruco/image/debug')
        self.declare_parameter('aruco_debug', False)  # ArUco 오버레이 패널 표시 여부
        self.declare_parameter('ramp_debug_topic', '/ramp_detection/image/debug')
        self.declare_parameter('ramp_debug', False)  # 진입로(노란 램프) 오버레이 패널 표시 여부
        self.declare_parameter('bluesign_debug_topic', '/bluesign/image/debug')
        self.declare_parameter('bluesign_debug', False)  # 파란 표지판 트리거 오버레이 패널 표시 여부
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('storage_path', '/')
        self.declare_parameter('storage_poll_interval_sec', 1.0)
        self.declare_parameter('web_host', '192.168.0.12')
        self.declare_parameter('web_port', 5000)
        self.declare_parameter('page_title', 'D-Racer Monitor')
        self.declare_parameter('refresh_interval_ms', 1000)
        self.declare_parameter('image_refresh_interval_ms', 100)
        self.declare_parameter('stale_timeout_sec', 3.0)
        self.declare_parameter('image_source_width', 160)
        self.declare_parameter('image_source_height', 120)
        self.declare_parameter('image_display_width', 160)
        self.declare_parameter('image_display_height', 120)
        self.declare_parameter('debug_log', False)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        yaml_config = self.load_vehicle_config()

        self.battery_topic = self.get_yaml_or_param_str(yaml_config, 'BATTERY_TOPIC', 'battery_topic')
        self.image_topic = self.get_yaml_or_param_str(yaml_config, 'IMAGE_TOPIC', 'image_topic')
        self.control_topic = self.get_yaml_or_param_str(yaml_config, 'CONTROL_TOPIC', 'control_topic')
        self.debug_image = self.get_yaml_or_param_bool_multi(yaml_config, ('OPENCV_DEBUG_MODE', 'DEBUG_IMAGE'), 'debug_image')
        self.opencv_grayscale_topic = self.get_yaml_or_param_str(yaml_config, 'OPENCV_GRAYSCALE_TOPIC', 'opencv_grayscale_topic')
        self.opencv_blur_topic = self.get_yaml_or_param_str(yaml_config, 'OPENCV_BLUR_TOPIC', 'opencv_blur_topic')
        self.opencv_edge_topic = self.get_yaml_or_param_str(yaml_config, 'OPENCV_EDGE_TOPIC', 'opencv_edge_topic')
        self.yolo_debug_topic = self.get_yaml_or_param_str(yaml_config, 'YOLO_DEBUG_TOPIC', 'yolo_debug_topic')
        self.yolo_debug = bool(self.get_parameter('yolo_debug').value)
        self.aruco_debug_topic = self.get_yaml_or_param_str(yaml_config, 'ARUCO_DEBUG_TOPIC', 'aruco_debug_topic')
        self.aruco_debug = bool(self.get_parameter('aruco_debug').value)
        self.ramp_debug_topic = self.get_yaml_or_param_str(yaml_config, 'RAMP_DEBUG_TOPIC', 'ramp_debug_topic')
        self.ramp_debug = bool(self.get_parameter('ramp_debug').value)
        self.bluesign_debug_topic = self.get_yaml_or_param_str(yaml_config, 'BLUESIGN_DEBUG_TOPIC', 'bluesign_debug_topic')
        self.bluesign_debug = bool(self.get_parameter('bluesign_debug').value)
        if not self.control_topic:
            # Backward compatibility for legacy typo key.
            self.control_topic = str(yaml_config.get('CONTORL_TOPIC', '')).strip()
        if not self.control_topic:
            self.control_topic = str(self.get_parameter('control_topic').value)

        requested_storage_path = Path(
            self.get_yaml_or_param_str(yaml_config, 'STORAGE_PATH', 'storage_path')
        ).expanduser()
        self.storage_poll_interval_sec = float(
            self.get_parameter('storage_poll_interval_sec').value
        )
        self.web_host = self.get_yaml_or_param_str(yaml_config, 'WEB_HOST', 'web_host')
        self.web_port = self.get_yaml_or_param_int(yaml_config, 'WEB_PORT', 'web_port')
        self.page_title = str(self.get_parameter('page_title').value)
        self.refresh_interval_ms = int(self.get_parameter('refresh_interval_ms').value)
        self.image_refresh_interval_ms = int(
            self.get_parameter('image_refresh_interval_ms').value
        )
        self.debug_log = bool(self.get_parameter('debug_log').value)
        stale_timeout_sec = float(self.get_parameter('stale_timeout_sec').value)
        image_source_width = int(self.get_parameter('image_source_width').value)
        image_source_height = int(self.get_parameter('image_source_height').value)
        self.image_source_width = image_source_width
        self.image_source_height = image_source_height
        self.image_display_width = self.get_yaml_or_param_int(
            yaml_config, 'IMAGE_DISPLAY_WIDTH', 'image_display_width'
        )
        self.image_display_height = self.get_yaml_or_param_int(
            yaml_config, 'IMAGE_DISPLAY_HEIGHT', 'image_display_height'
        )

        if requested_storage_path.exists():
            self.storage_path = str(requested_storage_path)
        else:
            self.storage_path = '/'
            self.get_logger().warning(
                f'Storage path {requested_storage_path} does not exist. Falling back to /.'
            )

        self.header_logo_path = resolve_resource_path('Telechips-CI-White.png')
        self.telechips_logo_path = resolve_resource_path('Telechips-CI-White.png')
        self.topst_logo_path = resolve_resource_path('TOPST-Logo(White).png')
        self.state = MonitorState(
            stale_timeout_sec,
            image_source_width,
            image_source_height,
        )
        self.app = create_app(
            self.state,
            self.page_title,
            self.battery_topic,
            self.image_topic,
            self.control_topic,
            self.storage_path,
            self.refresh_interval_ms,
            self.image_refresh_interval_ms,
            self.header_logo_path,
            self.telechips_logo_path,
            self.topst_logo_path,
            self.image_display_width,
            self.image_display_height,
            self.debug_image,
            self.opencv_grayscale_topic,
            self.opencv_blur_topic,
            self.opencv_edge_topic,
            yolo_debug=self.yolo_debug,
            yolo_debug_topic=self.yolo_debug_topic,
            aruco_debug=self.aruco_debug,
            aruco_debug_topic=self.aruco_debug_topic,
            ramp_debug=self.ramp_debug,
            ramp_debug_topic=self.ramp_debug_topic,
            bluesign_debug=self.bluesign_debug,
            bluesign_debug_topic=self.bluesign_debug_topic,
            graph_snapshot_provider=self.get_graph_snapshot,
        )
        self.server_thread = FlaskServerThread(self.app, self.web_host, self.web_port)

        self.create_subscription(
            Battery,
            self.battery_topic,
            self.battery_callback,
            10,
        )
        self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10,
        )
        if self.debug_image:
            self.create_subscription(
                CompressedImage,
                self.opencv_grayscale_topic,
                self.debug_grayscale_callback,
                10,
            )
            self.create_subscription(
                CompressedImage,
                self.opencv_blur_topic,
                self.debug_blur_callback,
                10,
            )
            self.create_subscription(
                CompressedImage,
                self.opencv_edge_topic,
                self.debug_edge_callback,
                10,
            )
            if self.yolo_debug:
                self.create_subscription(
                    CompressedImage,
                    self.yolo_debug_topic,
                    self.debug_yolo_callback,
                    10,
                )
            if self.aruco_debug:
                self.create_subscription(
                    CompressedImage,
                    self.aruco_debug_topic,
                    self.debug_aruco_callback,
                    10,
                )
            if self.ramp_debug:
                self.create_subscription(
                    CompressedImage,
                    self.ramp_debug_topic,
                    self.debug_ramp_callback,
                    10,
                )
            if self.bluesign_debug:
                self.create_subscription(
                    CompressedImage,
                    self.bluesign_debug_topic,
                    self.debug_bluesign_callback,
                    10,
                )
        self.create_subscription(
            Control,
            self.control_topic,
            self.control_callback,
            10,
        )
        self.storage_timer = self.create_timer(
            self.storage_poll_interval_sec,
            self.storage_timer_callback,
        )
        self.storage_timer_callback()

        self.server_thread.start()

        display_host = '127.0.0.1' if self.web_host == '0.0.0.0' else self.web_host
        self.get_logger().info(
            f'[Monitor node started] \n'
            f'battery_topic={self.battery_topic} \n'
            f'image_topic={self.image_topic} \n'
            f'debug_image={self.debug_image}, \n'
            f'control_topic={self.control_topic}, \n'
            f'storage_path={self.storage_path}, \n'
            f'web=http://{display_host}:{self.web_port} \n'
            f'vehicle_config_file={self.vehicle_config_file} \n'
        )

    def get_graph_snapshot(self):
        return build_graph_snapshot(self)

    def load_vehicle_config(self):
        if not os.path.exists(self.vehicle_config_file):
            return {}

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                return yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return {}

    def get_yaml_or_param_str(self, yaml_config, yaml_key, param_key):
        raw_value = yaml_config.get(yaml_key)
        if raw_value is not None:
            text_value = str(raw_value).strip()
            if text_value:
                return text_value
        return str(self.get_parameter(param_key).value)

    def get_yaml_or_param_int(self, yaml_config, yaml_key, param_key):
        raw_value = yaml_config.get(yaml_key)
        if raw_value is not None:
            text_value = str(raw_value).strip()
            if text_value:
                return int(raw_value)
        return int(self.get_parameter(param_key).value)

    def get_yaml_or_param_bool(self, yaml_config, yaml_key, param_key):
        raw_value = yaml_config.get(yaml_key)
        if raw_value is not None:
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, str):
                return raw_value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(raw_value)
        return bool(self.get_parameter(param_key).value)

    def get_yaml_or_param_bool_multi(self, yaml_config, yaml_keys, param_key):
        for yaml_key in yaml_keys:
            if yaml_key in yaml_config:
                return self.get_yaml_or_param_bool(yaml_config, yaml_key, param_key)
        return bool(self.get_parameter(param_key).value)

    def battery_callback(self, msg):
        self.state.update_battery(msg.battery_status)

        if self.debug_log:
            self.get_logger().info(f'Battery status updated: {msg.battery_status:.1f}%')

    def image_callback(self, msg):
        try:
            frame_bytes = bytes(msg.data)
            width, height = extract_jpeg_dimensions(frame_bytes)
            if width is None or height is None:
                width = self.image_source_width
                height = self.image_source_height

            self.state.update_image(frame_bytes, width, height)
        except Exception as exc:
            self.get_logger().error(f'Failed to process {self.image_topic} frame: {exc}')

    def control_callback(self, msg):
        self.state.update_control(msg.throttle, msg.steering)

        if self.debug_log:
            self.get_logger().info(
                f'Control updated: throttle={msg.throttle:.2f}, steering={msg.steering:.2f}'
            )

    def _debug_image_callback(self, msg, image_key, topic):
        try:
            frame_bytes = bytes(msg.data)
            width, height = extract_jpeg_dimensions(frame_bytes)
            if width is None or height is None:
                width = self.image_source_width
                height = self.image_source_height

            self.state.update_debug_image(image_key, frame_bytes, width, height)
        except Exception as exc:
            self.get_logger().error(f'Failed to process {topic} frame: {exc}')

    def debug_grayscale_callback(self, msg):
        self._debug_image_callback(msg, 'grayscale', self.opencv_grayscale_topic)

    def debug_blur_callback(self, msg):
        self._debug_image_callback(msg, 'blur', self.opencv_blur_topic)

    def debug_edge_callback(self, msg):
        self._debug_image_callback(msg, 'edge', self.opencv_edge_topic)

    def debug_yolo_callback(self, msg):
        self._debug_image_callback(msg, 'yolo', self.yolo_debug_topic)

    def debug_aruco_callback(self, msg):
        self._debug_image_callback(msg, 'aruco', self.aruco_debug_topic)

    def debug_ramp_callback(self, msg):
        self._debug_image_callback(msg, 'ramp', self.ramp_debug_topic)

    def debug_bluesign_callback(self, msg):
        self._debug_image_callback(msg, 'bluesign', self.bluesign_debug_topic)

    def storage_timer_callback(self):
        try:
            usage = shutil.disk_usage(self.storage_path)
            self.state.update_storage(usage.used, usage.total)
        except Exception as exc:
            self.get_logger().error(f'Failed to read storage usage from {self.storage_path}: {exc}')

    def destroy_node(self):
        if hasattr(self, 'server_thread') and self.server_thread is not None:
            self.server_thread.shutdown()
            self.server_thread.join(timeout=2.0)

        super().destroy_node()


def main(args=None):
    node = None
    rclpy.init(args=args)

    try:
        node = MonitorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None:
            node.get_logger().info('Shutting down monitor node.')
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
