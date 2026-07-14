import ctypes
import fcntl
import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import yaml


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # ROS parameters
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('publish_topic', 'camera/image/compressed')
        # 캡처/발행 주기(Hz). GStreamer 캡처 framerate 와 발행 타이머를 함께 결정한다.
        # CPU 가 빠듯하면 param 으로 낮춰(예: 20) YOLO 추론에 여유를 줄 수 있다.
        self.declare_parameter('publish_hz', 30.0)
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('usb_camera_device', '/dev/video1')
        self.declare_parameter('mipi_camera_device', '/dev/video0')
        self.declare_parameter('flip_method', 'rotate-180')
        self.declare_parameter('jpeg_quality', 90)
        # 노출/게인 고정 (저조도 fps 방어 + 검출 일관성, USB 카메라 전용).
        # USB 웹캠 오토 익스포저는 어두우면 노출시간을 늘려 프레임레이트를 반토막 낸다
        # (2026-07-14 대회장 실측: 30->15fps. 원인=UVC "Exposure, Auto Priority"=1).
        # 수동 노출로 고정하면 빛이 부족해도 fps 유지 + 조명 변화에 검출이 일관됨.
        # 적용은 캡처 오픈 뒤 apply_v4l2_controls() 에서 ioctl 로 직접 세팅한다(가장 확실 —
        # GStreamer extra-controls 로는 exposure_absolute 가 156 으로 리셋돼 안 먹혔음).
        #   exposure_absolute: 노출시간(단위 100us). fps 상한=1/(exp*100us), 30fps→<=333.
        #     (실측: 이 C920e 는 스트리밍 중 156 근처로 되돌아가는 경향 → 밝기는 주로 gain.)
        #   camera_gain: 0~255, 신호 증폭(밝기↑, 노이즈↑, fps 무관). 밝기 조절의 주 레버.
        # 2026-07-14 대회장 확정: exposure 156 + gain 20 (30fps 유지 + 어두운 구간 라인 확보).
        # (조명 따라 gain 재조정 필요할 수 있음 — python3 ~/cam.py 156 <gain> 로 라이브 튜닝.
        #  트랙에 밝은/어두운 구간이 섞여 있으면 단일 gain 으로 양쪽 만족 어려움 — 어두운
        #  구간 기준으로 맞추거나(라인 놓침이 더 위험) 오토익스포저+AutoPriority=0 검토.)
        # <=0 이면 수동제어 미적용(오토 익스포저=기존 동작). launch 인자 exposure/gain 로 튜닝.
        self.declare_parameter('exposure_absolute', 156)
        self.declare_parameter('camera_gain', 20)
        self.declare_parameter('debug_log', True)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        publish_topic = str(self.get_parameter('publish_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')
        default_camera_device = str(self.get_parameter('camera_device').value)
        usb_camera_device = str(self.get_parameter('usb_camera_device').value)
        mipi_camera_device = str(self.get_parameter('mipi_camera_device').value)
        flip_method = str(self.get_parameter('flip_method').value)
        jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.publish_hz = publish_hz
        # GStreamer caps 의 framerate 는 정수 fps 여야 하므로 publish_hz 를 반올림해 사용.
        self.capture_fps = max(1, int(round(publish_hz)))
        self.jpeg_quality = jpeg_quality
        self.exposure_absolute = int(self.get_parameter('exposure_absolute').value)
        self.camera_gain = int(self.get_parameter('camera_gain').value)

        self.image_width, self.image_height = self.load_image_size()
        self.usb_cam_enabled, self.mipi_cam_enabled = self.load_camera_source_flags()
        usb_camera_device, mipi_camera_device = self.load_camera_device_overrides(
            usb_camera_device,
            mipi_camera_device,
        )
        if self.usb_cam_enabled:
            self.camera_source = 'usb'
            camera_device = usb_camera_device or default_camera_device
        else:
            self.camera_source = 'mipi'
            camera_device = mipi_camera_device or default_camera_device

        self.camera_device = camera_device
        self.flip_method = flip_method

        # QoS compatible with web_video_server and monitor subscribers.
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher_ = self.create_publisher(CompressedImage, publish_topic, self.image_qos)
        self.cap = None
        self.pipeline = None
        if not self.open_capture():
            raise RuntimeError(
                'Failed to open camera with GStreamer pipeline '
                f'(source={self.camera_source}, device={camera_device}, '
                f'width={self.image_width}, height={self.image_height})'
            )

        # 캡처 오픈 직후 노출/게인을 수동 고정(ioctl). USB + exposure_absolute>0 일 때만.
        self.apply_v4l2_controls()

        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_callback)
        self.get_logger().info('\n'
            f'[Camera Node] : topic={publish_topic} \n'
            f'[camera source] : {self.camera_source} \n'
            f'[width] : {self.image_width}, [height] : {self.image_height} \n'
            f'[capture_fps] : {self.capture_fps} \n'
            f'[exposure/gain] : '
            f'{self.exposure_absolute if self.exposure_absolute > 0 else "auto"}'
            f'/{self.camera_gain} \n'
            f'[camera_device] : {camera_device} \n'
            f'[flip_method] : {flip_method} \n'
            f'[jpeg_quality] : {self.jpeg_quality} \n'
            f'[vehicle_config_file] : {self.vehicle_config_file} \n'
            f'[debug_log] : {self.debug_log} \n'
        )

    def load_image_size(self):
        default_size = (640, 480)
        if not os.path.exists(self.vehicle_config_file):
            return default_size

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_size

        image_width = int(config_data.get('IMAGE_WIDTH', default_size[0]))
        image_height = int(config_data.get('IMAGE_HEIGHT', default_size[1]))
        return image_width, image_height

    def load_camera_source_flags(self):
        # Backward-compatible default: MIPI enabled.
        default_usb_cam = False
        default_mipi_cam = True

        if not os.path.exists(self.vehicle_config_file):
            return default_usb_cam, default_mipi_cam

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_cam, default_mipi_cam

        usb_cam = bool(config_data.get('USB_CAM', default_usb_cam))
        mipi_cam = bool(config_data.get('MIPI_CAM', default_mipi_cam))

        if usb_cam and mipi_cam:
            raise ValueError('Only one of USB_CAM or MIPI_CAM can be true.')
        if not usb_cam and not mipi_cam:
            raise ValueError('One of USB_CAM or MIPI_CAM must be true.')

        return usb_cam, mipi_cam

    def build_candidate_pipelines(self, camera_device, flip_method):
        if self.usb_cam_enabled:
            # 노출/게인은 파이프라인이 아니라 apply_v4l2_controls() 에서 ioctl 로 세팅한다
            # (GStreamer extra-controls 로는 exposure_absolute 가 156 으로 리셋돼 안 먹혔음).
            # Many USB webcams expose MJPG by default.
            mjpg_pipeline = (
                f"v4l2src device={camera_device} io-mode=2 ! "
                f"image/jpeg,framerate={self.capture_fps}/1 ! jpegdec ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={self.image_width},height={self.image_height},framerate={self.capture_fps}/1 ! "
                "appsink sync=false drop=true max-buffers=1"
            )
            # Fallback for raw USB camera modes.
            raw_pipeline = (
                f"v4l2src device={camera_device} io-mode=2 ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={self.image_width},height={self.image_height},framerate={self.capture_fps}/1 ! "
                "appsink sync=false drop=true max-buffers=1"
            )
            return [mjpg_pipeline, raw_pipeline]

        mipi_pipeline = (
            f"v4l2src device={camera_device} io-mode=2 ! "
            f"video/x-raw,format=NV12,width={self.image_width},height={self.image_height},framerate={self.capture_fps}/1 ! "
            f"videoconvert ! videoflip method={flip_method} ! "
            "video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
        )
        return [mipi_pipeline]

    def open_capture(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

        for candidate_pipeline in self.build_candidate_pipelines(self.camera_device, self.flip_method):
            cap = cv2.VideoCapture(candidate_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.cap = cap
                self.pipeline = candidate_pipeline
                self.get_logger().info(f'Camera capture opened with pipeline: {candidate_pipeline}')
                return True

            cap.release()
            self.get_logger().warning(f'Failed to open candidate pipeline: {candidate_pipeline}')

        self.cap = None
        self.pipeline = None
        return False

    def apply_v4l2_controls(self):
        """캡처 오픈 뒤 노출/게인을 ioctl 로 직접 세팅(USB 카메라 전용).
        GStreamer extra-controls 로는 exposure_absolute 가 스트리밍 중 156 으로 리셋돼
        안 먹혔다(2026-07-14 실측). ioctl 로 exposure_auto=1(Manual)->exposure_absolute->
        gain 순서로 세팅하면 확실히 적용된다. 실패해도 캡처는 계속(경고만)."""
        if not self.usb_cam_enabled or self.exposure_absolute <= 0:
            return
        vidioc_s_ctrl = 0xc008561c
        ids = {'exposure_auto': 0x009a0901,        # 1 = Manual Mode
               'exposure_absolute': 0x009a0902,
               'gain': 0x00980913}

        class _Ctrl(ctypes.Structure):
            _fields_ = [('id', ctypes.c_uint32), ('value', ctypes.c_int32)]

        try:
            fd = os.open(self.camera_device, os.O_RDWR)
        except OSError as exc:
            self.get_logger().warning(f'노출/게인 세팅용 장치 열기 실패: {exc}')
            return
        try:
            for name, val in (('exposure_auto', 1),
                              ('exposure_absolute', self.exposure_absolute),
                              ('gain', self.camera_gain)):
                c = _Ctrl()
                c.id = ids[name]
                c.value = int(val)
                try:
                    fcntl.ioctl(fd, vidioc_s_ctrl, c)
                except OSError as exc:
                    self.get_logger().warning(f'{name}={val} 세팅 실패: {exc}')
            self.get_logger().info(
                f'카메라 수동 노출 적용(ioctl): exposure_absolute={self.exposure_absolute} '
                f'gain={self.camera_gain} (Manual Mode)')
        finally:
            os.close(fd)

    def load_camera_device_overrides(self, default_usb_camera_device, default_mipi_camera_device):
        if not os.path.exists(self.vehicle_config_file):
            return default_usb_camera_device, default_mipi_camera_device

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_camera_device, default_mipi_camera_device

        usb_camera_device = str(
            config_data.get('USB_CAM_DEVICE', default_usb_camera_device)
        ).strip()
        mipi_camera_device = str(
            config_data.get('MIPI_CAM_DEVICE', default_mipi_camera_device)
        ).strip()
        return usb_camera_device, mipi_camera_device

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warning('Camera capture is not opened')
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning('Failed to read frame')
            return

        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            self.get_logger().warning('Failed to encode frame as JPEG')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.publisher_.publish(msg)
        if self.debug_log:
            self.get_logger().info(f'Published frame: {len(msg.data)} bytes')

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
