"""YOLO 객체검출 노드 (onnxruntime CPU).

카메라 압축영상(/camera/image/compressed, 원본 640x480)을 구독해 YOLO ONNX 로 추론하고,
검출 결과(/yolo/detections, DetectionArray)와 디버그 오버레이(/yolo/image/debug,
CompressedImage)를 발행한다. 차선 파이프라인과 독립된 병렬 인지 브랜치라 기존 주행에
영향이 없다(카메라 원본을 직접 소비 — LaneDetection 등에 의존하지 않음).

설계 메모:
  - 이 보드는 aarch64 CPU 전용이라 추론이 무겁다. QoS 를 BEST_EFFORT/depth=1 로 두어
    추론이 밀리면 최신 프레임만 처리하고 나머지는 드롭한다(지연 누적 방지). 추가로
    infer_every_n 으로 처리 프레임을 솎아 CPU 여유를 확보할 수 있다.
  - 모델 파일이 없거나 로드 실패해도 노드는 죽지 않는다: 빈 DetectionArray 를 계속
    발행(소비자가 '검출 없음'과 '노드 정지'를 구분)하고, 오버레이엔 원본만 내보낸다.
    → 커스텀 모델 학습 전에도 파이프라인 전체를 세워 검증할 수 있다.
  - 정지/감속 판단(제어 연동)은 여기서 하지 않는다. interpret 노드가 /yolo/detections
    를 구독해 담당한다(인지/판단 분리 — 기존 lane_detection↔interpret 구조와 동일).
"""

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from interface.msg import Detection, DetectionArray

from .yolo_infer import ORT_IMPORT_ERROR, YoloDetector, load_labels


def _find_up(rel_parts):
    """이 파일의 상위 경로들을 훑어 rel_parts(예: src/yolo/models/best.onnx)를 찾는다.
    소스 트리 기준 경로를 안정적으로 얻기 위한 헬퍼(다른 노드들의 config 탐색과 동일 패턴)."""
    for base in Path(__file__).resolve().parents:
        candidate = base.joinpath(*rel_parts)
        if candidate.exists():
            return str(candidate)
    return ''


def get_default_model_path():
    # 소스 트리의 모델을 우선(‑‑symlink‑install 로 소스에 떨군 모델이 바로 잡힘).
    found = _find_up(['src', 'yolo', 'models', 'best.onnx'])
    if found:
        return found
    # 아직 best.onnx 가 없으면 src/yolo/models 디렉터리를 가리켜, "not found" 안내가
    # 실제로 파일을 둬야 할 위치(README 와 동일)를 알려주도록 한다.
    models_dir = _find_up(['src', 'yolo', 'models'])
    if models_dir:
        return str(Path(models_dir) / 'best.onnx')
    return str(Path(__file__).resolve().parents[1] / 'models' / 'best.onnx')


def get_default_labels_path():
    found = _find_up(['src', 'yolo', 'models', 'labels.txt'])
    return found or str(Path(__file__).resolve().parents[1] / 'models' / 'labels.txt')


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # --- Topics ---------------------------------------------------------
        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('debug_topic', '/yolo/image/debug')

        # --- 모델/라벨 ------------------------------------------------------
        self.declare_parameter('model_path', get_default_model_path())
        self.declare_parameter('labels_path', get_default_labels_path())

        # --- 추론 하이퍼파라미터 -------------------------------------------
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)     # 모델이 정적크기면 자동으로 덮어씀
        self.declare_parameter('num_threads', 0)      # onnxruntime intra-op(0=자동, 코어수)
        # N 프레임마다 1번만 추론(나머지 프레임은 스킵). CPU 여유 확보용. 1=매 프레임.
        self.declare_parameter('infer_every_n', 1)

        # 전원 게이트: active=False 면 추론·발행을 통째로 건너뛴다(CPU 확보). 모델은
        # 로드된 채라 재개 시 재로딩 없음. /yolo/active(Bool) 토픽으로 런타임 토글
        # (interpret 가 초록불 출발 시 off, ArUco 마커 인지 시 on 을 발행).
        self.declare_parameter('active', True)
        self.declare_parameter('active_topic', '/yolo/active')

        # --- 디버그 오버레이 ------------------------------------------------
        self.declare_parameter('debug_image', True)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('debug_log', False)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.labels_path = os.path.expanduser(str(self.get_parameter('labels_path').value))
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.input_size = int(self.get_parameter('input_size').value)
        self.num_threads = int(self.get_parameter('num_threads').value)
        self.infer_every_n = max(1, int(self.get_parameter('infer_every_n').value))
        self.active = bool(self.get_parameter('active').value)
        self.debug_image = bool(self.get_parameter('debug_image').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)

        # --- 라벨 로드 ------------------------------------------------------
        self.labels = []
        if os.path.exists(self.labels_path):
            try:
                self.labels = load_labels(self.labels_path)
            except Exception as exc:
                self.get_logger().warning(f'Failed to read labels {self.labels_path}: {exc}')

        # --- 검출기 로드(실패해도 노드는 계속: 빈 검출 발행) --------------
        self.detector = self._try_load_detector()

        # --- QoS ---
        # 카메라 구독: 최신 프레임만(추론 밀리면 드롭). BEST_EFFORT sub 는 카메라의
        # RELIABLE pub 과 호환된다(요청<=제공).
        infer_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # 디버그 오버레이 발행: monitor 가 기본 RELIABLE 로 구독하므로(다른 디버그 영상과
        # 동일) 여기도 RELIABLE 이어야 한다. BEST_EFFORT 면 QoS 비호환으로 monitor 가
        # 프레임을 못 받아 YOLO 패널이 빈 채로 남는다.
        overlay_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.detections_pub = self.create_publisher(
            DetectionArray, self.detections_topic, 10
        )
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(
                CompressedImage, self.debug_topic, overlay_qos
            )

        self.subscription = self.create_subscription(
            CompressedImage, subscribe_topic, self.image_callback, infer_qos
        )

        # 전원 게이트 구독. latched(transient_local) 라 나중에 떠도 마지막 명령을 받는다.
        active_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.active_sub = self.create_subscription(
            Bool, str(self.get_parameter('active_topic').value),
            self.on_active, active_qos
        )

        self._frame_count = 0
        self._infer_ms_ema = None  # 추론시간 EMA(로그용)

        self.get_logger().info(
            'yolo node started (onnxruntime CPU):\n'
            f'  subscribe_topic={subscribe_topic}\n'
            f'  detections_topic={self.detections_topic}\n'
            f'  debug_topic={self.debug_topic if self.debug_image else "(disabled)"}\n'
            f'  model={"LOADED " + self.model_path if self.detector else "MISSING (empty detections)"}\n'
            f'  labels={len(self.labels)} classes\n'
            f'  conf={self.conf_threshold} iou={self.iou_threshold} '
            f'input_size={self.detector.input_size if self.detector else self.input_size} '
            f'infer_every_n={self.infer_every_n}'
        )

    def _try_load_detector(self):
        if ORT_IMPORT_ERROR is not None:
            self.get_logger().error(
                'onnxruntime not installed -> empty detections only. '
                'Run: pip install onnxruntime==1.18.1'
            )
            return None
        if not os.path.exists(self.model_path):
            self.get_logger().warning(
                f'Model file not found: {self.model_path} -> publishing empty detections. '
                'Drop a trained best.onnx there (see models/README.md).'
            )
            return None
        try:
            det = YoloDetector(
                self.model_path,
                labels=self.labels,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                input_size=self.input_size,
                num_threads=self.num_threads,
            )
            self.get_logger().info(f'YOLO model loaded: {self.model_path}')
            return det
        except Exception as exc:
            self.get_logger().error(
                f'Failed to load YOLO model {self.model_path}: {exc} -> empty detections.'
            )
            return None

    # ------------------------------------------------------------------ callbk
    def on_active(self, msg: Bool):
        """전원 게이트 토글(/yolo/active). 상태가 바뀔 때만 로그."""
        new_active = bool(msg.data)
        if new_active != self.active:
            self.get_logger().info(
                'YOLO 추론 %s' % ('재개(on)' if new_active else '일시정지(off, CPU 확보)'))
        self.active = new_active

    def image_callback(self, msg: CompressedImage):
        # 전원 게이트: 꺼져 있으면 디코드·추론·발행 전부 건너뛴다(CPU 확보).
        if not self.active:
            return
        self._frame_count += 1
        # 프레임 솎기: 스킵하는 프레임은 추론/발행을 건너뛴다(CPU 여유).
        if (self._frame_count % self.infer_every_n) != 0:
            return

        bgr = self._decode(msg)
        if bgr is None:
            return
        h, w = bgr.shape[:2]

        detections = []
        if self.detector is not None:
            try:
                t0 = self.get_clock().now()
                detections = self.detector.infer(bgr)
                dt_ms = (self.get_clock().now() - t0).nanoseconds * 1e-6
                self._infer_ms_ema = dt_ms if self._infer_ms_ema is None else (
                    0.2 * dt_ms + 0.8 * self._infer_ms_ema
                )
            except Exception as exc:
                self.get_logger().error(f'Inference failed: {exc}', throttle_duration_sec=2.0)
                detections = []

        self._publish_detections(msg, detections, w, h)

        if self.debug_pub is not None:
            self._publish_overlay(msg, bgr, detections)

        if self.debug_log:
            fps = (1000.0 / self._infer_ms_ema) if self._infer_ms_ema else 0.0
            self.get_logger().info(
                f'dets={len(detections)} infer={self._infer_ms_ema or 0:.0f}ms (~{fps:.1f}fps)',
                throttle_duration_sec=1.0,
            )

    def _decode(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warning('Failed to decode camera image')
        return bgr

    def _publish_detections(self, src: CompressedImage, detections, width, height):
        out = DetectionArray()
        out.header.stamp = src.header.stamp
        out.header.frame_id = 'yolo'
        out.image_width = int(width)
        out.image_height = int(height)
        for d in detections:
            det = Detection()
            det.label = str(d['label'])
            det.class_id = int(d['class_id'])
            det.confidence = float(d['confidence'])
            det.x = float(d['x'])
            det.y = float(d['y'])
            det.width = float(d['width'])
            det.height = float(d['height'])
            out.detections.append(det)
        self.detections_pub.publish(out)

    def _publish_overlay(self, src: CompressedImage, bgr, detections):
        canvas = bgr  # 원본 위에 직접 그린다(별도 복사 불필요 — 이후 재사용 안 함)
        for d in detections:
            x1, y1 = int(d['x']), int(d['y'])
            x2, y2 = int(d['x'] + d['width']), int(d['y'] + d['height'])
            cid = int(d['class_id'])
            color = (int((cid * 47) % 256), int((cid * 97) % 256), int((cid * 151) % 256))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytxt = max(0, y1 - 4)
            cv2.rectangle(canvas, (x1, ytxt - th - 4), (x1 + tw + 2, ytxt), color, -1)
            cv2.putText(canvas, text, (x1 + 1, ytxt - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(
            '.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = src.header.stamp
        out.header.frame_id = 'yolo_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
