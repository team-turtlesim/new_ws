#!/usr/bin/env python3
"""디버그 뷰어 노드 — 실차 파이프라인 결과를 cv2 창으로 띄운다(색 영상 + 차선 오버레이).

실차 팀 노드는 화면을 cv2.imshow 로 띄우지 않고 CompressedImage 토픽으로 발행한다.
게다가 lane_node 의 debug 이미지는 '이진 마스크(흑백)' 위에 그려져 배경이 흑백이다
(lane_node 는 효율상 색 영상을 안 받고 마스크만 받기 때문. 실차 코드는 못 건드림).

그래서 이 뷰어는 예전 camera_viewer 처럼 '실제 색 카메라 영상 위에 차선 인지선'을
보여주기 위해, 두 토픽을 합성한다:
  1) 색 영상   : /camera/image/compressed  (image_bridge 가 발행, 컬러)
  2) 차선 debug: /lane_detection/image/debug (마스크 배경 + 색 주석: ROI/점/곡선)
  -> debug 에서 '색 주석 픽셀'(회색이 아닌 픽셀)만 뽑아 색 영상 위에 얹는다.
     결과: 실제 컬러 카메라 화면 + 실제 검출 차선점/곡선/ROI 오버레이.
추가로 /lane/detection 값으로 차선 중심선(초록)·이미지 중심선(회색)·수치 텍스트를 그린다.

창:
  - lane overlay : 색 영상 + 차선 인지선 (기본, 제일 유용)
  - edge         : /opencv/image/edge  흰/노랑 색마스크 (show_edge=true 일 때)

q 키로 종료.

실행: ros2 run sim_bridge debug_viewer
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from interface.msg import LaneDetection


class DebugViewer(Node):
    def __init__(self):
        super().__init__('debug_viewer')

        self.declare_parameter('color_topic', '/camera/image/compressed')
        self.declare_parameter('debug_topic', '/lane_detection/image/debug')
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('edge_topic', '/opencv/image/edge')
        self.declare_parameter('show_edge', True)
        # 색 주석으로 인정할 채널 편차 임계값(회색=0, 순색=큼). 낮추면 더 민감.
        self.declare_parameter('annot_sat_min', 40)
        # 최종 창 확대 배율(320x160 은 작아서 키워 보여준다).
        self.declare_parameter('display_scale', 3)

        self.color_topic = str(self.get_parameter('color_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        edge_topic = str(self.get_parameter('edge_topic').value)
        self.show_edge = bool(self.get_parameter('show_edge').value)
        self.annot_sat_min = int(self.get_parameter('annot_sat_min').value)
        self.display_scale = max(1, int(self.get_parameter('display_scale').value))

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.latest_color = None       # 최신 색 프레임(BGR)
        self.latest_detection = None   # 최신 LaneDetection

        self.create_subscription(CompressedImage, self.color_topic, self.on_color, image_qos)
        self.create_subscription(CompressedImage, debug_topic, self.on_debug, image_qos)
        self.create_subscription(LaneDetection, detection_topic, self.on_detection, 10)
        if self.show_edge:
            self.create_subscription(CompressedImage, edge_topic, self.on_edge, image_qos)

        self.get_logger().info(
            'debug_viewer 시작(색 영상+차선 오버레이). '
            f'color={self.color_topic}, debug={debug_topic} — q 로 종료'
        )

    @staticmethod
    def _decode(msg):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def on_color(self, msg):
        img = self._decode(msg)
        if img is not None:
            self.latest_color = img

    def on_detection(self, msg):
        self.latest_detection = msg

    def on_debug(self, msg):
        """debug(마스크+색주석)를 받으면 색 영상과 합성해 오버레이 창을 갱신."""
        dbg = self._decode(msg)
        if dbg is None:
            return
        h, w = dbg.shape[:2]

        # 배경: 색 영상(있으면) 을 debug 해상도로 맞춤. 없으면 debug 그대로.
        if self.latest_color is not None:
            base = cv2.resize(self.latest_color, (w, h), interpolation=cv2.INTER_AREA)
        else:
            base = dbg.copy()

        # debug 에서 '색 주석'(회색이 아닌 픽셀)만 골라 색 영상 위에 얹는다.
        # 회색(마스크 흑/백/그레이)은 R=G=B 라 채널 편차가 작다 -> 제외.
        b, g, r = dbg[:, :, 0].astype(np.int16), dbg[:, :, 1].astype(np.int16), dbg[:, :, 2].astype(np.int16)
        sat = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
        annot = sat > self.annot_sat_min
        out = base.copy()
        out[annot] = dbg[annot]

        # /lane/detection 값으로 중심선/텍스트 추가(있을 때).
        det = self.latest_detection
        if det is not None and det.image_width > 0:
            sx = w / float(det.image_width)   # debug 폭 대비 스케일
            cx = int(det.center_x * sx)
            cv2.line(out, (cx, 0), (cx, h), (160, 160, 160), 1)  # 이미지 중심(회색)
            if det.lane_center_px >= 0.0:
                lx = int(det.lane_center_px * sx)
                cv2.line(out, (lx, 0), (lx, h), (0, 255, 0), 2)  # 차선 중심(초록)

        # 확대 후 텍스트.
        if self.display_scale > 1:
            out = cv2.resize(
                out, (w * self.display_scale, h * self.display_scale),
                interpolation=cv2.INTER_NEAREST,
            )
        if det is not None:
            txt = (f"off={det.raw_offset:+.2f} head={det.raw_heading:+.2f} "
                   f"conf={det.confidence:.2f} L={int(det.left_detected)} R={int(det.right_detected)}")
            cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('lane overlay', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def on_edge(self, msg):
        img = self._decode(msg)
        if img is not None:
            cv2.imshow('edge (color mask)', img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DebugViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
