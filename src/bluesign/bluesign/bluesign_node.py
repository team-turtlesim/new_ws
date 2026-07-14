"""파란 표지판 트리거 노드 (BlueSign) — 값싼 캐스케이드 게이트로 YOLO 를 깨운다.

문제: 파란 방향표지판을 YOLO 로 인식해야 하는데, 출발부터 YOLO 를 켜두면 S자 코스에서
FPS 가 떨어져 차선 인지가 흔들린다(YOLO 백본이 CPU 를 크게 먹음). 표지판은 갈림길에서만
필요하다.

해법: opencv_node 가 이미 만들어 발행하는 파란색 전용 마스크(/opencv/image/blue)를 구독해,
'상단 ROI 에 파란 픽셀이 충분한가'만 디바운스로 판정하고 /sign/near(Bool) 로 알린다.
interpret 이 이 신호를 받아 YOLO 전원(/yolo/active)을 켠다. 파란 마스크는 opencv_node 가
흰/노랑과 같은 hsv 로 한 번에 만든 것이라 여기서 추가 디코드·색변환이 없다(마스크만 받아
카운트). 고전 CV 라 ~수 ms 로 가벼워 S자 내내 돌려도 부담이 없다.

발행:
  - /sign/near (Bool)                       : 파란 표지판 근접(디바운스+히스테리시스) 트리거
  - /bluesign/image/debug (CompressedImage) : 대시보드용 오버레이(ROI 박스 + 파란비율 + 상태)

설계 메모: 이 노드는 '파란 게 상단에 보이나'만 말한다(순수 검출). YOLO 를 켤지/끌지 같은
판단은 interpret 이 한다(인지↔판단 분리). 트리거는 '켜기'만 하므로 오검출은 CPU 만 잠깐
낭비할 뿐 위험하지 않다 → 놓치지 않도록 민감하게(낮은 임계) 두는 게 안전하다.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class BlueSignNode(Node):
    def __init__(self):
        super().__init__('bluesign_node')

        # --- Topics ---------------------------------------------------------
        self.declare_parameter('blue_mask_topic', '/opencv/image/blue')
        self.declare_parameter('sign_near_topic', '/sign/near')
        self.declare_parameter('debug_topic', '/bluesign/image/debug')

        # --- 상단 ROI (프레임 대비 비율). 표지판은 앞쪽=화면 위에 나타나므로 상단만 본다.
        #     lane_node 가 하단 ROI 로 차선을 보는 것과 대칭. 값은 라이브 튜닝 가능.
        self.declare_parameter('roi_top_frac', 0.0)     # ROI 위 경계(0=맨 위)
        self.declare_parameter('roi_bottom_frac', 0.5)  # ROI 아래 경계(0.5=상단 절반)
        self.declare_parameter('roi_left_frac', 0.0)
        self.declare_parameter('roi_right_frac', 1.0)

        # --- 트리거 임계 + 디바운스/히스테리시스 --------------------------
        # blue_frac = ROI 내 파란 픽셀 비율. on 임계 이상이 on_frames 연속이면 near=True.
        # off 임계 미만이 off_frames 연속이어야 near=False (깜빡임 방지). off<=on 권장.
        self.declare_parameter('blue_frac_on', 0.02)    # 2% (민감하게 시작 — 놓침 방지)
        self.declare_parameter('blue_frac_off', 0.01)   # 1% (히스테리시스 하한)
        self.declare_parameter('on_frames', 2)          # 등장 확정: N프레임 연속
        self.declare_parameter('off_frames', 8)         # 소멸 확정: M프레임 연속
        self.declare_parameter('mask_threshold', 127)   # JPEG 압축된 마스크 재이진화 임계

        # --- 디버그 오버레이 -----------------------------------------------
        self.declare_parameter('debug_image', True)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('debug_log', False)

        blue_mask_topic = str(self.get_parameter('blue_mask_topic').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.debug_image = bool(self.get_parameter('debug_image').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)

        # --- 디바운스 상태 --------------------------------------------------
        self.on_count = 0
        self.off_count = 0
        self.near_state = False

        # --- QoS: opencv 마스크(RELIABLE)와 호환. 오버레이는 monitor(RELIABLE) 에 맞춤. ---
        mask_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                              reliability=ReliabilityPolicy.RELIABLE)
        overlay_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE)

        self.near_pub = self.create_publisher(
            Bool, str(self.get_parameter('sign_near_topic').value), 10)
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(
                CompressedImage, self.debug_topic, overlay_qos)

        self.subscription = self.create_subscription(
            CompressedImage, blue_mask_topic, self.mask_callback, mask_qos)

        self.get_logger().info(
            'bluesign node started:\n'
            '  blue_mask_topic=%s\n'
            '  sign_near_topic=%s\n'
            '  debug_topic=%s\n'
            '  roi(t,b,l,r)=(%.2f,%.2f,%.2f,%.2f)\n'
            '  blue_frac on/off=%.3f/%.3f  frames on/off=%d/%d' % (
                blue_mask_topic,
                str(self.get_parameter('sign_near_topic').value),
                self.debug_topic if self.debug_image else '(disabled)',
                float(self.get_parameter('roi_top_frac').value),
                float(self.get_parameter('roi_bottom_frac').value),
                float(self.get_parameter('roi_left_frac').value),
                float(self.get_parameter('roi_right_frac').value),
                float(self.get_parameter('blue_frac_on').value),
                float(self.get_parameter('blue_frac_off').value),
                int(self.get_parameter('on_frames').value),
                int(self.get_parameter('off_frames').value),
            )
        )

    # ------------------------------------------------------------------ callbk
    def mask_callback(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self.get_logger().warning('파란 마스크 디코드 실패')
            return

        h, w = mask.shape[:2]
        gp = self.get_parameter
        thr = int(gp('mask_threshold').value)

        # 상단 ROI 잘라내기(비율 -> 픽셀). 안전하게 clamp.
        y0 = max(0, min(h, int(float(gp('roi_top_frac').value) * h)))
        y1 = max(0, min(h, int(float(gp('roi_bottom_frac').value) * h)))
        x0 = max(0, min(w, int(float(gp('roi_left_frac').value) * w)))
        x1 = max(0, min(w, int(float(gp('roi_right_frac').value) * w)))
        if y1 <= y0 or x1 <= x0:
            self.get_logger().warning('ROI 가 비었음(비율 파라미터 확인)', throttle_duration_sec=2.0)
            return

        roi = mask[y0:y1, x0:x1]
        # JPEG 압축으로 값이 번져 있으니 임계로 재이진화 후 파란 픽셀 비율 계산.
        blue_px = int(cv2.countNonZero((roi >= thr).astype(np.uint8)))
        frac = blue_px / float(roi.size)

        near = self.update_near_signal(frac)

        if self.debug_pub is not None:
            self.publish_overlay(mask, (x0, y0, x1, y1), frac, near, msg)
        if self.debug_log:
            self.get_logger().info(
                'blue_frac=%.3f near=%s (on=%d off=%d)' % (
                    frac, near, self.on_count, self.off_count),
                throttle_duration_sec=0.5)

    # ------------------------------------------------------------------ signal
    def update_near_signal(self, frac):
        """파란비율을 디바운스+히스테리시스로 near 상태로 바꾸고 /sign/near 발행(매 프레임).
          - 등장: blue_frac_on 이상이 on_frames 연속 -> near=True
          - 소멸: blue_frac_off 미만이 off_frames 연속 -> near=False
          - 두 임계 사이(밴드): 애매 -> 카운터 리셋하고 현 상태 유지(깜빡임 억제)"""
        gp = self.get_parameter
        frac_on = float(gp('blue_frac_on').value)
        frac_off = float(gp('blue_frac_off').value)
        on_frames = int(gp('on_frames').value)
        off_frames = int(gp('off_frames').value)

        if frac >= frac_on:
            self.on_count += 1
            self.off_count = 0
        elif frac < frac_off:
            self.off_count += 1
            self.on_count = 0
        else:
            self.on_count = 0
            self.off_count = 0

        if self.on_count >= on_frames:
            self.near_state = True
        elif self.off_count >= off_frames:
            self.near_state = False

        m = Bool()
        m.data = bool(self.near_state)
        self.near_pub.publish(m)
        return self.near_state

    # ------------------------------------------------------------------ io
    def publish_overlay(self, mask, roi_box, frac, near, src: CompressedImage):
        if self.debug_pub is None:
            return
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        x0, y0, x1, y1 = roi_box
        # ROI 박스: near 면 초록, 아니면 회색
        color = (0, 220, 0) if near else (160, 160, 160)
        cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        status = 'SIGN NEAR -> YOLO ON' if near else 'no sign'
        cv2.putText(display, '%s  blue=%.1f%%' % (status, frac * 100.0),
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        ok, enc = cv2.imencode(
            '.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = src.header.stamp
        out.header.frame_id = 'bluesign_debug'
        out.format = 'jpeg'
        out.data = enc.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = BlueSignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
