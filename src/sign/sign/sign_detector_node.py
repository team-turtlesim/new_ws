#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 표지판/신호등 인지 노드 (sign_detector).

역할(인지만):
  - 카메라 이미지(/camera/image_raw)를 받아 YOLO(dracer_n.onnx)로 추론하고,
  - 검출된 클래스 이름을 /detected_sign(String)으로 발행한다.
  - 제어는 하지 않는다. 신호등 정지/표지판 반응은 mission_override 가 담당.

클래스 순서(모델 내장, 고정):
  0=green_light, 1=left_sign, 2=red_light, 3=right_sign

new_ws 이식판:
  - 모델 경로/토픽을 파라미터화. 모델 기본값은 이 패키지 share/sign/models/dracer_n.onnx.
  - 시뮬(가제보)은 /camera/image_raw(raw Image)를 주므로 그대로 구독한다.
    (실차 car_ws 로 옮길 땐 aruco_detector 처럼 compressed 구독으로 바꿔야 함.)
"""

import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from ultralytics import YOLO
import cv2

from ament_index_python.packages import get_package_share_directory


# 클래스 번호 -> 이름 (모델 내장 학습 순서와 일치. 불일치 시 노드가 경고).
CLASS_NAMES = {
    0: "green_light",
    1: "left_sign",
    2: "red_light",
    3: "right_sign",
}


def default_model_path():
    try:
        return os.path.join(
            get_package_share_directory('sign'), 'models', 'dracer_n.onnx'
        )
    except Exception:
        return 'dracer_n.onnx'


class SignDetectorNode(Node):
    def __init__(self):
        super().__init__('sign_detector_node')

        self.declare_parameter('model_path', default_model_path())
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detected_sign_topic', '/detected_sign')
        # 클래스별 신뢰도 커트라인 (신호등은 오검출 위험 커서 높게).
        self.declare_parameter('light_min_confidence', 0.80)
        self.declare_parameter('sign_min_confidence', 0.30)
        # 화살표 near-gate: 표지판 바운딩박스 높이가 화면의 이 비율 이상일 때만
        # (=충분히 가까울 때만) /detected_sign 발행. 멀리서 미리 꺾어 차선 이탈하는 것 방지.
        # 신호등(red/green)은 멀리서 미리 서야 하므로 이 게이트 적용 안 함.
        self.declare_parameter('arrow_near_ratio', 0.25)

        model_path = str(self.get_parameter('model_path').value)
        image_topic = str(self.get_parameter('image_topic').value)
        detected_sign_topic = str(self.get_parameter('detected_sign_topic').value)

        # --- 모델 로드 ---
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO 모델 로드 완료: {model_path}")

        # 클래스 매핑 안전장치: 모델 내장 순서와 CLASS_NAMES 비교(라벨 뒤바뀜 방지).
        self.get_logger().info(f"모델 실제 클래스 순서: {self.model.names}")
        for i, nm in self.model.names.items():
            if CLASS_NAMES.get(i) != nm:
                self.get_logger().warn(
                    f"⚠️ 클래스 매핑 불일치! CLASS_NAMES[{i}]='{CLASS_NAMES.get(i)}' "
                    f"!= 모델[{i}]='{nm}' → CLASS_NAMES를 모델 순서에 맞춰 고치세요.")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(String, detected_sign_topic, 10)

        self.get_logger().info(
            f"🔥 sign_detector 시작. 구독={image_topic} 발행={detected_sign_topic}. 이미지 대기 중...")

    def image_callback(self, msg):
        try:
            if msg.encoding == 'rgb8':
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"이미지 변환 실패(YOLO): {e}")
            return

        results = self.model(frame, verbose=False)

        light_min = float(self.get_parameter('light_min_confidence').value)
        sign_min = float(self.get_parameter('sign_min_confidence').value)
        arrow_near = float(self.get_parameter('arrow_near_ratio').value)
        frame_h = float(frame.shape[0])

        for box in results[0].boxes:
            cls_id = int(box.cls)
            try:
                conf = float(box.conf[0])
            except (TypeError, IndexError):
                conf = float(box.conf)

            name = CLASS_NAMES.get(cls_id, "unknown")
            min_confidence = light_min if name in ("red_light", "green_light") else sign_min
            if conf < min_confidence:
                continue

            # 화살표 near-gate: 박스 높이가 화면의 arrow_near 미만이면 '아직 멂' -> 발행 안 함.
            # (가까워져 박스가 커지면 그때 발행 -> mission_override 가 그때부터 꺾음.)
            if name in ("left_sign", "right_sign"):
                try:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    h_ratio = (y2 - y1) / frame_h
                except Exception:
                    h_ratio = 1.0  # 박스 크기 못구하면 게이트 통과(안전측)
                near = h_ratio >= arrow_near
                self.get_logger().info(
                    f"🎯 [YOLO 검출] {name} (신뢰도 {conf:.2f}) 크기 {h_ratio:.2f} "
                    f"{'NEAR->발행(꺾음)' if near else 'far->대기(차선유지)'}")
                if not near:
                    continue
            else:
                self.get_logger().info(
                    f"🎯 [YOLO 검출] {name} (신뢰도 {conf:.2f}, 기준: {min_confidence:.2f})")

            out_msg = String()
            out_msg.data = name
            self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
