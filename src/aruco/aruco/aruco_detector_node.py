#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        # 가제보 카메라 이미지 구독 (/camera/image_raw 확인 완료)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            Int32,
            '/detected_marker_id',
            10
        )

        # ── 동적 장애물(아루코) 정지 신호: 지정 ID 마커 보이면 True(정지), 안 보이면 False(출발) ──
        self.stop_pub = self.create_publisher(Bool, '/aruco_stop', 10)
        self.declare_parameter('target_marker_id', 3)   # 이 ID만 장애물로 반응 (다른 ID 무시)
        self.declare_parameter('stop_on_frames', 1)     # 등장: N프레임 연속 보이면 정지 (빠르게=안전)
        self.declare_parameter('go_after_frames', 5)    # 소멸: N프레임 연속 안 보이면 출발 (떨림 방지)
        self.seen_count = 0
        self.notseen_count = 0
        self.stop_state = False

        # 기본 탐색용 규격 (유저님이 확인해주신 기존 세팅)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 💡 [교차 검증 필터 추가] 가짜/오류 마커의 진짜 정체를 알아내기 위한 백업 딕셔너리 목록
        self.alt_dicts = {
            "6X6_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
            "5X5_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
            "5X5_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            "4X4_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        }
        
        self.get_logger().info("🔥 [블랙박스 판독 시스템 가동] 노드가 시작되었습니다. 진짜 마커 규격을 강제 추적합니다...")

    def image_callback(self, msg):
        self.get_logger().info(f"📸 가제보 영상 수신 중... (인코딩 형식: {msg.encoding})", throttle_duration_sec=1.0)
        
        try:
            # 원본 인코딩 안전 변환 로직 (rgb8 -> BGR 행렬 형식화)
            if 'rgb' in msg.encoding.lower():
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"❌ 이미지 변환 실패: {e}")
            return
        
        # 원본 전처리 루틴
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # 🎨 시각화용 컬러 프레임 (여기에 네모/ID를 그림)
        display = frame.copy()

        # 1차 시도 (기본 6X6_50 검출)
        corners, ids, rejected = self.detector.detectMarkers(equalized)

        # ── 동적 장애물 정지 신호 (지정 ID 디바운스 → /aruco_stop 매 프레임 발행) ──
        self.update_stop_signal(ids)

        # 만약 기본 규격으로 인지가 성공했다면 즉시 결과 발행 후 리턴
        if ids is not None:
            # ✅ 인식된 마커 둘레에 초록 네모 + ID 그리기
            cv2.aruco.drawDetectedMarkers(display, corners, ids, borderColor=(0, 255, 0))
            marker_id = int(ids[0][0])
            self.get_logger().info(f"🎯 [대성공] 기본 규격(6X6_50)에서 마커 발견: {marker_id}")
            self.publish_id(marker_id)
            cv2.putText(display, f"DETECTED  ID = {marker_id}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            self.show(display)
            return

        # 💡 [직접 수정 부분: 무차별 대입 기반 교차 검증]
        # 6X6_50 패치 실패 후, 화면에 사각형 후보(rejected)가 남아 있다면 다른 딕셔너리로 강제 분석합니다.
        if rejected is not None and len(rejected) > 0:
            for dict_name, alt_dict in self.alt_dicts.items():
                alt_detector = cv2.aruco.ArucoDetector(alt_dict, self.aruco_params)
                alt_corners, alt_ids, _ = alt_detector.detectMarkers(equalized)

                # 다른 규격 필터와 딱 맞아떨어지는 실체를 잡았을 때
                if alt_ids is not None:
                    # 주황 네모 + ID (규격 불일치 경고)
                    cv2.aruco.drawDetectedMarkers(display, alt_corners, alt_ids, borderColor=(0, 165, 255))
                    real_id = int(alt_ids[0][0])
                    self.get_logger().warn(
                        f"🚨 [정체 발각] 간판과 달리 이 마커의 진짜 규격은 👉 [{dict_name}] 이며, "
                        f"실제 식별 ID는 👉 [{real_id}] 번입니다! 코드 설정을 이 규격으로 바꾸셔야 작동합니다!"
                    )
                    cv2.putText(display, f"{dict_name}  ID={real_id} (규격불일치)", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    self.publish_id(real_id)
                    self.show(display)
                    return

            # 모든 백업 규격으로도 해독이 불가능한 픽셀 노이즈일 때 → 빨간 후보 박스 표시
            cv2.aruco.drawDetectedMarkers(display, rejected, borderColor=(0, 0, 255))
            self.get_logger().info(
                f"⚠️ [비트 불일치 디버그] 화면에서 사각형 {len(rejected)}개를 스캔했으나 "
                f"OpenCV 표준 딕셔너리 규격 외의 임의의 이미지 패널입니다. (인쇄 오류 가능성 높음)",
                throttle_duration_sec=2.0
            )

        cv2.putText(display, "NO MARKER", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.get_logger().info("🔍 마커를 찾고 있으나 영상 속에 유효한 아루코 마커가 보이지 않습니다.", throttle_duration_sec=2.0)
        self.show(display)

    def update_stop_signal(self, ids):
        """지정 ID 마커의 등장/소멸을 디바운스해 /aruco_stop(Bool) 발행. 매 프레임 호출.
        - 등장(보임)  : stop_on_frames 연속 보이면 즉시 정지(True)  → 장애물 밟기 방지
        - 소멸(안보임): go_after_frames 연속 안 보여야 출발(False)  → 경계 깜빡임 방지"""
        target   = self.get_parameter('target_marker_id').value
        stop_on  = self.get_parameter('stop_on_frames').value
        go_after = self.get_parameter('go_after_frames').value

        seen = ids is not None and target in [int(x) for x in ids.flatten()]
        if seen:
            self.seen_count += 1
            self.notseen_count = 0
        else:
            self.notseen_count += 1
            self.seen_count = 0

        if self.seen_count >= stop_on:
            self.stop_state = True
        elif self.notseen_count >= go_after:
            self.stop_state = False

        m = Bool()
        m.data = bool(self.stop_state)
        self.stop_pub.publish(m)

    def show(self, img):
        # 현재 /aruco_stop 상태 배너 (장애물 정지 신호를 눈으로 확인)
        if self.stop_state:
            cv2.putText(img, "OBSTACLE -> STOP", (20, img.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(img, "CLEAR -> GO", (20, img.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        # 시각화 창 (창 크기 조절 가능)
        cv2.imshow("ArUco Detection", img)
        cv2.waitKey(1)

    def publish_id(self, marker_id):
        out_msg = Int32()
        out_msg.data = marker_id
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()