"""노란 차선 인지 노드 (지름길/링 전용, 순수 인지).

lane_detection/lane_node.py 를 그대로 복사해 만들었다. 알고리즘은 동일하다:
행별 클러스터 추적으로 좌/우 차선을 잡고, 한쪽만 보이면 학습한 차선폭으로
반대편을 추정한다. 입력만 노란 마스크(/opencv/image/yellow)로 바꿨다.

왜 lane_node 를 재사용하지 않고 복사했나 (2026-07-10, 사용자 결정):
  - 앞으로 이 노드에는 링(원형 차로) 주행과 탈출 알고리즘이 들어간다.
    12시 마커 카운트, 안쪽/바깥쪽 경계 선택, 곡률 추종 같은 것들이다.
  - 같은 노드를 두 인스턴스로 쓰면 그 변경이 흰 차선 주행까지 건드린다.
    흰 차선은 트랙에서 검증이 끝난 코드라 절대 건드리면 안 된다.
  - 그래서 지금은 동일한 사본이지만, 앞으로 갈라진다.

발행: LaneDetection on /yellow/lane  (흰 차선과 같은 메시지 타입)
      디버그 오버레이 on /yellow_lane/image/debug
소비: interpret 의 RampEntry 가 구독해 커밋/추종을 판단한다.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from interface.msg import LaneDetection


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class RampDetectionNode(Node):
    def __init__(self):
        super().__init__('ramp_detection_node')

        # --- ROS parameters -------------------------------------------------
        self.declare_parameter('edge_topic', '/opencv/image/yellow')
        self.declare_parameter('detection_topic', '/yellow/lane')
        self.declare_parameter('debug_topic', '/yellow_lane/image/debug')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('num_scan_rows', 12)     # ROI 안에서 스캔할 가로줄 개수
        self.declare_parameter('min_detect_rows', 3)    # 차선으로 인정할 최소 검출 줄 수
        # 2026-07-05 실측: 이 트랙 차선폭 ≈ 178px(0.556×320). 기본/범위를 실측에 맞춤.
        self.declare_parameter('default_lane_width_ratio', 0.556)  # 초기 차선폭(이미지폭 대비)
        # 학습된 차선폭(px)을 이미지폭 대비 이 범위로 clamp. 단일차선 추종 시 half 가
        # 과도하게 커져(=반대편으로 overshoot) 반대 차선을 넘는 것을 좌우 대칭으로 방지.
        self.declare_parameter('lane_width_min_ratio', 0.42)
        self.declare_parameter('lane_width_max_ratio', 0.62)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_log', False)  # lane_width/검출상태 진단 로그

        edge_topic = str(self.get_parameter('edge_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.num_scan_rows = max(2, int(self.get_parameter('num_scan_rows').value))
        self.min_detect_rows = max(1, int(self.get_parameter('min_detect_rows').value))
        self.default_lane_width_ratio = float(self.get_parameter('default_lane_width_ratio').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_image = bool(self.get_parameter('debug_image').value)

        # --- vehicle_config.yaml 에서 ROI 읽기 ------------------------------
        self.roi_top, self.roi_left = self.load_roi()
        # config 값을 기본값으로 하되, 실시간 튜닝을 위해 ROS 파라미터로도 노출.
        # detect_lane / publish_debug 는 매 프레임 파라미터를 다시 읽으므로
        # `ros2 param set /lane_detection_node roi_top N` 으로 라이브 조정 가능.
        self.declare_parameter('roi_top', int(self.roi_top))
        self.declare_parameter('roi_left', int(self.roi_left))
        # 라인 피팅 이상치 제거 임계값(px). 이 값보다 선에서 멀면 사물로 보고 버림.
        self.declare_parameter('line_fit_outlier_px', 12.0)
        # 피팅 차수. 근거리 밴드엔 1(직선)이 안정적. 2는 과적합→가짜곡선 요동.
        self.declare_parameter('line_fit_degree', 1)
        # 단일선 판별: 좌x·우x 간격이 (차선폭 * 이 비율)보다 작으면 사실 같은 선
        # 하나가 중심을 가로질러 좌/우로 잘린 것으로 보고 하나의 차선으로 병합.
        self.declare_parameter('single_line_gap_ratio', 0.55)
        # --- 좌/우 분류(클러스터 추적)용 ---
        # cluster_gap_px: 한 행에서 이 간격(px) 이하로 붙은 엣지 픽셀은 한 선으로 묶음.
        # (한 선의 Canny 양쪽 엣지는 붙여서 1개로, 서로 다른 두 차선은 분리)
        self.declare_parameter('cluster_gap_px', 30.0)
        # min_lane_sep_ratio: 근거리 씨앗에서 두 클러스터가 (이 비율*이미지폭) 이상
        # 떨어져야 두 차선으로 인정. 미만이면 단일선(중앙 걸침)으로 취급 → 유령선 방지.
        self.declare_parameter('min_lane_sep_ratio', 0.2)
        # track_tol_px: 인접 스캔행 간 같은 차선으로 매칭할 최대 x 이동(px).
        self.declare_parameter('track_tol_px', 40.0)

        # =====================================================================
        # 12시 가로 마커 + 점선 추종 (링 한 바퀴)
        # =====================================================================
        # 트랙: 링을 반시계로 돈다(중앙섬이 차의 왼쪽). 12시에 노란 '가로 실선' 마커가
        # 있고, 그 옆으로 출구가 갈라진다. 한 바퀴를 돌려면 첫 12시에서 안 나가야 한다.
        # 사용자 관찰: 12시에서는 **오른쪽 점선**을 따라가면 링에 남는다.
        # (진입 구간은 왼쪽 점선/오른쪽 실선이라, '실선=오른쪽' 같은 고정 규칙은 못 쓴다.)
        self.declare_parameter('marker_enabled', True)
        # 한 행에서 '연속으로 이어진' 노란 픽셀의 최대 길이(px)가 이 값 이상이면 가로 마커.
        # 2026-07-10 실측 (320px 폭):
        #   정지 상태 세로 차선   : run 34px   (행 채움비 0.206)
        #   **링 주행 중** 세로 차선 : run 28~68px  <- 곡선에서 선이 비스듬해지면 길어진다
        #   12시 마커             : run 138~220px (행 채움비 0.456)
        # 행 채움비(fill)는 차의 좌우 위치에 따라 흔들린다(마커가 화면 한쪽만 채우므로).
        # 연속 런은 마커가 시야에 있기만 하면 길게 나온다. 그래서 런 길이를 쓴다.
        # 문턱 70px 은 주행 중 오검출(68px)에 걸렸다 -> 120px (68 과 138 의 정중앙).
        self.declare_parameter('marker_run_px', 120)
        # 마커로 인정할 최소 '가로행 수'. 진짜 12시 실선은 두꺼운 밴드라 여러 행이
        # run≥marker_run_px 를 만족한다(2026-07-12 run3 실측: 4~16행, 피크 213px).
        # 오검출(곡선에서 비스듬해진 세로차선이 순간 120px 런)은 1행뿐 -> 이걸로 배제.
        # (run3: #1 진짜=밴드 4~16행, #2 거짓=1행. min_rows=3 이면 #2 제거, #1 유지.)
        self.declare_parameter('marker_min_rows', 3)
        # 마커 판정 디바운스(연속 프레임). rising-edge 로 랩 카운트.
        self.declare_parameter('marker_frames', 2)
        # 마커를 한 번 센 뒤 이 프레임 수 동안은 다시 세지 않는다(중복 카운트 방지).
        # 2026-07-10: run 이 문턱을 넘나들 때마다 marker_active 가 풀렸다 걸려 한 바퀴도
        # 안 돌았는데 마커를 9번 세었다. 마커 하나를 지나는 데 걸리는 시간보다 길게 잡는다.
        self.declare_parameter('marker_cooldown_frames', 90)   # ≈3초 @30Hz
        # (DEPRECATED, 미사용) 예전엔 마커 후 이 프레임 수만큼 점선추종을 유지했으나,
        # 2026-07-12 마커 카운트 기반 유지로 바뀜: 1번째 마커~2번째 마커 사이(marker_count==1)
        # 내내 점선추종, 2번째 마커 후 일반차선 복귀(탈출). 프레임 타임아웃 게이팅 폐기.
        self.declare_parameter('marker_hold_frames', 120)
        # 마커 구간에서 앵커할 선 선택. **'dashed' = 실선 제외, 점선 중 맨 오른쪽(기본).**
        # 2026-07-12 실측: 12시 출구엔 점선(고어)+바깥 실선이 같이 보이는데 'right'(맨
        # 오른쪽 선)는 바깥 '실선'을 물고 출구로 나가버렸다. 점선(끊김=검출 행 적음)만
        # 후보로 삼고 실선(연속=행 많음, ≥solid_row_min)은 뺀 뒤, 점선이 여럿이면 맨
        # 오른쪽을 고른다. 'right'/'left'=위치 고정, 'auto'=행 적은 선 자동(점선2개면 불안정).
        self.declare_parameter('marker_follow_side', 'dashed')
        # 이 행 수 이상 검출되면 '실선'으로 보고 점선 후보에서 제외(num_scan_rows=12 기준).
        self.declare_parameter('solid_row_min', 8)
        # 점선에서 '얼마나 왼쪽(안쪽)을 겨냥할지'(px). 예전엔 lane_width_px/2(≈99)를 썼는데
        # lane_width_px 학습이 오염돼(≈198, 실제 간격 ~85) 목표가 과도하게 왼쪽으로 쏠렸다.
        # 이 값으로 분리해 직접 튜닝한다(작을수록 점선에 더 붙어 감).
        self.declare_parameter('dash_aim_px', 45)
        # 'auto' 동점 가드: 두 선이 다 후보일 때 행 수 차이가 이 값 이상이어야 '적은 선'을
        # 점선으로 고른다. 근소차/동점은 애매 -> 직전 선택 유지(07-10 포화 재발 방지).
        self.declare_parameter('dashed_row_margin', 2)
        # 점선 공백 다리(bridge) + 연속성 게이트: 오른쪽 점선(파란선)은 끊겨 있어 순간
        # 검출이 비거나(공백), 4시 진입 접합부에선 점선보다 입구 실선이 먼저 보여 앵커가
        # 엉뚱한 데로 튄다. 그때 직전 앵커(lane_center)를 이 프레임 수까지 유지하며
        # 코스팅해 진짜 점선을 재획득할 때까지 버틴다(2026-07-12 실측: 공백/4시 이탈 대응).
        self.declare_parameter('dash_hold_frames', 30)   # ≈1초 @30Hz
        # 연속성 게이트: 새 앵커가 직전 앵커보다 이 px 이상 튀면 '다른 선을 문 것'으로 보고
        # 버린 뒤 직전 앵커를 유지한다(4시 입구 실선이 맨 오른쪽에 나타나도 안 끌려감).
        self.declare_parameter('max_anchor_jump_px', 50)
        # 마커에 걸린 스캔행은 차선 검출에서 제외한다. 안 그러면 가로선이 좌/우 차선으로
        # 잘못 잡혀(한 행이 통째로 한 클러스터) 검출이 오염된다.
        self.declare_parameter('marker_exclude_rows', True)

        # --- 내부 상태 ------------------------------------------------------
        # 차선폭(px)은 기하 상태라 인지에 둔다. 양쪽 검출 시 EMA로 학습해
        # 한쪽만 보일 때 반대편 차선 위치를 추정하는 데 쓴다. (시간 평활 아님)
        self.lane_width_px = None

        # --- 12시 마커 상태 --------------------------------------------------
        self.marker_count = 0        # 마커 통과 횟수(랩 카운터)
        self.marker_active = False   # 지금 마커 위인가(중복 카운트 방지 래치)
        self.marker_on = 0           # 마커 감지 디바운스(rising)
        self.marker_off = 0          # 마커 해제 디바운스(falling)
        self.cooldown_left = 0       # 이 프레임 동안은 마커를 다시 세지 않는다
        self._dash_prev_side = None  # 'auto' 애매 프레임에서 유지할 직전 점선 쪽
        self._last_dash_center = None  # 점선 공백 다리용: 직전 유효 앵커 lane_center
        self._dash_miss = 0            # 점선 연속 공백(미검출) 프레임 수

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            edge_topic,
            self.image_callback,
            image_qos,
        )
        self.detection_pub = self.create_publisher(LaneDetection, detection_topic, 10)
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(CompressedImage, debug_topic, image_qos)

        self.get_logger().info(
            'ramp_detection node started (yellow lane, perception only):\n'
            f'  edge_topic={edge_topic}\n'
            f'  detection_topic={detection_topic}\n'
            f'  roi_top={self.roi_top}, roi_left={self.roi_left}\n'
            f'  num_scan_rows={self.num_scan_rows}, min_detect_rows={self.min_detect_rows}\n'
            f'  debug_image={self.debug_image}'
        )

    # ------------------------------------------------------------------ config
    def load_roi(self):
        roi_top, roi_left = 0, 0
        if not os.path.exists(self.vehicle_config_file):
            self.get_logger().warning(
                f'vehicle config not found ({self.vehicle_config_file}); ROI defaults 0.'
            )
            return roi_top, roi_left
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(f'Failed to read vehicle config: {exc}')
            return roi_top, roi_left
        roi_top = int(config_data.get('ROI_TOP', 0))
        roi_left = int(config_data.get('ROI_LEFT', 0))
        return roi_top, roi_left

    # ------------------------------------------------------- 12시 마커 / 점선추종
    @staticmethod
    def max_run(row):
        """한 행에서 연속으로 이어진 0 아닌 픽셀의 최대 길이(px)."""
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return 0
        # 연속 구간의 경계에서 끊고, 각 구간 길이의 최대를 취한다.
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [idx.size - 1]))
        return int((ends - starts).max()) + 1

    def marker_rows(self, edge):
        """ROI 안에서 '가로 마커'에 걸린 스캔행들을 찾는다.
        한 행의 '연속 런 길이'가 marker_run_px 이상이면 그 행은 가로선 위다.
        세로 차선은 한 행에서 34px 정도만 연속이고, 가로 마커는 138px 까지 간다(실측)."""
        height, width = edge.shape
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        roi_left = min(max(int(self.get_parameter('roi_left').value), 0), width - 1)
        thr = int(self.get_parameter('marker_run_px').value)
        rows = []
        best = 0
        for y in range(roi_top, height):
            run = self.max_run(edge[y, roi_left:])
            if run > best:
                best = run
            if run >= thr:
                rows.append(y)
        return rows, best

    def update_marker(self, has_marker, rows=None, best=0):
        """마커를 rising-edge 로 카운트하고, 지나간 뒤에도 점선 추종을 유지할
        hold 프레임을 채운다. 마커 위에 올라서는 '그 순간' 한 번만 +1.

        쿨다운: 한 번 센 뒤 marker_cooldown_frames 동안은 다시 세지 않는다. run 이
        문턱을 넘나들며 marker_active 가 풀렸다 걸리는 것을 막는다(중복 카운트).

        진단(2026-07-12): 검출 순간의 run 길이·걸린 행 수·행 y범위를 찍는다. 진짜 12시
        가로실선은 두꺼운 밴드라 여러 행(넓은 y범위)이 걸리고, 곡선에서 비스듬해진
        세로차선의 오검출은 1~2행에 그친다 -> 이걸로 진짜/거짓을 판별한다. 쿨다운에
        억제된 검출도 남겨 트리거 빈도를 본다."""
        need = max(1, int(self.get_parameter('marker_frames').value))
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

        if has_marker:
            self.marker_on += 1
            self.marker_off = 0
        else:
            self.marker_off += 1
            self.marker_on = 0

        if not self.marker_active and self.marker_on >= need:
            self.marker_active = True
            span = (f'{min(rows)}-{max(rows)}' if rows else '-')
            nrows = len(rows) if rows else 0
            if self.cooldown_left == 0:
                self.marker_count += 1
                self.cooldown_left = int(
                    self.get_parameter('marker_cooldown_frames').value)
                self._dash_prev_side = None   # 새 마커 -> 점선 선택 초기화(기본 right)
                self._last_dash_center = None  # 새 마커 -> 공백 다리 상태 초기화
                self._dash_miss = 0
                # 1번째 = 점선추종 시작(2번째 마커까지), 2번째 = 탈출(일반차선 복귀).
                mode = ('점선추종 시작(2번째 마커까지)' if self.marker_count == 1
                        else '일반차선 복귀=탈출' if self.marker_count == 2
                        else '추가마커')
                self.get_logger().info(
                    f'12시 마커 통과 #{self.marker_count} -> {mode} '
                    f'| run={best}px rows={nrows} y=[{span}]')
            else:
                self.get_logger().info(
                    f'마커 감지(쿨다운 억제, cd={self.cooldown_left}) '
                    f'run={best}px rows={nrows} y=[{span}]')
        elif self.marker_active and self.marker_off >= need:
            self.marker_active = False

    def follow_dashed(self, result):
        """12시 마커 구간: '노란색 발행이 적은 선'(=점선)에 앵커해 링 안에 남는다.

        왜 점선인가: 12시에서 출구가 갈라진다. 왼쪽 실선(픽셀 많음)을 물거나 두 선의
        '중앙'을 잡으면 출구로 끌려나간다. 오른쪽 점선(고어, 픽셀 적음)만 계속 따라가면
        링에 남는다. 점선은 끊겨 있어 검출 행 수가 실선보다 적다 -> 이걸로 고른다.

        선택(marker_follow_side): 'auto' = 후보(≥min_detect_rows) 중 행 수가 더 적은
        선을 점선으로. 화면 위치가 아니라 픽셀 양으로 고르므로 곡선에서 선이 좌/우로
        넘어가도(뒤집힘) 강건하다. 'right'/'left' = 위치 고정(예전 동작).

        동점 가드: 2026-07-10 실측에서 좌/우가 각각 1행씩만 잡힌 프레임(마커 잔재)이
        동점을 만들어 반대쪽을 골랐고 offset 이 -1.000 으로 포화했다. 그래서
        (a) 후보는 min_detect_rows 이상만, (b) 두 선이 다 후보면 행 수 차이가
        dashed_row_margin 이상일 때만 '적은 선'을 고르고, 근소차/동점은 직전 선택을
        유지한다."""
        side = str(self.get_parameter('marker_follow_side').value)
        left_pts, right_pts = result['left_pts'], result['right_pts']
        ln, rn = len(left_pts), len(right_pts)
        left_ok = ln >= self.min_detect_rows
        right_ok = rn >= self.min_detect_rows

        # 1) 이번 프레임에 앵커할 점선을 고른다(없으면 pts=None).
        pts, chosen = None, None
        if side == 'dashed':
            # 실선을 아예 빼지 않는다: 점선(끊김=행 적음)과 실선(연속=행 많음)이 둘 다 있으면
            # **점선 우선**, 점선이 하나도 없으면 실선으로 폴백(정지 방지). 같은 부류 중에선
            # 맨 오른쪽. -> 12시 출구에서 점선이 보이면 점선(링 유지), 점선을 놓친 프레임엔
            # 실선이라도 잡아 계속 굴러가되, 점선이 돌아오면 다시 점선 우선.
            solid_min = int(self.get_parameter('solid_row_min').value)
            dashed_c, solid_c = [], []
            if ln >= self.min_detect_rows:
                x = float(np.median([px for _, px in left_pts]))
                (dashed_c if ln < solid_min else solid_c).append((x, left_pts))
            if rn >= self.min_detect_rows:
                x = float(np.median([px for _, px in right_pts]))
                (dashed_c if rn < solid_min else solid_c).append((x, right_pts))
            picks = dashed_c if dashed_c else solid_c   # 점선 우선, 없으면 실선 폴백
            if picks:
                picks.sort(key=lambda c: c[0])          # x 오름차순
                _, pts = picks[-1]                       # 그중 맨 오른쪽
                chosen = 'dashed'
        elif side in ('right', 'left'):
            cand = right_pts if side == 'right' else left_pts
            if len(cand) >= self.min_detect_rows:
                pts, chosen = cand, side
        else:  # 'auto': 노란색 적은 선(행 수 적은 선) = 점선
            margin = int(self.get_parameter('dashed_row_margin').value)
            if left_ok and right_ok:
                if abs(ln - rn) >= margin:
                    chosen = 'left' if ln < rn else 'right'
                elif self._dash_prev_side in ('left', 'right'):
                    chosen = self._dash_prev_side      # 애매 -> 직전 유지
                else:
                    chosen = 'right'                   # 기본: 12시 점선 위치
            elif left_ok:
                chosen = 'left'
            elif right_ok:
                chosen = 'right'
            if chosen is not None:
                pts = left_pts if chosen == 'left' else right_pts

        width = result['image_width']
        center_x = result['center_x']
        # 완전 오른쪽 점선 전용(2026-07-12, 사용자 요청): 오른쪽 점선을 못 본 프레임엔
        # 왼쪽/중앙잡기(detect_lane 결과)로 폴백하지 않는다 — 왼쪽에 끌려 안쪽으로 이탈하는
        # 것을 막기 위함. 대신 그 프레임은 신뢰도 0 으로 둬 조향 목표를 만들지 않고(오른쪽
        # 점선을 다시 볼 때까지 대기), 하류 페일세이프에 맡긴다. (hold/코스팅도 없음.)
        if pts is None:
            out = dict(result)
            out['lane_center'] = None
            out['raw_offset'] = 0.0
            out['confidence'] = 0.0
            out['left_detected'] = False
            out['right_detected'] = False
            out['dashed_anchored'] = False
            out['dash_held'] = False
            return out
        self._dash_prev_side = chosen
        dashed_x = float(np.median([x for _, x in pts]))
        # 점선(고어)은 링 차로의 '오른쪽 경계' -> 차로 중심은 그 왼쪽 dash_aim_px.
        # (예전 lane_width_px/2 대신 별도 튠값. side='left' 고정일 때만 오른쪽으로.)
        aim = float(self.get_parameter('dash_aim_px').value)
        lane_center = dashed_x + aim if side == 'left' else dashed_x - aim
        held = False

        out = dict(result)
        out['lane_center'] = lane_center
        out['raw_offset'] = float(
            np.clip((lane_center - center_x) / (width / 2.0), -1.0, 1.0))
        # 점선 하나만(또는 유지값) 믿고 가는 구간이라 신뢰도 바닥을 줘 페일세이프 오정지 방지.
        out['confidence'] = max(result['confidence'], 0.5)
        out['left_detected'] = out['right_detected'] = True
        out['dashed_anchored'] = True
        out['dash_held'] = held
        return out

    # ------------------------------------------------------------------ decode
    def decode_edge(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        edge = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            self.get_logger().warning('Failed to decode edge image')
        return edge

    # --------------------------------------------------------------- detection
    def detect_lane(self, edge):
        """ROI 안에서 행별로 좌/우 차선 x좌표를 찾아 '그 순간'의 차선 중심과
        offset/confidence 를 계산한다. 시간 평활은 하지 않는다."""
        height, width = edge.shape
        center_x = width / 2.0
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        roi_left = min(max(int(self.get_parameter('roi_left').value), 0), width - 1)

        if self.lane_width_px is None:
            self.lane_width_px = self.default_lane_width_ratio * width

        scan_ys = np.linspace(roi_top, height - 1, self.num_scan_rows).astype(int)

        # 좌/우 차선 점 검출: 화면 중심으로 자르지 않고, 행별 엣지 픽셀을 '선(클러스터)'
        # 으로 묶은 뒤 근거리(하단)에서 차선 수를 확정하고 위로 추적한다. 중앙 근처에
        # 걸친 한 개의 선이 좌/우 두 개로 쪼개지는 오분류(유령 반대선)를 막는다.
        left_raw, right_raw = self.scan_lanes(edge, scan_ys, roi_left, center_x, width)

        # 검출점에 다항식(곡선)을 피팅해 선에서 벗어난 엉뚱한 사물 점(이상치)을
        # 걸러내고, 차선을 정교한 곡선으로 표현한다.
        left_pts, left_poly = self.fit_and_filter(left_raw)
        right_pts, right_poly = self.fit_and_filter(right_raw)

        left_detected = len(left_pts) >= self.min_detect_rows
        right_detected = len(right_pts) >= self.min_detect_rows
        left_x = float(np.median([x for _, x in left_pts])) if left_detected else None
        right_x = float(np.median([x for _, x in right_pts])) if right_detected else None

        # --- 단일선 판별 (곡선에서 한 선이 중심을 가로질러 좌/우로 잘리는 문제) ---
        # 좌x·우x 간격이 실제 차선폭보다 훨씬 작으면 둘은 같은 선. 하나의 차선으로
        # 병합하고, 근거리(맨 아래) 위치가 중심의 어느 쪽인지로 좌/우를 판정한다.
        if left_detected and right_detected:
            ref_width = self.lane_width_px if self.lane_width_px else float(width)
            gap_ratio = float(self.get_parameter('single_line_gap_ratio').value)
            if (right_x - left_x) < gap_ratio * ref_width:
                all_pts = left_pts + right_pts
                _, x_near = max(all_pts, key=lambda p: p[0])  # 가장 아래(근거리) 점
                line_x = float(np.median([x for _, x in all_pts]))
                line_poly = left_poly if left_poly is not None else right_poly
                if x_near < center_x:      # 근거리에서 중심 왼쪽 -> 좌차선
                    left_detected, right_detected = True, False
                    left_x, right_x = line_x, None
                    left_pts, right_pts = all_pts, []
                    left_poly, right_poly = line_poly, None
                else:                       # 근거리에서 중심 오른쪽 -> 우차선
                    left_detected, right_detected = False, True
                    left_x, right_x = None, line_x
                    left_pts, right_pts = [], all_pts
                    left_poly, right_poly = None, line_poly

        # 필터된 점으로 per-row 차선중심 재구성 (단일선 병합 반영)
        left_map = {y: x for y, x in left_pts}
        right_map = {y: x for y, x in right_pts}
        center_pts = []
        for y in scan_ys:
            y = int(y)
            lx = left_map.get(y)
            rx = right_map.get(y)
            if lx is not None and rx is not None:
                center_pts.append((y, (lx + rx) / 2.0))
            elif lx is not None:
                center_pts.append((y, lx + self.lane_width_px / 2.0))
            elif rx is not None:
                center_pts.append((y, rx - self.lane_width_px / 2.0))

        # 양쪽 검출 시 차선폭 학습(EMA) — 기하 상태 추정(시간 평활 아님)
        if left_detected and right_detected and right_x > left_x:
            self.lane_width_px = 0.8 * self.lane_width_px + 0.2 * (right_x - left_x)

        # 차선폭을 안전 범위로 clamp -> 단일차선 half overshoot(반대선 침범) 방지.
        w_min = float(self.get_parameter('lane_width_min_ratio').value) * width
        w_max = float(self.get_parameter('lane_width_max_ratio').value) * width
        if w_max > w_min:
            self.lane_width_px = float(np.clip(self.lane_width_px, w_min, w_max))

        half = self.lane_width_px / 2.0
        if left_detected and right_detected:
            lane_center = (left_x + right_x) / 2.0
        elif left_detected:
            lane_center = left_x + half
        elif right_detected:
            lane_center = right_x - half
        else:
            lane_center = None

        detected_rows = len(center_pts)
        confidence = detected_rows / float(self.num_scan_rows)

        # raw offset: 그 순간의 정규화 횡오차. 미검출 시엔 0(=값 유지는 interpret 담당).
        if lane_center is not None:
            raw_offset = (lane_center - center_x) / (width / 2.0)
            raw_offset = float(np.clip(raw_offset, -1.0, 1.0))
        else:
            raw_offset = 0.0
            confidence = 0.0  # 완전 미검출: 신뢰도 0

        return {
            'raw_offset': raw_offset,
            'left_detected': left_detected,
            'right_detected': right_detected,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'lane_center': lane_center,
            'center_x': center_x,
            'image_width': int(width),
            'image_height': int(height),
            'left_pts': left_pts,
            'right_pts': right_pts,
            'left_poly': left_poly,
            'right_poly': right_poly,
        }

    def row_clusters(self, edge_row, roi_left, cluster_gap):
        """한 행의 엣지 픽셀을 x 간격 기준으로 묶어 클러스터 목록을 만든다.
        각 클러스터 = (mean_x, min_x, max_x). x(mean) 오름차순 정렬."""
        xs = np.where(edge_row[roi_left:] > 0)[0]
        if xs.size == 0:
            return []
        xs = np.sort(xs + roi_left)
        if xs.size == 1:
            x = int(xs[0])
            return [(float(x), x, x)]
        splits = np.where(np.diff(xs) > cluster_gap)[0]
        groups = np.split(xs, splits + 1)
        clusters = [(float(g.mean()), int(g.min()), int(g.max())) for g in groups]
        clusters.sort(key=lambda c: c[0])
        return clusters

    def scan_lanes(self, edge, scan_ys, roi_left, center_x, width):
        """행별 클러스터를 근거리(하단)→원거리(상단)로 추적해 좌/우 차선 점열을 만든다.

        목표: (1) 두 선이 있으면 둘 다 잡아 '두 선 사이 중앙'을 유지(정상 동작),
              (2) 한 선만 있으면(중앙에 걸쳐도) 유령 반대선을 만들지 않는다.

        - 매 행: 먼저 기존 좌/우 차선을 가장 가까운 클러스터에 track_tol 내에서
          매칭·갱신. 그다음 아직 없는 차선을 '충분히 떨어진(≥min_lane_sep)'
          미사용 클러스터에서 새로 시작한다 → 두 번째 선이 위쪽에서 늦게 나타나도
          받아들이되(정상 두 선 복원), 단일선은 행마다 클러스터가 하나뿐이라
          먼 미사용 클러스터가 없어 유령선이 생기지 않는다.
        - 좌차선 점은 안쪽 엣지(=오른쪽=max_x), 우차선은 안쪽(=왼쪽=min_x)을 기록해
          기존 캘리브레이션(차선폭/half) 관례를 유지한다."""
        cluster_gap = float(self.get_parameter('cluster_gap_px').value)
        track_tol = float(self.get_parameter('track_tol_px').value)
        min_lane_sep = float(self.get_parameter('min_lane_sep_ratio').value) * width

        left_raw, right_raw = [], []
        left_ref, right_ref = None, None  # 각 차선의 직전 행 mean x (추적 기준)

        def nearest_unused(ref, means, used):
            cand = [(abs(means[k] - ref), k) for k in range(len(means)) if k not in used]
            return min(cand)[1] if cand else None

        for y in sorted((int(v) for v in scan_ys), reverse=True):  # 근거리부터
            clusters = self.row_clusters(edge[y], roi_left, cluster_gap)
            if not clusters:
                continue
            means = [c[0] for c in clusters]
            used = set()

            # 1) 기존 차선 추적: 가장 가까운 미사용 클러스터를 tol 내에서 매칭
            if left_ref is not None:
                j = nearest_unused(left_ref, means, used)
                if j is not None and abs(means[j] - left_ref) <= track_tol:
                    left_ref = means[j]
                    left_raw.append((y, clusters[j][2]))  # 좌 안쪽 엣지 = max_x
                    used.add(j)
            if right_ref is not None:
                j = nearest_unused(right_ref, means, used)
                if j is not None and abs(means[j] - right_ref) <= track_tol:
                    right_ref = means[j]
                    right_raw.append((y, clusters[j][1]))  # 우 안쪽 엣지 = min_x
                    used.add(j)

            remaining = [k for k in range(len(clusters)) if k not in used]

            # 2) 아직 없는 차선을 '충분히 떨어진' 미사용 클러스터에서 시작
            if left_ref is None and right_ref is None:
                if len(remaining) >= 2 and \
                        (means[remaining[-1]] - means[remaining[0]]) >= min_lane_sep:
                    # 두 선 동시 씨앗 (최좌=좌, 최우=우)
                    a, b = remaining[0], remaining[-1]
                    left_ref, right_ref = means[a], means[b]
                    left_raw.append((y, clusters[a][2]))
                    right_raw.append((y, clusters[b][1]))
                elif remaining:
                    # 단일선(또는 붙은 덩어리): 한 덩어리로 보고 화면 중심 기준 한쪽만
                    all_min = min(clusters[k][1] for k in remaining)
                    all_max = max(clusters[k][2] for k in remaining)
                    m = 0.5 * (all_min + all_max)
                    if m < center_x:
                        left_ref = m
                        left_raw.append((y, all_max))
                    else:
                        right_ref = m
                        right_raw.append((y, all_min))
            elif left_ref is None:
                # 우차선만 있음 → 우차선보다 min_lane_sep 이상 왼쪽인 클러스터로 좌차선 시작
                cands = [k for k in remaining if right_ref - means[k] >= min_lane_sep]
                if cands:
                    k = min(cands, key=lambda k: means[k])
                    left_ref = means[k]
                    left_raw.append((y, clusters[k][2]))
            elif right_ref is None:
                # 좌차선만 있음 → 좌차선보다 min_lane_sep 이상 오른쪽인 클러스터로 우차선 시작
                cands = [k for k in remaining if means[k] - left_ref >= min_lane_sep]
                if cands:
                    k = max(cands, key=lambda k: means[k])
                    right_ref = means[k]
                    right_raw.append((y, clusters[k][1]))

        left_raw.sort()
        right_raw.sort()
        return left_raw, right_raw

    def fit_degree(self, n_points):
        """요청 차수를 파라미터에서 읽되, 점 수로 상한을 둔다(차수 = 점수-1 이하).
        근거리 밴드엔 기본 1차(직선)가 안정적."""
        req = int(self.get_parameter('line_fit_degree').value)
        return max(1, min(req, n_points - 1))

    def fit_and_filter(self, pts):
        """검출점들에 다항식 x=f(y)를 피팅(차선은 수직에 가까움)해 이상치를
        제거하고 (필터된 점 리스트, np.poly1d 또는 None)을 반환한다.
        점이 적으면 그대로 반환. 차수는 line_fit_degree 파라미터(기본 1차)."""
        if len(pts) < 3:
            return list(pts), None
        ys = np.array([p[0] for p in pts], dtype=np.float64)
        xs = np.array([p[1] for p in pts], dtype=np.float64)
        try:
            poly = np.poly1d(np.polyfit(ys, xs, self.fit_degree(len(pts))))
        except Exception:
            return list(pts), None

        resid = np.abs(xs - poly(ys))
        thresh = max(
            float(self.get_parameter('line_fit_outlier_px').value),
            2.5 * float(np.std(resid)),
        )
        keep = resid <= thresh
        if keep.all() or keep.sum() < 2:
            return list(pts), poly

        # 이상치 제거 후 1회 재피팅으로 선을 더 정교화
        ys2, xs2 = ys[keep], xs[keep]
        try:
            degree2 = self.fit_degree(int(ys2.size))
            poly = np.poly1d(np.polyfit(ys2, xs2, degree2))
        except Exception:
            pass
        filtered = [(int(y), int(x)) for y, x in zip(ys2, xs2)]
        return filtered, poly

    # ------------------------------------------------------------------ callbk
    def image_callback(self, msg: CompressedImage):
        edge = self.decode_edge(msg)
        if edge is None:
            return

        # --- 12시 가로 마커: 검출 -> 랩 카운트 -> 점선 추종 hold ---
        marker_on = False
        fill = 0
        mrows = 0
        if bool(self.get_parameter('marker_enabled').value):
            rows, fill = self.marker_rows(edge)
            # 진짜 12시 실선(두꺼운 밴드)만 인정: run≥문턱인 행이 min_rows 이상일 때.
            # 단일 세로선 오검출(1행)은 여기서 걸러진다.
            min_rows = int(self.get_parameter('marker_min_rows').value)
            marker_on = len(rows) >= min_rows
            mrows = len(rows)
            self.update_marker(marker_on, rows, fill)
            # 마커에 걸린 행을 지운 뒤 차선을 찾는다. 안 그러면 가로선이 한 행 전체를
            # 하나의 클러스터로 만들어 좌/우 차선 검출을 오염시킨다.
            if rows and bool(self.get_parameter('marker_exclude_rows').value):
                edge = edge.copy()
                for y in rows:
                    edge[y, :] = 0

        result = self.detect_lane(edge)

        # 점선(오른쪽 고어) 추종 활성 구간 = 1번째 12시 마커를 본 뒤 ~ 2번째 마커를 볼
        # 때까지(marker_count == 1). 프레임 타임아웃(marker_hold_frames)이 아니라 마커
        # 카운트로 유지 구간을 정한다 -> 링 한 바퀴 내내 출구로 안 새고 점선을 따라간다.
        # 2번째 마커를 보면(marker_count >= 2) 이제는 '탈출'해야 하므로 첫 마커 전과 같은
        # 일반 차선인지(detect_lane)로 복귀한다(follow_dashed 미적용).
        holding = (self.marker_count == 1)
        if holding:
            result = self.follow_dashed(result)

        if bool(self.get_parameter('debug_log').value):
            lc = result['lane_center']
            tag = ''
            if marker_on:
                tag += ' MARKER'
            if holding:
                ok = result.get('dashed_anchored')
                heldr = result.get('dash_held')
                if heldr == 'jump':
                    tag += f" DASH-HOLD(실선튐거부 {self._dash_miss})"
                elif heldr == 'gap':
                    tag += f" DASH-HOLD(공백다리 {self._dash_miss})"
                else:
                    tag += f" DASH{'' if ok else '(없음)'}"
            elif self.marker_count >= 2:
                tag += ' EXIT(일반차선복귀)'
            self.get_logger().info(
                f"lane_width_px={self.lane_width_px:.0f} "
                f"L={int(result['left_detected'])} R={int(result['right_detected'])} "
                f"run={fill:3d}px mrows={mrows} mcnt={self.marker_count} "
                f"lane_center={('%.0f' % lc) if lc is not None else 'None'} "
                f"raw_offset={result['raw_offset']:+.3f} conf={result['confidence']:.2f}{tag}",
                throttle_duration_sec=0.5,
            )

        detection = LaneDetection()
        detection.header.stamp = msg.header.stamp
        detection.header.frame_id = 'ramp_detection'
        detection.image_width = result['image_width']
        detection.image_height = result['image_height']
        detection.center_x = float(result['center_x'])
        detection.lane_center_px = (
            float(result['lane_center']) if result['lane_center'] is not None else -1.0
        )
        detection.raw_offset = result['raw_offset']
        detection.left_detected = result['left_detected']
        detection.right_detected = result['right_detected']
        detection.confidence = result['confidence']
        self.detection_pub.publish(detection)

        if self.debug_pub is not None:
            self.publish_debug(edge, result, msg)

    # ------------------------------------------------------------------- debug
    def publish_debug(self, edge, result, source_msg: CompressedImage):
        canvas = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        height, width = edge.shape
        center_x = int(result['center_x'])

        # ROI 상단 경계선(노랑)
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        cv2.line(canvas, (0, roi_top), (width, roi_top), (0, 255, 255), 1)
        # 이미지 중심선(흰색)
        cv2.line(canvas, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        # 좌/우 차선 검출점(좌=빨강, 우=파랑) — 참고용 작은 점
        for y, x in result['left_pts']:
            cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)
        for y, x in result['right_pts']:
            cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)
        # 피팅된 차선 곡선(좌=빨강, 우=파랑) — 정교한 실선
        for poly, color in ((result.get('left_poly'), (0, 0, 255)),
                            (result.get('right_poly'), (255, 0, 0))):
            if poly is None:
                continue
            ys = np.arange(roi_top, height)
            xs = np.clip(poly(ys), 0, width - 1).astype(np.int32)
            pts_line = np.stack([xs, ys.astype(np.int32)], axis=1)
            cv2.polylines(canvas, [pts_line], False, color, 2)
        # 차선 중심선(초록)
        if result['lane_center'] is not None:
            lc = int(result['lane_center'])
            cv2.line(canvas, (lc, roi_top), (lc, height), (0, 255, 0), 2)

        ok, encoded = cv2.imencode(
            '.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'lane_detection_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RampDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
