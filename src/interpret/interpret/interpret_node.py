"""차선 판단+제어결정 노드 (interpret).

lane_detection 이 발행하는 '그 순간'의 인지값(LaneDetection, /lane/detection)을
구독하여, 한 콜백 안에서 두 가지를 연속으로 수행한다:

  1) 판단(시간필터/안전): offset EMA 저역통과. 결과를 LaneInfo(/lane_info)로도
     발행(디버그/rosbag 용; 현재 런타임 구독자는 없음).
  2) 제어결정(PID): offset 횡오차 PID 를 돌려 조향/스로틀을 계산하고
     Control(/control)로 발행. 하드웨어는 안 만진다(그건 control_node).

왜 인지(lane_detection)와 이 노드를 나누고, 예전의 lane_follow(제어)를 여기로
합쳤나:
  - 인지 = "지금 프레임에 선이 어디 있나"(기하). 그것만 lane_detection 이 한다.
  - 판단+제어결정 = "그걸 얼마나 믿고, 얼마나 꺾을까". 시간적 맥락이 필요한 이
    둘을 한 노드에 모아 한 콜백에서 처리한다.
  - 이렇게 하면 '프레임 도착 → 즉시 조향명령'이 되어(이벤트구동) 예전 lane_follow
    의 고정 20Hz 타이머가 만들던 최대 ~50ms 위상지연과 /lane_info 홉이 사라진다.

제어 방식(offset 전용):
  - 2026-07-06: heading(진행방향 기울기)은 이 셋업에서 주행 중 신뢰불가로 판명
    (종횡비 640x480->320x160 세로 1.5배 압축으로 기울기 증폭 + 점선 중앙선/비대칭
    검출 + ROI 외삽으로 직진에서도 heading 이 ±0.5 스파이크). 그래서 heading 기반
    곡선제어(선행조향/curve_bias)를 전부 걷어내고 offset 만으로 제어한다.
  - 직진: 순수 offset=0 중앙추종(PID). 곡선: offset 이 커지면(바깥 밀림) kp 를
    올리고(반응형) 감속해 라인을 유지. 곡선 감지도 |offset| 기반.

안전(페일세이프):
  - 이벤트구동이라 프레임이 끊기면 발행이 멈춘다. "명령이 끊기면 중립+정지"는
    control_node 의 stale 워치독이 담당한다.
  - 차선 신뢰도가 낮으면(lane lost) 스로틀을 0 으로 램프다운한다.
  - cruise_throttle 기본 0.0: 첫 기동엔 조향만 계산하고 차는 안 움직인다.
"""

import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import yaml

from interface.msg import Control, DetectionArray, LaneDetection, LaneInfo
from std_msgs.msg import Bool, Int32


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def smoothstep(x, lo, hi):
    """lo~hi 구간을 0~1 로 S자 보간(양끝 기울기 0 -> 경계에서 게인이 부드럽게).
    x<=lo -> 0, x>=hi -> 1. 하드 분기 대신 연속 블렌딩용."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


class Judgment:
    """판단(Decision): 인지(차선/YOLO/ArUco)를 시간필터·신뢰·중재해 '목표'를 만든다.
    상태(필터값·인지 플래그)를 소유하고 파라미터/시계는 노드에서 읽는다. 제어(PID)는
    전혀 모른다 — offset 안정화 + 정지/감속 판정만 한다. (동작은 기존과 동일; 코드 조직만)"""

    def __init__(self, node):
        self.node = node
        self.ema_alpha = node.ema_alpha            # 노드가 읽고 검증한 값 그대로
        self.offset_filtered = 0.0                 # 미검출 시 마지막 값 유지
        # YOLO/ArUco 연동 상태(라벨/임계값 등은 콜백에서 매번 param 재읽기 → 라이브 튜닝)
        self.yolo_enabled = bool(node.get_parameter('yolo_enabled').value)
        self.yolo_stop = False
        self.yolo_slow = False
        self.last_yolo_time = None
        self.aruco_enabled = bool(node.get_parameter('aruco_enabled').value)
        self.aruco_stop = False
        self.last_aruco_time = None
        # 신호등 상태 래치: 빨간불→정지(True), 초록불→출발(False). 초기값 출발.
        # (순간 게이트가 아니라 '상태' — 초록불을 봐야 출발, 그전엔 계속 정지)
        self.traffic_light_enabled = bool(node.get_parameter('traffic_light_enabled').value)
        self.traffic_stop = False
        # 출발 게이트: 런치 직후엔 throttle 0 으로 정지해 있다가, 초록불을 '확정'한 순간
        # 해제(래치)해 출발한다. 해제 후엔 다시 잠기지 않는다(이후엔 red/green 래치가 관장).
        # 확정 = 강한 초록(conf≥green_start_confidence)이 green_start_frames 연속 — 오검출
        # 1~2프레임에 안 풀리게 디바운스한다.
        self.started = False
        self._green_run = 0          # 강한 초록 연속 프레임 카운터(출발 게이트 디바운스)
        self._last_green_conf = 0.0  # 최근 프레임 초록 최고 conf(로그 가시화용)
        # 방향표지 바이어스 상태(Y자 갈래 선택): 처음 근접(near-gate 첫 통과)한 순간에
        # 방향을 래치하고, 그 시각부터 sign_bias_hold_sec 동안 고정 bias 를 준다(표지판이
        # 프레임 밖으로 나가도 유지 = 일회성 회전). 창이 끝나면 해제하고, 표지판이 화면에서
        # 사라질 때까지(_sign_refractory) 재래치를 막아 같은 표지판에 반복 반응하지 않는다.
        self.sign_bias_enabled = bool(node.get_parameter('sign_bias_enabled').value)
        self.sign_dir = 0
        self.first_sign_near_time = None   # 고정창 시작 시각(None=미래치)
        self._sign_refractory = False      # 창 소비 후 표지판이 아직 안 사라짐 -> 재래치 금지

    # --- 시간필터 ---
    def filter_offset(self, raw_offset, detected, single_line):
        """검출 시 EMA 저역통과, 미검출 시 마지막 필터값 유지.
        단독선일 때는 offset 을 축소·클램프해 신뢰도를 낮춘다(과대/요동 억제)."""
        gp = self.node.get_parameter
        if single_line:
            scale = float(gp('single_line_offset_scale').value)
            limit = float(gp('single_line_offset_limit').value)
            raw_offset = max(-limit, min(limit, raw_offset * scale))
        if detected:
            self.offset_filtered = (
                self.ema_alpha * raw_offset
                + (1.0 - self.ema_alpha) * self.offset_filtered
            )
        # 미검출: self.offset_filtered 를 직전 값 그대로 유지
        return float(max(-1.0, min(1.0, self.offset_filtered)))

    # --- 인지 플래그 갱신(구독 콜백에서 호출) ---
    def on_yolo(self, msg):
        """YOLO 검출을 받아 정지/감속 플래그를 갱신. 임계값·라벨은 매번 param 재읽기."""
        self.last_yolo_time = self.node.get_clock().now()
        gp = self.node.get_parameter
        self.yolo_enabled = bool(gp('yolo_enabled').value)
        if not self.yolo_enabled:
            self.yolo_stop = False
            self.yolo_slow = False
            return
        min_conf = float(gp('yolo_min_confidence').value)
        min_area = float(gp('yolo_min_box_area_ratio').value)
        stop_labels = set(gp('yolo_stop_labels').value)
        slow_labels = set(gp('yolo_slow_labels').value)
        self.traffic_light_enabled = bool(gp('traffic_light_enabled').value)
        red_label = str(gp('red_label').value)
        green_label = str(gp('green_label').value)
        img_area = float(max(1, int(msg.image_width) * int(msg.image_height)))
        stop = False
        slow = False
        saw_red = False
        saw_green = False
        green_conf = 0.0
        for d in msg.detections:
            if float(d.confidence) < min_conf:
                continue
            # 신호등(red/green)은 '상태'라 거리 무관하게 판정 — 근접(면적) 게이트 미적용.
            # (초록불이 작게 잡혀도 출발이 풀려야 하므로 면적 게이트에 걸리면 안 됨.)
            if d.label == red_label:
                saw_red = True
            elif d.label == green_label:
                saw_green = True
                green_conf = max(green_conf, float(d.confidence))
            # 일반 정지/감속(사람 등)은 근접(면적) 게이트 적용 — 멀리서 조기반응 방지.
            if (float(d.width) * float(d.height)) / img_area < min_area:
                continue
            if d.label in stop_labels:
                stop = True
            elif d.label in slow_labels:
                slow = True
        self.yolo_stop = stop
        self.yolo_slow = slow
        # 신호등 래치: 빨간불 보이면 정지, 초록불 보이면 출발, 둘 다 안 보이면 마지막 상태
        # 유지. red 가 green 보다 우선(애매하면 안전하게 정지).
        self._last_green_conf = green_conf
        if self.traffic_light_enabled:
            if saw_red:
                self.traffic_stop = True
                self._green_run = 0
            elif saw_green:
                self.traffic_stop = False
                # 출발 게이트 해제는 엄격하게: 강한 초록(conf≥green_start_confidence)이
                # green_start_frames 프레임 연속 잡혀야 확정. 오검출 1~2프레임엔 안 풀린다.
                if green_conf >= float(gp('green_start_confidence').value):
                    self._green_run += 1
                else:
                    self._green_run = 0
                if (not self.started
                        and self._green_run >= int(gp('green_start_frames').value)):
                    self.started = True   # 초록불 확정 -> 출발 게이트 영구 해제
                    self.node.get_logger().info(
                        '출발 게이트 해제: 초록불 확정(conf=%.2f, %d프레임 연속)'
                        % (green_conf, self._green_run))
                    if bool(gp('yolo_power_gate').value):
                        self.node.set_yolo_active(False, '초록불 출발')   # 출발과 동시에 YOLO off
            else:
                self._green_run = 0   # 초록 끊기면 카운터 리셋(연속 요구)

        # 방향표지 near-gate + 일회성 래치: left/right 표지판이 '가까우면'(박스 높이비 ≥ 문턱)
        # '처음 근접'한 그 순간에 방향을 래치해 고정창(sign_bias_hold_sec)을 시작한다.
        # 멀면 무시(미리 안 꺾게). 창 진행 중/소비 후엔 재래치하지 않는다(lane_bias 참고).
        self.sign_bias_enabled = bool(gp('sign_bias_enabled').value)
        if self.sign_bias_enabled:
            near_ratio = float(gp('sign_near_ratio').value)
            left_label = str(gp('left_sign_label').value)
            right_label = str(gp('right_sign_label').value)
            img_h = float(max(1, int(msg.image_height)))
            near_dir = 0   # 이번 프레임에서 near 로 본 방향(0=없음, +1=left, -1=right)
            for d in msg.detections:
                if float(d.confidence) < min_conf:
                    continue
                if (float(d.height) / img_h) < near_ratio:   # near-gate: 가까울 때만
                    continue
                if d.label == left_label:
                    near_dir = 1
                elif d.label == right_label:
                    near_dir = -1
            if near_dir != 0:
                # 처음 근접한 순간에만 래치(고정창 시작). 이미 창이 진행 중이거나,
                # 직전 창을 소비한 뒤 표지판이 아직 안 사라졌으면(refractory) 재래치 안 함.
                if self.first_sign_near_time is None and not self._sign_refractory:
                    self.sign_dir = near_dir
                    self.first_sign_near_time = self.node.get_clock().now()
                    self.node.note_sign_latch()   # (yolo_power_gate=true 때만 유효, 지금은 무해)
            else:
                # near 표지판이 화면에서 사라짐 -> 다음 표지판을 위해 재무장.
                self._sign_refractory = False

    def on_aruco(self, msg):
        """aruco_node 정지신호(/aruco_stop, 이미 디바운스됨)를 받아 플래그 갱신."""
        self.last_aruco_time = self.node.get_clock().now()
        self.aruco_enabled = bool(self.node.get_parameter('aruco_enabled').value)
        self.aruco_stop = bool(msg.data) if self.aruco_enabled else False

    # --- 인지 중재: 정지/감속 판정 ---
    def perception_stop_slow(self):
        """YOLO/ArUco 인지를 종합해 (stop, slow) 판정. 각 소스는 stale(노드 사망 등)이면
        무시한다(죽은 인지가 브레이크를 영구히 잡지 않게)."""
        now = self.node.get_clock().now()
        gp = self.node.get_parameter
        stop = False
        slow = False
        # 출발 게이트: 첫 초록불을 아직 못 봤으면 무조건 정지(YOLO 첫 메시지 도착 전에도
        # 적용 — 런치 직후 오출발 방지). YOLO+신호등이 켜져 있을 때만 — 꺼져 있으면
        # (예: 초록불 없는 벤치/링 테스트) 게이트 미적용해 즉시 주행한다.
        if (bool(gp('require_green_start').value) and self.yolo_enabled
                and self.traffic_light_enabled and not self.started):
            stop = True
        if self.yolo_enabled and self.last_yolo_time is not None:
            age = (now - self.last_yolo_time).nanoseconds * 1e-9
            if age <= float(gp('yolo_stop_timeout_sec').value):
                if self.yolo_stop:
                    stop = True
                elif self.yolo_slow:
                    slow = True
                # 신호등 정지 래치(빨간불 상태)도 반영 — 정지가 우선
                if self.traffic_light_enabled and self.traffic_stop:
                    stop = True
        if self.aruco_enabled and self.last_aruco_time is not None:
            age = (now - self.last_aruco_time).nanoseconds * 1e-9
            if age <= float(gp('aruco_stop_timeout_sec').value):
                if self.aruco_stop:
                    stop = True
        return stop, slow

    def lane_bias(self):
        """방향표지 바이어스(Y자 갈래 선택). '처음 근접'한 순간부터 sign_bias_hold_sec 동안
        방향을 고정 유지한다(표지판이 프레임 밖으로 나가도 유지 = 일회성 회전). 창이 끝나면
        해제하고, 표지판이 화면에서 사라질 때까지 재래치를 막는다(_sign_refractory).
        반환: 목표 offset 에 줄 치우침(0=중앙유지)."""
        gp = self.node.get_parameter
        if (not self.sign_bias_enabled or self.sign_dir == 0
                or self.first_sign_near_time is None):
            return 0.0
        age = (self.node.get_clock().now() - self.first_sign_near_time).nanoseconds * 1e-9
        if age > float(gp('sign_bias_hold_sec').value):
            self.sign_dir = 0                 # 고정창 종료 -> bias 해제
            self.first_sign_near_time = None
            self._sign_refractory = True      # 표지판이 사라지기 전까지 재래치 금지
            return 0.0
        return float(self.sign_dir) * float(gp('sign_bias_magnitude').value)

    def debug_tag(self):
        """디버그 로그 접미사(원본과 동일 포맷)."""
        tag = ''
        if self.yolo_enabled:
            tag += ' yolo=' + ('STOP' if self.yolo_stop else 'slow' if self.yolo_slow else '-')
            if self.traffic_light_enabled:
                tag += ' signal=' + ('RED(stop)' if self.traffic_stop else 'GREEN(go)')
                # 출발 게이트 상태 가시화: 아직 대기면 연속카운트·초록 conf 를 보여준다
                # (오검출 원인 진단용). started 후엔 표시 안 함.
                if not self.started:
                    tag += ' START=HELD(g%d,c%.2f)' % (self._green_run, self._last_green_conf)
        if self.aruco_enabled and self.aruco_stop:
            tag += ' aruco=STOP'
        if self.sign_bias_enabled and self.sign_dir != 0:
            tag += ' sign=' + ('LEFT' if self.sign_dir > 0 else 'RIGHT')
        return tag


class RampEntry:
    """진입로(노란 램프) 진입 판단.

    상태: WAIT -> APPROACH -> RAMP  (되돌아가지 않는다; 순서 게이팅)
      WAIT     : 흰 차선 주행. 출발 후 arm_delay_sec 동안 움직인 뒤에야 감지를 무장한다
                 (출발선 근처 오탐 차단). 조향/속도 무영향.
      APPROACH : 원거리 밴드에서 '왼쪽 앞에 노랑' 단서를 봤다. 아직 흰 차선을 따라가되
                 미리 감속한다. 램프 곡률이 본선 코너보다 조이므로 반응형 감속으론 늦다.
      RAMP     : 근거리에서 노란 선이 안정적으로 잡혔다(커밋). 이제 노란 램프 중심을
                 목표로 삼는다. 흰 차선은 더 안 본다.

    커밋 조건을 '노랑이 처음 보일 때'가 아니라 '근거리에서 안정적으로 잡힐 때'로 둔 이유:
    전자는 노란 선이 아직 멀어 근거리 목표를 못 만드는데도 목표를 갈아끼우는 셈이라,
    하필 꺾어야 할 지점에서 confidence 0 -> 페일세이프 정지가 난다. 후자면 그 순간
    노란 경계는 흰 경계가 있던 자리에 있어 offset 이 크게 안 튄다(기하적 연속성).

    RAMP 중 노란 인지가 무너지면 정지가 아니라 흰 차선으로 '후퇴'한다(degraded).
    노란색은 조명에 민감하므로, 인지 실패가 곧 이탈이 되지 않게 한다."""

    def __init__(self, node):
        self.node = node
        # 시작 상태: 기본은 WAIT(흰차선 주행→노랑 커밋시 RAMP). 링 위에서 출발할 땐
        # ramp_start_state:=RAMP 로 부팅해 WAIT 를 건너뛰고 첫 프레임부터 노란 차선 추종.
        start = str(node.get_parameter('ramp_start_state').value).upper()
        self.state = 'RAMP' if start == 'RAMP' else 'WAIT'
        self.moving_sec = 0.0        # 실제로 굴러간 시간(무장 타이머 기준)
        # 링 출발(RAMP 부팅)이면 arm_delay·커밋 가드를 건너뛴다(이미 노란 구간 위이므로).
        self.armed = (self.state == 'RAMP')
        if self.state == 'RAMP':
            node.get_logger().info(
                'ramp: RAMP 상태로 부팅(링 출발) — 흰차선 WAIT 단계 건너뜀')
        self.cue_frames = 0          # APPROACH 진입 디바운스
        self.commit_frames = 0       # RAMP 커밋 디바운스
        self.degrade_frames = 0      # 후퇴 디바운스
        self.recover_frames = 0      # 복귀 디바운스
        self.degraded = False        # RAMP 이지만 노란 인지가 무너져 흰 차선으로 후퇴 중
        self.latest = None           # 최신 /yellow/lane (LaneDetection)
        self.latest_time = None
        self.heading = 0.0           # 조감도 heading (RAMP 추종 중에만 의미 있음)
        self.approach_sec = 0.0      # APPROACH 체류 시간(분기 놓침 감지용)

    def on_ramp(self, msg):
        self.latest = msg
        self.latest_time = self.node.get_clock().now()

    def fresh(self):
        """램프 인지가 살아있나. 노드가 죽었거나 프레임이 끊기면 없는 셈 친다."""
        if self.latest is None or self.latest_time is None:
            return None
        age = (self.node.get_clock().now() - self.latest_time).nanoseconds * 1e-9
        if age > float(self.node.get_parameter('ramp_stale_timeout_sec').value):
            return None
        return self.latest

    def step(self, dt, throttle_cmd):
        """한 프레임 진행. 반환: (use_ramp, raw_offset, confidence, throttle_scale, switched).

        규칙 (2026-07-10, 단순화):
          WAIT : 흰 차선 주행. 노란 차선이 commit_frames 연속 잡히면 곧장 RAMP.
          RAMP : 노란 차선만 따라간다. **흰 차선으로 절대 후퇴하지 않는다.**
                 노란 인지가 무너지면 신뢰도가 그대로 내려가고, interpret 의 기존
                 페일세이프가 스로틀을 0 으로 내린다 — 정지가 오조향보다 낫다.
                 (램프에 반쯤 올라탄 상태에서 흰 차선 명령을 받으면 엉뚱한 데로 간다.)
        det 은 /yellow/lane 의 LaneDetection — lane_node 를 노란 마스크로 한 번 더 돌린
        결과다. 단독선일 때 학습한 차선폭으로 반대편을 추정하는 로직이 그대로 산다."""
        gp = self.node.get_parameter
        none = (False, 0.0, 0.0, 1.0, False)
        self.heading = 0.0            # offset 전용 (lane_node 규약)
        if not bool(gp('ramp_enabled').value):
            return none

        if throttle_cmd > 0.0:
            self.moving_sec += dt
        if not self.armed and self.moving_sec >= float(gp('arm_delay_sec').value):
            self.armed = True

        det = self.fresh()
        scale = float(gp('ramp_throttle_scale').value)

        if det is None:
            if self.state == 'RAMP':
                # 인지 노드가 죽었다 -> 신뢰도 0 을 내려보내 페일세이프가 세우게 한다.
                return (True, 0.0, 0.0, scale, False)
            return none

        if self.state == 'WAIT':
            if self.armed and self._yellow_seen(det):
                self.commit_frames += 1
                if self.commit_frames >= int(gp('commit_frames').value):
                    self.state = 'RAMP'
                    self.node.get_logger().info(
                        f'ramp: WAIT -> RAMP (노란 차선 커밋, off={det.raw_offset:+.3f} '
                        f'conf={det.confidence:.2f})')
                    return (True, float(det.raw_offset), float(det.confidence), scale, True)
            else:
                self.commit_frames = 0
            return none

        # --- RAMP: 후퇴 없음. 노란 차선만 본다. ---
        return (True, float(det.raw_offset), float(det.confidence), scale, False)

    def _yellow_seen(self, det):
        """커밋해도 되는가. 커밋은 되돌릴 수 없으므로 엄격하게 본다.
        진짜 램프 앞에서는 양쪽 노란 차선이 안정적으로 보인다 — 스치는 단독선이 아니라."""
        gp = self.node.get_parameter
        if float(det.confidence) < float(gp('commit_confidence').value):
            return False
        if bool(gp('commit_require_both').value):
            return bool(det.left_detected) and bool(det.right_detected)
        return bool(det.left_detected) or bool(det.right_detected)

    def debug_tag(self):
        if not bool(self.node.get_parameter('ramp_enabled').value):
            return ''
        tag = f' ramp={self.state}'
        if self.state == 'WAIT' and not self.armed:
            tag += '(disarmed)'
        if self.degraded:
            tag += '(degraded)'
        return tag


class Controller:
    """제어(Control law): 판단이 준 목표(offset + 정지/감속)를 PID 로 추종해 조향/스로틀
    명령을 계산한다. '무엇을/왜' 는 모른다 — 목표 추종만. 상태(적분·이전값·명령) 소유.
    (run_control 로직을 그대로 옮긴 것; 동작 동일. 발행/로그는 노드가 한다.)"""

    def __init__(self, node, steer_trim):
        self.node = node
        self.steer_trim = steer_trim
        self.prev_offset_for_d = 0.0
        self.prev_time = None                      # 미분/슬루 dt 계산용 (콜백 간 시간)
        self.throttle_cmd = 0.0
        self.steer_cmd_filtered = steer_trim       # EMA-smoothed 조향 출력
        self.was_low_conf = False                  # 직전 프레임 신뢰도 미달?
        self.integral = 0.0                        # offset 오차 적분(I 항)

    def reseed(self, offset):
        """목표의 출처가 바뀔 때(흰 차선 <-> 노란 램프) 호출. offset 이 다른 기준으로
        점프하므로 이 프레임의 미분은 의미가 없다 -> 직전값을 새 offset 으로 맞춰
        d(offset)/dt = 0 으로 만들고 적분을 비운다. (차선 재획득 시의 처리와 같은 이유)"""
        self.prev_offset_for_d = offset
        self.integral = 0.0

    def step(self, offset, confidence, stop, slow, bias=0.0, throttle_scale_extra=1.0,
             heading=0.0):
        """offset PID + 스로틀 목표(정지/감속 반영) + 슬루 -> (steering, throttle, diag).
        bias: 방향표지 바이어스(목표 offset 치우침; 0=중앙유지). P/I 오차에만 반영,
        미분(D)은 원 offset 기준(setpoint 변화에 미분 튐 방지).
        throttle_scale_extra: 구간별 추가 감속 배율(진입 접근/램프 주행). 1.0 = 무영향."""
        gp = self.node.get_parameter
        now = self.node.get_clock().now()
        if self.prev_time is None:
            dt = 1.0 / 30.0   # 첫 프레임 가정치(카메라 ~30fps)
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 30.0
        self.prev_time = now

        kp_straight = float(gp('kp_offset').value)
        kd = float(gp('kd_offset').value)
        ki = float(gp('ki_offset').value)
        i_limit = float(gp('i_limit').value)
        kp_curve = float(gp('kp_offset_curve').value)
        sched_off_lo = float(gp('sched_offset_lo').value)
        sched_off_hi = float(gp('sched_offset_hi').value)
        steer_limit = float(gp('steer_limit').value)
        steer_sign = float(gp('steer_sign').value)
        d_limit = float(gp('d_offset_limit').value)
        alpha = clip(float(gp('steer_smooth_alpha').value), 0.05, 1.0)
        min_conf = float(gp('min_confidence').value)

        low_conf = confidence < min_conf

        # 게인 스케줄링: |offset| 로 직진<->곡선 블렌딩(반응형).
        w = smoothstep(abs(offset), sched_off_lo, sched_off_hi)
        kp = lerp(kp_straight, kp_curve, w)

        error = offset - bias  # 목표=차선중앙(0); 방향표지 bias 로 좌/우 치우침

        # 미분(클램프); 검출 복귀 프레임엔 리셋해 슬램 방지.
        if self.was_low_conf and not low_conf:
            self.prev_offset_for_d = offset
        d_offset = clip((offset - self.prev_offset_for_d) / dt, -d_limit, d_limit)
        self.prev_offset_for_d = offset
        self.was_low_conf = low_conf

        # 적분(anti-windup): 추종 중 & 전진 중일 때만 누적; 아니면 리셋.
        if low_conf or self.throttle_cmd <= 0.0:
            self.integral = 0.0
        elif ki > 0.0:
            self.integral += error * dt
            self.integral = clip(self.integral, -i_limit / ki, i_limit / ki)
        i_term = clip(ki * self.integral, -i_limit, i_limit)

        # PID 후 EMA 저역통과로 부드러운 출력.
        # heading 항은 진입(RAMP)에서만 0 이 아니다. 그 구간에선 횡오차가 ≈0 이라
        # "얼마나 틀어져 있나"가 유일한 조향 신호다. 부호 규약은 offset 과 동일(음수=좌).
        h_lim = float(gp('heading_limit_rad').value)
        h_term = float(gp('kh_ramp').value) * clip(heading, -h_lim, h_lim)
        logical = kp * error + i_term + kd * d_offset + h_term
        logical = clip(logical, -steer_limit, steer_limit)
        steering_raw = clip(self.steer_trim + steer_sign * logical, -1.0, 1.0)
        self.steer_cmd_filtered = (
            alpha * steering_raw + (1.0 - alpha) * self.steer_cmd_filtered
        )
        steering = clip(self.steer_cmd_filtered, -1.0, 1.0)

        # 스로틀: lane lost 아니면 cruise; 곡선(w↑) 감속; 인지 정지/감속; 슬루 제한.
        cruise = float(gp('cruise_throttle').value)
        max_throttle = float(gp('max_throttle').value)
        slew = float(gp('throttle_slew_per_sec').value)
        curve_thr_scale = float(gp('curve_throttle_scale').value)
        throttle_scale = lerp(1.0, curve_thr_scale, w) * clip(throttle_scale_extra, 0.0, 1.0)
        target_throttle = 0.0 if low_conf else clip(cruise * throttle_scale, 0.0, max_throttle)
        # 판단이 준 정지/감속 반영 (원래 apply_perception_gate 의 적용부와 동일).
        if stop:
            target_throttle = 0.0
        elif slow:
            target_throttle = target_throttle * float(gp('yolo_slow_scale').value)

        step = slew * dt
        if target_throttle > self.throttle_cmd:
            self.throttle_cmd = min(self.throttle_cmd + step, target_throttle)
        else:
            self.throttle_cmd = max(self.throttle_cmd - step, target_throttle)

        return steering, self.throttle_cmd, {'i': i_term, 'd': d_offset, 'w': w, 'kp': kp}


class InterpretNode(Node):
    def __init__(self):
        super().__init__('interpret_node')

        # --- Topics / IO ----------------------------------------------------
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('lane_topic', '/lane_info')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 판단: offset 시간필터 -----------------------------------------
        # offset 저역통과 필터 계수(0~1, 클수록 민감/덜 평활).
        self.declare_parameter('ema_alpha', 0.4)
        # 단독선(한쪽만 검출)일 때 offset 축소/클램프 실험용 훅.
        # 2026-07-04 트랙 테스트: 축소(scale=0.5)하면 오히려 위치보정이 죽어 직진을
        # 못했다 → 기본값 1.0(비활성)으로 되돌림. 두 선 중앙잡기가 정상 동작이므로
        # 단독선 offset도 그대로 신뢰한다. 훅은 남겨두되 함부로 낮추지 말 것.
        self.declare_parameter('single_line_offset_scale', 1.0)
        self.declare_parameter('single_line_offset_limit', 1.0)

        # --- 제어결정: 횡오차 PID (offset 전용) ----------------------------
        # logical steer: 양수 = 물리적으로 우회전(LaneInfo 규약: +offset 이면 차선
        # 중심이 이미지 오른쪽 -> 차가 왼쪽으로 치우침 -> 우회전). steer_sign 이
        # 이를 서보 퍼센트 극성으로 매핑(-1.0 은 이전 차량 인계값, 트랙에서 검증).
        # 2026-07-06 주행 측정으로 확정한 게인. 이벤트구동 전환 후 남은 직진
        # 뱀주행(리밋사이클, ~0.5Hz)을 실측 튜닝으로 잡음:
        #   원본(kp0.45/kd0.12/ki0.2) offset std 0.110 -> 최종 std 0.060(-45%).
        self.declare_parameter('kp_offset', 0.25)   # 0.45->0.25 루프게인 축소(리밋사이클)
        self.declare_parameter('kd_offset', 0.16)   # 0.12->0.16 감쇠 강화(offset 깨끗해 여유)
        # 적분: 원래 정상상태 offset 제거용이나, 저속 위치루프에선 위상지연을 키워
        # 뱀주행을 되살린다(실측: ki=0.2/0.05 모두 std~0.11, ki=0 은 0.06). 그래서
        # 기본 0 으로 비활성. 정상상태 편향은 steer_trim 이 담당(측정 잔차 ±0.03).
        # 긴 직선에서 드리프트가 문제되면 i_limit 을 크게 낮춰(≤0.1) 소량만 재도입.
        self.declare_parameter('ki_offset', 0.0)
        self.declare_parameter('i_limit', 0.3)       # clamp on |ki*integral|
        self.declare_parameter('steer_limit', 0.7)   # max |logical steer|
        self.declare_parameter('steer_sign', -1.0)
        # steer_trim(직진/중립 조향값)은 vehicle_config 에서 읽는다.
        # --- 평활(직진 twitch 억제) ---
        # d_offset: 검출이 잠깐 끊겼다 복귀할 때 offset 이 튀어 미분이 폭발 -> 클램프.
        # steer EMA 로 최종 명령을 저역통과.
        self.declare_parameter('d_offset_limit', 2.0)     # clamp |d(offset)/dt|
        self.declare_parameter('steer_smooth_alpha', 0.30)  # 1.0 = no smoothing
        # --- 게인 스케줄링(직진<->곡선 연속 블렌딩, |offset| 기준) ---
        # w = smoothstep(|offset|, lo, hi): 0=중앙(직진) ~ 1=크게 벌어짐(곡선/이탈).
        # w 로 kp 를 직진값<->곡선값 사이에서 연속 보간. 하드 분기의 경계 튐을 피하려
        # S자 블렌딩. offset 이 커지면(곡선에서 바깥 밀림) kp 를 올려 강하게 복구.
        # 직진 뱀주행(|offset|~0.1)은 lo(0.3) 아래라 안 건드림 -> 직진 튜닝 유지.
        # heading 기반 곡선감지는 신뢰불가로 제거, offset(항상 유효)만으로 감지.
        self.declare_parameter('kp_offset_curve', 0.45)  # 곡선 kp(offset 교정 강화)
        self.declare_parameter('sched_offset_lo', 0.3)   # 이 |offset| 부터 복구게인 시작
        self.declare_parameter('sched_offset_hi', 0.6)   # 이 |offset| 에서 완전 곡선게인

        # --- 스로틀 ---------------------------------------------------------
        self.declare_parameter('cruise_throttle', 0.0)  # 0 => 조향부터 검증
        self.declare_parameter('max_throttle', 0.30)
        self.declare_parameter('throttle_slew_per_sec', 0.6)  # ramp rate
        # 곡선 감속: 조향 스케줄과 같은 w 로 throttle 을 줄인다("코너에서 브레이크").
        # 2026-07-06: kp0.45 로 강하게 걸어도 급곡선은 속도가 빠르면 못 버티고
        # 가장자리를 스침 -> 감속해야 라인 유지.
        # throttle = cruise × lerp(1.0, curve_throttle_scale, w). w=1 에서 이 비율로.
        self.declare_parameter('curve_throttle_scale', 0.9)

        # --- 진입로(노란 램프) 진입 판단 ------------------------------------
        # 마스터 스위치. 기본 False -> ramp_detection 이 떠 있어도 주행에 영향 없다.
        # 대시보드 Ramp 패널로 /yellow/lane 을 확인한 뒤에 켠다:
        #   ros2 param set /interpret_node ramp_enabled true
        self.declare_parameter('ramp_enabled', False)
        # 노란 차선 인지 = lane_node 를 /opencv/image/yellow 로 한 번 더 돌린 것.
        # 메시지도 LaneDetection 그대로다(별도 msg/알고리즘 없음).
        self.declare_parameter('ramp_detection_topic', '/yellow/lane')
        # 인지가 이 시간 넘게 끊기면 없는 셈(노드 사망이 브레이크를 잡지 않게).
        self.declare_parameter('ramp_stale_timeout_sec', 0.5)
        # 무장: 실제로 굴러간 시간이 이만큼 지나야 감지 시작(출발선 근처 오탐 차단).
        # 램프 바로 앞에서 출발시킬 땐 짧게 줄일 것(안 그러면 무장 전에 분기를 지나친다).
        self.declare_parameter('arm_delay_sec', 3.0)
        # 시작 상태 선택. 'WAIT'(기본): 흰차선부터 시작해 노랑 커밋시 RAMP 전환.
        # 'RAMP': 링 위에서 출발 — WAIT/arm_delay/커밋 가드를 건너뛰고 노란차선 즉시 추종.
        self.declare_parameter('ramp_start_state', 'WAIT')

        # 커밋: 노란 차선이 이만큼 잡히면 곧장 RAMP. 흰 차선으로 후퇴는 없다.
        # 커밋은 되돌릴 수 없으므로 '스치는 노란 선'과 '진짜 램프'를 신뢰도로 가른다.
        # 2026-07-10 실주행으로 문턱을 훑어 확정:
        #   conf 0.6  -> t=7.0s 커밋, 한 프레임 뒤 소실 -> 즉시 정지 (너무 늦게 커밋)
        #   conf 0.5  -> t=10.0s 커밋, 2.5초 추종 후 정지
        #   conf 0.35 -> t=8.5s 커밋, 3.5초 추종, **램프 안 진입 성공**  <= 채택
        #   conf 0.3/3프레임 -> 본선 주행 중 트랙 저편 노란 링(conf 0.25~0.33)에 오커밋
        # 낮출수록 더 일찍 커밋해 더 오래 따라간다. 0.35 가 오커밋(0.33)과 진짜 램프
        # 사이의 좁은 틈이다. 유지 프레임(8)이 스치는 선을 한 번 더 거른다.
        self.declare_parameter('commit_confidence', 0.35)
        self.declare_parameter('commit_frames', 8)      # 30Hz 기준 약 0.27초 유지
        # True 면 양쪽 노란 차선이 모두 보일 때만 커밋. 주행 중엔 너무 엄격해 기본 False.
        self.declare_parameter('commit_require_both', False)
        self.declare_parameter('ramp_throttle_scale', 0.6)     # 램프 곡률이 조여 감속 유지

        # 진입 조향에 쓰는 heading 게인. RAMP 상태에서만 적용된다.
        # 2026-07-09: 램프 진입 시 횡오차는 이미 ≈0 인데 경계선이 지면 기준 −14° 기울어
        # 있다. 즉 "옆으로 얼마나 벗어났나"가 아니라 "얼마나 틀어져 있나"가 전부다.
        # 조감도(IPM)에서 잰 heading 이라 지면 실각도 — 원본 화면 heading(07-06 제거,
        # 원근 왜곡으로 ±0.5rad 스파이크)과는 완전히 다른 물건이다.
        # 흰 차선 주행은 여전히 offset 전용이라 검증된 튜닝이 안 깨진다.
        # 2026-07-10 실주행: kh=1.0 은 진입은 되는데 '너무 꺾여' 램프 안에서 탈선했다.
        # heading 이 커질수록 더 꺾고, 조향을 풀어줄 신호가 없었다(중앙잡기 전환 부재).
        self.declare_parameter('kh_ramp', 0.5)
        # heading 항의 기여 상한(rad). 급커브에서 heading 이 커져도 조향이 폭주하지 않게.
        self.declare_parameter('heading_limit_rad', 0.30)

        # --- 페일세이프 -----------------------------------------------------
        self.declare_parameter('min_confidence', 0.2)   # 미만 -> lost 취급
        self.declare_parameter('debug_log', False)

        # --- YOLO 검출 연동 -------------------------------------------------
        # yolo_enabled = /yolo/detections 를 판단에 쓸지 마스터 스위치. True 면 아래가
        # 모두 동작: 신호등 정지/출발(traffic_light) · 방향표지 바이어스(sign_bias) ·
        # 일반 정지/감속(yolo_stop_labels, 기본 빈 목록). 모델 검증 완료라 기본 True.
        # 검출을 '표시만' 하고 제어에서 완전히 떼려면:
        #   ros2 param set /interpret_node yolo_enabled false
        # (yolo:=true 로 노드는 띄웠지만 제어 연동만 끄고 싶을 때)
        self.declare_parameter('yolo_enabled', True)
        self.declare_parameter('yolo_detections_topic', '/yolo/detections')
        self.declare_parameter('yolo_min_confidence', 0.5)   # 미만 검출은 무시
        # 박스가 이미지의 이 비율 이상일 때만 반응(근접 프록시). 멀리 있는 작은 검출 무시.
        self.declare_parameter('yolo_min_box_area_ratio', 0.03)
        # 일반 정지/감속 클래스 (labels.txt 이름과 정확히 일치). 라이브 튜닝 가능.
        # 신호등(red_light/green_light)은 아래 '신호등 래치'가 상태로 처리하므로 여기선
        # 뺀다(기본 빈 목록). left_sign/right_sign 은 표시만(조향 영역, 미연동).
        # 다른 즉시정지 클래스(사람 등)를 추가하려면 여기에 넣으면 순간 정지로 동작.
        self.declare_parameter('yolo_stop_labels', [''])
        # 감속 전용 클래스는 없음. ['']=사실상 빈 목록(어떤 라벨과도 불일치). 필요시 추가.
        self.declare_parameter('yolo_slow_labels', [''])
        self.declare_parameter('yolo_slow_scale', 0.5)       # 감속 시 throttle 배율
        # 검출 끊김(노드 사망 등) 이 시간 초과면 게이팅 해제 — 죽은 인지가 브레이크를
        # 영구히 잡지 않도록. 실제 모션 페일세이프는 control_node stale 워치독이 담당.
        self.declare_parameter('yolo_stop_timeout_sec', 1.0)

        # --- 신호등 상태 래치 (빨간불 정지 / 초록불 출발) ---
        # 순간 게이트가 아니라 '상태'로 처리: 빨간불(red_label)을 보면 정지로 래치,
        # 초록불(green_label)을 보면 출발로 해제. 둘 다 안 보이면 마지막 상태 유지 →
        # 빨간불이 잠깐 가려져도(각도/가림) 초록불을 볼 때까지 계속 정지(안전).
        # 라벨은 yolo_min_confidence / yolo_min_box_area_ratio 문턱을 공유한다.
        self.declare_parameter('traffic_light_enabled', True)
        # 출발 게이트: True 면 런치 직후 정지해 있다가 첫 초록불을 봐야 출발한다.
        # (yolo_enabled & traffic_light_enabled 가 모두 켜져 있을 때만 유효.)
        # 초록불 없는 곳에서 바로 굴리려면 False 로: ros2 param set /interpret_node require_green_start false
        self.declare_parameter('require_green_start', True)
        # 출발 게이트 해제 조건(오검출 방어): 강한 초록 conf 문턱 + 연속 프레임 수.
        # 오검출로 바로 출발하면 conf 를 올리거나 frames 를 늘린다(로그의 START=HELD 참고).
        self.declare_parameter('green_start_confidence', 0.60)
        # 10 = 오검출(실측 최대 4연속)을 확실히 거른다. 진짜 초록불은 연속 잡히니 여유.
        self.declare_parameter('green_start_frames', 10)
        self.declare_parameter('red_label', 'red_light')
        self.declare_parameter('green_label', 'green_light')
        # YOLO 전원 게이트(자동 on/off로 CPU 절약): 초록불 출발 순간 off, ArUco 마커
        # 인지 시 on. False 면 이 자동 게이팅을 끈다(YOLO 계속 켜둠 — 링/벤치 테스트).
        self.declare_parameter('yolo_power_gate', True)
        self.declare_parameter('yolo_active_topic', '/yolo/active')
        # 초기 YOLO 추론(전원) 상태 = 부팅 시 /yolo/active. 런치 yolo_start:=on|off 가
        # yolo_node.active 와 함께 이 값을 맞춘다(둘이 어긋나면 set_yolo_active 의 dedup 이
        # 첫 wake 를 삼킴). 특정 구간부터 테스트할 때 그 구간에 맞는 on/off 로 시작한다.
        # 기본 True(켠 채 시작 = 출발선 신호등을 본다).
        self.declare_parameter('yolo_start_active', True)
        # 표지판 후 자동 off(transition 7) 초기 무장 여부. 보통 파랑/아루코 wake 때 무장되지만,
        # 표지판 구간부터 곧바로(yolo_start:=on) 테스트할 때 true 로 무장해 7번을 확인한다.
        # 기본 False(출발 phase 를 헛-표지판 off 로부터 보호).
        self.declare_parameter('sign_off_armed_start', False)
        self.declare_parameter('marker_id_topic', '/detected_marker_id')
        self.declare_parameter('yolo_wake_marker_id', -1)  # -1=아무 마커나 재가동, ≥0=해당 ID만
        # 파란 표지판 트리거(/sign/near, bluesign_node 발행)로도 YOLO 를 깨운다.
        # ArUco 마커 대신(또는 함께) 파란 표지판 색이 보이면 갈림길 근처로 보고 켠다.
        # ArUco 마커와 동일하게 yolo_power_gate 아래에서만 작동한다.
        self.declare_parameter('sign_near_topic', '/sign/near')
        # 표지판 인식 후 YOLO 자동 off 지연(초). 방향표지가 처음 래치(near-gate 통과)된
        # 시각부터 이 시간 뒤 YOLO 를 1회 끈다(CPU 회수). 방향은 이미 sign_dir 로 래치돼
        # 있어 조향엔 영향 없다. 빨간도로 ArUco 재가동과 충돌하지 않음(거긴 방향표지 없어
        # 래치가 안 생김). 라이브: ros2 param set /interpret_node sign_yolo_off_delay 5.0
        self.declare_parameter('sign_yolo_off_delay', 5.0)

        # --- ArUco 마커 연동(정지) -----------------------------------------
        # aruco_node 의 /aruco_stop(Bool, 이미 디바운스됨)을 구독해 True 면 정지시킨다.
        # 마커→정지의 '무엇을 정지대상으로 볼지'(target_marker_id) 는 aruco_node 가 판정하고,
        # interpret 은 그 신호를 받아 '그래서 스로틀을 0으로' = 판단의 나머지를 담당한다.
        # 조향엔 관여 안 함(차선추종 유지). 기본 활성(사용자 요청). off 하려면:
        #   ros2 param set /interpret_node aruco_enabled false
        self.declare_parameter('aruco_enabled', True)
        self.declare_parameter('aruco_stop_topic', '/aruco_stop')
        # 정지신호 끊김(노드 사망 등) 이 시간 초과면 게이팅 해제 — 죽은 인지가 브레이크를
        # 영구히 잡지 않도록.
        self.declare_parameter('aruco_stop_timeout_sec', 1.0)

        # --- 방향표지(left/right) 바이어스: Y자 갈림길에서 좌/우 갈래 선택 ---
        # left/right 표지판이 '가까우면'(박스 높이비 ≥ sign_near_ratio) 그 방향으로 목표
        # offset 을 치우쳐(bias) 갈림길에서 그 갈래로 붙는다(순수 조향 편향, 정지 아님).
        # 근접 표지가 sign_bias_hold_sec 동안 안 보이면 0 복귀(갈림길 통과 완료).
        # sign_bias_magnitude(치우침 세기)는 실제 Y 에서 라이브로 튜닝할 것 — 제일 중요.
        # +bias / -bias 가 물리적 좌/우 어느 쪽인지는 실측 후 필요시 부호만 뒤집으면 됨.
        self.declare_parameter('sign_bias_enabled', True)
        self.declare_parameter('sign_bias_magnitude', 0.35)   # 목표 offset 치우침(실측 튜닝)
        self.declare_parameter('sign_near_ratio', 0.25)       # 박스 높이비 문턱(가까움 판정)
        self.declare_parameter('sign_bias_hold_sec', 2.5)     # 결정 후 유지(갈림길 통과 시간)
        self.declare_parameter('left_sign_label', 'left_sign')
        self.declare_parameter('right_sign_label', 'right_sign')

        detection_topic = str(self.get_parameter('detection_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('ema_alpha must be in range (0, 1]')

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.steer_trim = self.load_steer_trim()

        # --- 판단/제어 분리 (동작 동일; 코드 조직만) ---
        # 판단(Judgment): 시간필터 + 인지 정지/감속 판정.  상태(필터값·인지 플래그) 소유.
        # 제어(Controller): 그 목표를 PID 로 추종.        상태(적분·이전값·명령) 소유.
        # 둘 다 파라미터/시계는 이 노드에서 읽는다. 한 콜백에서 이어 실행(이벤트구동 유지).
        self.judgment = Judgment(self)
        self.controller = Controller(self, self.steer_trim)
        # 진입로 진입 판단(순서 게이팅 상태기계). ramp_enabled=False 면 아무것도 안 한다.
        self.ramp = RampEntry(self)
        self.prev_step_time = None   # RampEntry 무장 타이머용 dt

        self.subscription = self.create_subscription(
            LaneDetection,
            detection_topic,
            self.detection_callback,
            10,
        )
        self.lane_pub = self.create_publisher(LaneInfo, lane_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        # YOLO 전원 게이트 발행(/yolo/active). latched(transient_local) 라 yolo_node 가
        # 늦게 떠도 마지막 명령을 받는다. 초록불 출발 시 off, ArUco 마커 인지 시 on.
        gate_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.yolo_active_pub = self.create_publisher(
            Bool, str(self.get_parameter('yolo_active_topic').value), gate_qos)
        # 현재 YOLO 전원 명령 상태(중복 발행 방지). 초기값 = yolo_start_active(런치 yolo_start).
        self._yolo_active_cmd = bool(self.get_parameter('yolo_start_active').value)
        # 표지판 인식 후 자동 off(transition 7). YOLO 가 켜질 때 '무장'되고, 방향표지가
        # 처음 래치되면 카운트다운을 시작해 sign_yolo_off_delay 뒤 1회 off. 초기 무장은
        # sign_off_armed_start(기본 False — 출발 phase 는 초록불→off 가 관장).
        self._sign_off_armed = bool(self.get_parameter('sign_off_armed_start').value)
        self._sign_latch_time = None   # 방향표지 최초 래치 시각(카운트다운 기준). None=미시작

        # YOLO 검출 구독(정지/감속 게이팅용). 항상 구독하되 gate 는 yolo_enabled 로 제어.
        self.yolo_sub = self.create_subscription(
            DetectionArray,
            str(self.get_parameter('yolo_detections_topic').value),
            self.yolo_callback,
            10,
        )

        # ArUco 정지신호 구독(정지 게이팅용). 항상 구독하되 gate 는 aruco_enabled 로 제어.
        self.aruco_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('aruco_stop_topic').value),
            self.aruco_callback,
            10,
        )

        # ArUco 마커 ID 구독(/detected_marker_id). 마커를 보면 YOLO 전원을 다시 켠다.
        self.marker_sub = self.create_subscription(
            Int32,
            str(self.get_parameter('marker_id_topic').value),
            self.marker_callback,
            10,
        )

        # 파란 표지판 근접 트리거 구독(/sign/near, bluesign_node). True 면 YOLO 전원을 켠다.
        self.sign_near_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('sign_near_topic').value),
            self.sign_near_callback,
            10,
        )

        # 노란 차선 인지 구독(/yellow/lane, LaneDetection). gate 는 ramp_enabled.
        self.ramp_sub = self.create_subscription(
            LaneDetection,
            str(self.get_parameter('ramp_detection_topic').value),
            self.ramp_callback,
            10,
        )

        self.get_logger().info(
            'interpret node started (judgment + control law, offset-only):\n'
            f'  detection_topic={detection_topic}\n'
            f'  lane_topic={lane_topic}\n'
            f'  control_topic={control_topic}\n'
            f'  steer_trim={self.steer_trim} steer_sign='
            f'{float(self.get_parameter("steer_sign").value)}\n'
            f'  ema_alpha={self.ema_alpha}\n'
            f'  kp={float(self.get_parameter("kp_offset").value)}'
            f'(curve {float(self.get_parameter("kp_offset_curve").value)}) '
            f'kd={float(self.get_parameter("kd_offset").value)} '
            f'ki={float(self.get_parameter("ki_offset").value)}\n'
            f'  cruise_throttle={float(self.get_parameter("cruise_throttle").value)} '
            f'(0 => 조향만; 검증 후 param 으로 올릴 것)'
        )

    # ------------------------------------------------------------------ callbk
    def detection_callback(self, msg: LaneDetection):
        """프레임 도착 즉시: 판단(시간필터+정지/감속 판정) -> LaneInfo 발행 -> 제어(PID)
        -> Control 발행. 이벤트구동(한 콜백에서 판단→제어 이어 실행)."""
        detected = bool(msg.left_detected or msg.right_detected)
        # 단독선 = 좌/우 중 정확히 한쪽만 검출 (XOR)
        single_line = bool(msg.left_detected) != bool(msg.right_detected)

        # --- 판단: 진입로 상태기계가 목표의 '출처'를 고른다 (흰 차선 vs 노란 램프) ---
        now = self.get_clock().now()
        if self.prev_step_time is None:
            dt = 1.0 / 30.0
        else:
            dt = (now - self.prev_step_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 30.0
        self.prev_step_time = now
        use_ramp, ramp_offset, ramp_conf, thr_scale, switched = self.ramp.step(
            dt, self.controller.throttle_cmd)

        if use_ramp:
            raw_offset, confidence = ramp_offset, ramp_conf
            detected, single_line = True, False   # 램프 목표는 이미 완성된 차로 중심
        else:
            raw_offset, confidence = float(msg.raw_offset), float(msg.confidence)

        # 출처가 바뀐 프레임: offset 기준이 달라져 미분이 무의미하고 EMA 가 옛 값을
        # 끌고 간다 -> EMA 를 새 값으로 시드하고 미분/적분을 리셋(조향 슬램 방지).
        if switched:
            self.judgment.offset_filtered = raw_offset
            self.controller.reseed(raw_offset)

        # --- 판단: offset 시간필터 ---
        offset = self.judgment.filter_offset(raw_offset, detected, single_line)

        # 판단 결과를 LaneInfo 로도 발행(디버그/rosbag; 런타임 구독자는 없음).
        lane_info = LaneInfo()
        lane_info.header.stamp = msg.header.stamp
        lane_info.header.frame_id = 'interpret'
        lane_info.lane_offset = offset
        lane_info.left_detected = bool(msg.left_detected)
        lane_info.right_detected = bool(msg.right_detected)
        lane_info.confidence = confidence
        self.lane_pub.publish(lane_info)

        # --- 판단: 인지 정지/감속 판정 -> 제어(PID)로 목표 추종 -> Control 발행 ---
        stop, slow = self.judgment.perception_stop_slow()
        bias = self.judgment.lane_bias()
        # heading 은 램프 목표를 실제로 쓰는 프레임에서만 조향에 들어간다(흰 차선 무영향).
        heading = self.ramp.heading if use_ramp else 0.0
        steering, throttle, diag = self.controller.step(
            offset, confidence, stop, slow, bias, thr_scale, heading)
        self.publish_control(steering, throttle)

        # 표지판 인식 후 YOLO 자동 off(transition 7): 래치 후 지연 지나면 1회 off.
        self.check_sign_yolo_off()

        if bool(self.get_parameter('debug_log').value):
            hd = f' hdg={heading:+.3f}' if use_ramp else ''
            self.get_logger().info(
                f'off={offset:+.3f} i={diag["i"]:+.3f} d={diag["d"]:+.3f} '
                f'conf={confidence:.2f} w={diag["w"]:.2f} kp={diag["kp"]:.2f}{hd} '
                f'-> steer={steering:+.3f} thr={throttle:.3f}'
                f'{self.judgment.debug_tag()}{self.ramp.debug_tag()}'
            )

    def ramp_callback(self, msg: LaneDetection):
        """노란 차선 인지(/yellow/lane)를 상태기계에 넘긴다(판정은 detection_callback 에서)."""
        self.ramp.on_ramp(msg)

    def publish_control(self, steering, throttle):
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        self.control_pub.publish(msg)

    # --------------------------------------------------------- 인지 구독(판단 위임)
    # 구독 콜백은 판단(Judgment)으로 위임한다. 실제 정지/감속 판정과 필터는 Judgment 가,
    # PID 는 Controller 가 담당한다(위쪽 클래스). 노드는 배선/발행/로그만.
    def yolo_callback(self, msg: DetectionArray):
        self.judgment.on_yolo(msg)

    def aruco_callback(self, msg: Bool):
        self.judgment.on_aruco(msg)

    def marker_callback(self, msg: Int32):
        """ArUco 마커 인지 -> YOLO 전원 재가동. yolo_wake_marker_id≥0 이면 그 ID만."""
        if not bool(self.get_parameter('yolo_power_gate').value):
            return
        wake = int(self.get_parameter('yolo_wake_marker_id').value)
        if wake >= 0 and int(msg.data) != wake:
            return
        self.set_yolo_active(True, 'ArUco 마커')

    def sign_near_callback(self, msg: Bool):
        """파란 표지판 근접(/sign/near=True) -> YOLO 전원 재가동. ArUco 마커 트리거와 동일하게
        yolo_power_gate 아래에서만 작동한다. 켜기만 하고, 끄는 것은 표지판 인식 후 자동 off
        (check_sign_yolo_off)와 초록불 출발이 관장한다."""
        if not bool(self.get_parameter('yolo_power_gate').value):
            return
        if bool(msg.data):
            self.set_yolo_active(True, '파란 표지판')

    def note_sign_latch(self):
        """방향표지가 near-gate 를 통과해 처음 래치되면 YOLO 자동 off 카운트다운을 시작한다.
        YOLO 가 켜져 무장(_sign_off_armed)돼 있을 때만, 아직 카운트다운 전일 때만 1회."""
        if self._sign_off_armed and self._sign_latch_time is None:
            self._sign_latch_time = self.get_clock().now()

    def check_sign_yolo_off(self):
        """표지판 래치 후 sign_yolo_off_delay 초가 지나면 YOLO 를 1회 끈다(transition 7).
        매 프레임 호출. set_yolo_active(False) 가 무장 해제 + 카운트다운 리셋을 겸한다."""
        if not bool(self.get_parameter('yolo_power_gate').value):
            return
        if not self._sign_off_armed or self._sign_latch_time is None:
            return
        delay = float(self.get_parameter('sign_yolo_off_delay').value)
        age = (self.get_clock().now() - self._sign_latch_time).nanoseconds * 1e-9
        if age >= delay:
            self.set_yolo_active(False, '표지판 인식 후')

    def set_yolo_active(self, on, reason=''):
        """YOLO 전원 게이트 명령(/yolo/active). 상태가 바뀔 때만 발행(중복 방지).
        켜질 때 표지판 자동 off 를 무장하고, 꺼질 때 해제한다(카운트다운 리셋)."""
        on = bool(on)
        if on == self._yolo_active_cmd:
            return
        self._yolo_active_cmd = on
        # 표지판 자동 off 무장/해제: ON 이면 무장(이후 표지판 래치가 카운트다운 시작),
        # OFF 면 해제. ArUco 재가동도 무장하지만 빨간도로엔 방향표지가 없어 래치가 안 생겨
        # 자동 off 가 걸리지 않는다(아루코 wake 와 충돌 없음).
        self._sign_off_armed = on
        self._sign_latch_time = None
        m = Bool()
        m.data = on
        self.yolo_active_pub.publish(m)
        self.get_logger().info(
            'YOLO 전원 -> %s%s' % ('ON' if on else 'OFF', ' (%s)' % reason if reason else ''))

    # ------------------------------------------------------------------ config
    def load_steer_trim(self):
        if not os.path.exists(self.vehicle_config_file):
            return 0.0
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return 0.0
        return float(config_data.get('STEER_TRIM', 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = InterpretNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
