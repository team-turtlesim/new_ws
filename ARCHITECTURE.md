# 자율주행 시뮬 구조 (new_ws) — 2026-07-08 기준

차선주행 + 미션(아루코 장애물 / YOLO 신호등·표지판)을 pinky 가제보에서 통합 운용하는 구조.
차선 구동(구동팀 원본)은 **무수정 보존**, 미션 제어는 전부 **mission_override 한 곳**에 병합.

## 전체 파이프라인 (시뮬)

```
                              ┌────────────── 가제보(pinky) ──────────────┐
                              │  /camera/image_raw (raw)   /cmd_vel → 로봇 │
                              └──────┬───────────────────────────▲────────┘
                                     │                           │
            ┌────────────────────────┼───────────────────────────┼──────────────┐
            │  [차선 인지·판단·제어 — 구동팀 원본 무수정]          │              │
            │                        ▼                           │              │
            │  sim_image_bridge → opencv_node → lane_node → interpret_node       │
            │  (raw→압축)          (색마스크)   (차선검출)   (판단+offset PID)     │
            │                                                    │ /control      │
            │                                                    ▼               │
   [인지]   │                                          ┌──────────────────┐      │
   aruco_detector ─ /aruco_stop ────────────────────▶ │  mission_override │      │
   sign_detector  ─ /detected_sign ─────────────────▶ │  (미션 제어 병합)  │      │
   (YOLO 320)                                          └────────┬─────────┘      │
            │                                                   │ /control_safe  │
            │                                          sim_cmd_bridge ───────────┘
            │                                          (/control_safe→/cmd_vel)
            └───────────────────────────────────────────────────────────────────┘
```

## mission_override — 미션 제어 병합 (우리 핵심)
`/control`(interpret) 을 받아 미션 신호를 반영해 `/control_safe` 로 재발행. **우선순위**:

| 순위 | 미션 | 입력 | 동작 |
|---|---|---|---|
| 1 (최우선) | 아루코 장애물 | `/aruco_stop`=true | throttle=0 정지 |
| 1 (최우선) | 빨간불 | `/detected_sign`=red_light | throttle=0 정지(래치, green까지) |
| 2 | 화살표 방향 | `/detected_sign`=left/right_sign | steering offset(전진유지). near-gate: 가까울때만 |
| 3 (평소) | 없음 | — | /control 그대로 통과 |

- **제자리 회전 금지**: 화살표는 throttle 안 건드리고 steering 만 offset(실차 Ackermann).
- 확정 파라미터: arrow_steer_sign=-1.0, arrow_offset=0.15, arrow_near_ratio=0.25, arrow_timeout=0.2.

## 패키지 구성 (new_ws/src)
| 패키지 | 출처 | 역할 |
|---|---|---|
| interface, opencv, lane_detection, interpret, control, config, camera, battery, topst_utils, monitor | 구동팀 원본(무수정) | 차선 인지·판단·제어(interpret가 PID까지), 실차 하드웨어 |
| **sim_bridge** (우리) | 시뮬 전용 | image_bridge(raw→압축), cmd_bridge(/control_safe→/cmd_vel), 런치 |
| **aruco** (우리) | 시뮬·실차 공용 | aruco_detector_node(→/aruco_stop), **mission_override**(미션 병합) |
| **sign** (우리) | 시뮬·실차 공용 | sign_detector_node(YOLO→/detected_sign). 모델 dracer_n.onnx(320) |

## 미션별 상태 (2026-07-08)
| 미션 | 인지 | 제어 | 비고 |
|---|---|---|---|
| 차선 주행 | ✅ | ✅ interpret | 구동팀 원본 |
| 아루코 정지 | ✅ | ✅ mission_override | 검증완료 |
| 신호등 red 정지 | ✅ | ✅ mission_override | end-to-end 검증. 320은 근거리(~0.56m)서 검출 |
| 신호등 green 해제 | ✅ | ✅ mission_override | 주입검증(sim에 green 모델 없음) |
| 화살표 방향 | ✅ | ✅ mission_override | 갈림길 실주행 검증(왼쪽분기 통과) |

## YOLO 모델
- 활성: `sign/models/dracer_n.onnx` = **320x320**(실차 카메라 맞춤, best(3).onnx).
- 백업: `dracer_640_backup.onnx`(구 640), `dracer_320.onnx`(원본 사본).
- 클래스: {0:green_light, 1:left_sign, 2:red_light, 3:right_sign}.
- 320 특성: 640보다 인식거리 짧음(디테일↓, 속도↑). 근거리 0.83~0.92. 실차 카메라는 더 잘 잡힐 가능성.
- red_light 정지거리 이슈: 실차서 light_min_confidence(0.80) 실측 조정 예정.

## 실차 워크스페이스(~/car_ws) — 별도, 갱신 필요
- 시뮬 브리지 제외, control_node(하드웨어) 사용. 배선: interpret→/control_raw→mission_override→/control→control_node.
- **아직 sign 패키지 + mission_override 신호등/화살표 최신본 미반영** → 실차 이식 시 갱신 필요.
  (sign_detector는 실차 compressed 영상 구독으로 변경 필요.)

## 실행 (시뮬)
```
ros2 launch pinky_gazebo launch_sim_empty.launch.xml            # 가제보(별터미널)
ros2 launch sim_bridge sim_pipeline.launch.py cruise_throttle:=0.15
#   인자: use_aruco / use_sign / use_arrow (기본 true), cruise_throttle
```
