"""YOLO ONNX 추론 래퍼 (onnxruntime, CPU).

ROS 와 분리된 순수 추론 모듈이라 단독 테스트가 쉽다. 하는 일:
    BGR 이미지 -> letterbox 전처리 -> onnxruntime 추론 -> 후처리(+필요시 NMS)
    -> 원본 좌표계의 검출 리스트(dict) 반환.

두 가지 ONNX 출력 형식을 자동 감지해 모두 지원한다:
  1) end2end / NMS-free  : shape (1, N, 6) = [x1, y1, x2, y2, score, class]
     (YOLO26 기본, YOLOv10 계열). 임계값만 적용, NMS 불필요.
  2) raw                 : shape (1, 4+nc, A) 또는 전치 (1, A, 4+nc)
     (일반 YOLOv8/v11 export). box=xywh(center, 입력픽셀), class score 0~1.
     디코드 후 클래스별 NMS 를 돌린다.

좌표는 항상 '추론에 넣은 원본 이미지'의 픽셀 기준으로 되돌려 반환한다(letterbox 역변환).
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_IMPORT_ERROR = None
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover
    ort = None
    ORT_IMPORT_ERROR = exc


def load_labels(labels_path):
    """labels.txt -> 클래스 이름 리스트. '#' 주석줄/빈 줄은 무시."""
    names = []
    with open(labels_path, 'r', encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            names.append(line)
    return names


class YoloDetector:
    """onnxruntime 세션을 감싼 YOLO 검출기.

    Parameters
    ----------
    model_path : str            로드할 .onnx 경로
    labels : list[str]          class_id -> 이름 (없으면 'id:<n>' 로 대체)
    conf_threshold : float      이 값 미만의 검출은 버림
    iou_threshold : float       raw 출력 NMS 의 IoU 임계값
    input_size : int            정사각 입력 한 변(px). 모델이 정적 크기면 그 값으로 덮어씀.
    num_threads : int           onnxruntime intra-op 스레드(0=자동)
    """

    def __init__(self, model_path, labels=None, conf_threshold=0.35,
                 iou_threshold=0.45, input_size=640, num_threads=0):
        if ORT_IMPORT_ERROR is not None:
            raise RuntimeError(
                'onnxruntime is not installed. Run: pip install onnxruntime==1.18.1'
            ) from ORT_IMPORT_ERROR

        self.labels = list(labels) if labels else []
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

        so = ort.SessionOptions()
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=['CPUExecutionProvider']
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # 모델 입력이 정적(H,W 고정)이면 그 크기를 쓰고, 동적(None/문자열)이면 param 값.
        static_hw = self._static_input_hw(inp.shape)
        self.input_size = static_hw if static_hw is not None else int(input_size)

    @staticmethod
    def _static_input_hw(shape):
        """입력 shape [N,C,H,W] 에서 정적 정사각 크기를 뽑는다(H==W 정수면 그 값)."""
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
            return int(h)
        return None

    def label_of(self, class_id):
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f'id:{class_id}'

    # ------------------------------------------------------------------ preprocess
    def _letterbox(self, bgr):
        """비율 유지 리사이즈 + 가운데 패딩(114 회색)으로 정사각 입력을 만든다.
        반환: (input_tensor[1,3,S,S] float32, ratio, pad_w, pad_h)."""
        s = self.input_size
        h0, w0 = bgr.shape[:2]
        r = min(s / w0, s / h0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_w = (s - new_w) // 2
        pad_h = (s - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # -> (1,3,S,S)
        return np.ascontiguousarray(tensor), r, pad_w, pad_h

    # ------------------------------------------------------------------ inference
    def infer(self, bgr):
        """BGR 이미지 한 장을 추론해 검출 dict 리스트를 반환.
        각 dict: {label, class_id, confidence, x, y, width, height} (원본 픽셀, 좌상단 xywh)."""
        tensor, r, pad_w, pad_h = self._letterbox(bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        out = outputs[0]
        h0, w0 = bgr.shape[:2]
        boxes, scores, class_ids = self._decode(out)
        return self._to_detections(boxes, scores, class_ids, r, pad_w, pad_h, w0, h0)

    # ------------------------------------------------------------------ postprocess
    def _decode(self, out):
        """ONNX 출력을 (boxes_xyxy[input px], scores, class_ids) 로 변환.
        형식(end2end vs raw)을 shape 로 자동 판별한다."""
        arr = np.asarray(out)
        if arr.ndim == 3:
            arr = arr[0]  # 배치 제거 -> 2D

        # end2end / NMS-free: (N, 6) = [x1,y1,x2,y2,score,class]
        if arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] != 6:
            return self._decode_end2end(arr)

        # raw: (4+nc, A) 또는 (A, 4+nc). 행이 열보다 적으면 (4+nc, A) 로 보고 전치.
        if arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T  # -> (A, 4+nc)
            return self._decode_raw(arr)

        raise ValueError(f'Unsupported YOLO ONNX output shape: {np.asarray(out).shape}')

    def _decode_end2end(self, arr):
        """(N,6) [x1,y1,x2,y2,score,class] -> 임계값 필터. NMS 불필요."""
        scores = arr[:, 4]
        keep = scores >= self.conf_threshold
        arr = arr[keep]
        boxes = arr[:, 0:4].astype(np.float32)
        scores = arr[:, 4].astype(np.float32)
        class_ids = arr[:, 5].astype(np.int32)
        return boxes, scores, class_ids

    def _decode_raw(self, pred):
        """(A, 4+nc): box=xywh(center, 입력픽셀), 뒤 nc=클래스 확률(0~1).
        최대확률 클래스로 스코어링 -> 임계값 -> 클래스별 NMS."""
        boxes_xywh = pred[:, 0:4]
        cls = pred[:, 4:]
        class_ids = np.argmax(cls, axis=1).astype(np.int32)
        scores = cls[np.arange(cls.shape[0]), class_ids].astype(np.float32)

        keep = scores >= self.conf_threshold
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if boxes_xywh.shape[0] == 0:
            return np.zeros((0, 4), np.float32), scores, class_ids

        # xywh(center) -> xyxy
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)

        keep_idx = self._nms_per_class(boxes, scores, class_ids)
        return boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]

    def _nms_per_class(self, boxes, scores, class_ids):
        """클래스별 NMS: 좌표에 클래스 오프셋을 더해 클래스 간 억제를 막는 표준 트릭."""
        if boxes.shape[0] == 0:
            return np.empty((0,), np.int32)
        max_coord = float(boxes.max()) if boxes.size else 0.0
        offsets = class_ids.astype(np.float32)[:, None] * (max_coord + 1.0)
        shifted = boxes + offsets
        return self._nms(shifted, scores, self.iou_threshold)

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        """단순 numpy NMS. 반환: 유지할 인덱스 배열."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thr]
        return np.asarray(keep, dtype=np.int32)

    def _to_detections(self, boxes, scores, class_ids, r, pad_w, pad_h, w0, h0):
        """letterbox 역변환으로 boxes(입력픽셀 xyxy)를 원본 픽셀 좌상단 xywh 로.
        원본 경계로 clamp."""
        dets = []
        for (bx1, by1, bx2, by2), sc, cid in zip(boxes, scores, class_ids):
            x1 = (bx1 - pad_w) / r
            y1 = (by1 - pad_h) / r
            x2 = (bx2 - pad_w) / r
            y2 = (by2 - pad_h) / r
            x1 = float(np.clip(x1, 0, w0 - 1))
            y1 = float(np.clip(y1, 0, h0 - 1))
            x2 = float(np.clip(x2, 0, w0 - 1))
            y2 = float(np.clip(y2, 0, h0 - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(cid)
            dets.append({
                'label': self.label_of(cid),
                'class_id': cid,
                'confidence': float(sc),
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1,
            })
        return dets
