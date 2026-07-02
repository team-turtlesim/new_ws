
from __future__ import annotations
from dataclasses import dataclass
from topst_utils.pca9685 import PCA9685

@dataclass
class ServoCalib:
    center_us: int = 1500
    span_us: int = 500          # +/- 범위 (예: 1500±500 => 1000~2000us)
    min_us: int = 1000
    max_us: int = 2000


@dataclass
class EscCalib:
    neutral_us: int = 1500
    fwd_us: int = 2000          # +1.0
    rev_us: int = 1000          # -1.0
    min_us: int = 1000
    max_us: int = 2000


class D3Racer:
    """
    PiRacerPro와 유사한 API:
      - set_steering_percent(x): -1.0 ~ +1.0 (좌/우)
      - set_throttle_percent(x): -1.0 ~ +1.0 (후/전), 0=중립
    """
    def __init__(
        self,
        i2c_bus: int = 3,
        pca9685_addr: int = 0x40,
        freq_hz: float = 50.0,
        steering_channel: int = 0,
        throttle_channel: int = 1,
        steering: ServoCalib = ServoCalib(),
        esc: EscCalib = EscCalib(),
    ):
        self.pwm = PCA9685(bus=i2c_bus, address=pca9685_addr, freq_hz=freq_hz)
        self.st_ch = steering_channel
        self.th_ch = throttle_channel
        self.st = steering
        self.esc = esc

        # 안전: 초기 중립
        self.set_steering_percent(0.0)
        self.set_throttle_percent(0.0)

    @staticmethod
    def clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def set_steering_percent(self, p: float):
        p = float(p)
        p = self.clip(p, -1.0, 1.0)

        pulse = self.st.center_us + p * self.st.span_us
        pulse = self.clip(pulse, self.st.min_us, self.st.max_us)
        self.pwm.set_pulse_us(self.st_ch, pulse)

    def set_throttle_percent(self, p: float):
        p = float(p)
        p = self.clip(p, -1.0, 1.0)

        if p > 0:
            pulse = self.esc.neutral_us + p * (self.esc.fwd_us - self.esc.neutral_us)
        elif p < 0:
            pulse = self.esc.neutral_us + p * (self.esc.neutral_us - self.esc.rev_us)
        else:
            pulse = self.esc.neutral_us

        pulse = self.clip(pulse, self.esc.min_us, self.esc.max_us)
        self.pwm.set_pulse_us(self.th_ch, pulse)

    def stop(self):
        self.set_throttle_percent(0.0)

    def close(self):
        self.stop()
        self.pwm.close()
