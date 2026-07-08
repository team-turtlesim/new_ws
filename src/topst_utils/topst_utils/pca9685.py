import time
try:
    from smbus2 import SMBus
except ImportError:
    # fallback (환경에 따라 smbus가 있을 수 있음)
    from smbus import SMBus  # type: ignore


# PCA9685 레지스터
MODE1      = 0x00
MODE2      = 0x01
PRESCALE   = 0xFE
LED0_ON_L  = 0x06

# MODE1 비트
RESTART = 0x80
SLEEP   = 0x10
ALLCALL = 0x01

# MODE2 비트
OUTDRV  = 0x04



class PCA9685:
    """
    PCA9685를 I2C로 직접 제어 (board/blinka 불필요)
    """
    def __init__(self, bus: int = 1, address: int = 0x40, freq_hz: float = 50.0, osc_hz: float = 25_000_000.0):
        self.busnum = bus
        self.addr = address
        self.osc_hz = osc_hz
        self.freq_hz = freq_hz
        self.i2c = SMBus(bus)

        # 초기화: MODE1, MODE2 설정
        self.write8(MODE1, ALLCALL)      # ALLCALL on
        self.write8(MODE2, OUTDRV)       # totem-pole
        time.sleep(0.01)

        self.set_pwm_freq(freq_hz)

    def write8(self, reg: int, val: int):
        self.i2c.write_byte_data(self.addr, reg, val & 0xFF)

    def read8(self, reg: int) -> int:
        return self.i2c.read_byte_data(self.addr, reg)

    def set_pwm_freq(self, freq_hz: float):
        """
        PCA9685 PWM frequency 설정
        prescale = round(osc / (4096*freq)) - 1
        """
        freq_hz = float(freq_hz)
        prescaleval = self.osc_hz / (4096.0 * freq_hz) - 1.0
        prescale = int(round(prescaleval))

        oldmode = self.read8(MODE1)
        newmode = (oldmode & 0x7F) | SLEEP     # sleep
        self.write8(MODE1, newmode)
        self.write8(PRESCALE, prescale)
        self.write8(MODE1, oldmode)
        time.sleep(0.005)
        self.write8(MODE1, oldmode | RESTART)  # restart

        self.freq_hz = freq_hz

    def us_to_ticks(self, us: float) -> int:
        period_us = 1_000_000.0 / self.freq_hz
        ticks = int(round((us / period_us) * 4096.0))
        # PCA9685는 12-bit(0~4095)
        if ticks < 0:
            return 0
        if ticks > 4095:
            return 4095
        return ticks

    def set_pwm(self, channel: int, on: int, off: int):
        base = LED0_ON_L + 4 * channel
        self.write8(base + 0, on & 0xFF)
        self.write8(base + 1, (on >> 8) & 0xFF)
        self.write8(base + 2, off & 0xFF)
        self.write8(base + 3, (off >> 8) & 0xFF)

    def set_pulse_us(self, channel: int, pulse_us: float):
        off = self.us_to_ticks(pulse_us)
        self.set_pwm(channel, 0, off)

    def close(self):
        try:
            self.i2c.close()
        except Exception:
            pass