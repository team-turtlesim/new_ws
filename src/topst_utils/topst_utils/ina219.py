import time
from smbus2 import SMBus
# from luma.core.interface.serial import i2c
# from luma.core.render import canvas
# from luma.oled.device import ssd1306
# from PIL import ImageFont

# =========================
# TOPST I2C settings
# =========================
I2C_BUS   = 3
OLED_ADDR = 0x3C
INA_ADDR  = 0x42   # i2cdetect -y 3에서 0x42가 잡힘 (0x40은 PCA9685로 보임)

MAX_VOLTAGE = 16.8
MIN_VOLTAGE = 12.0

def get_battery_percentage(voltage):
    if voltage >= MAX_VOLTAGE:
        return 100
    elif voltage <= MIN_VOLTAGE:
        return 0
    return round(((voltage - MIN_VOLTAGE) / (MAX_VOLTAGE - MIN_VOLTAGE)) * 100)

# =========================
# INA219 minimal driver (no adafruit)
# =========================
class INA219:
    REG_CONFIG  = 0x00
    REG_SHUNT_V = 0x01
    REG_BUS_V   = 0x02
    REG_POWER   = 0x03
    REG_CURRENT = 0x04
    REG_CALIB   = 0x05

    def __init__(self, bus: SMBus, addr: int, r_shunt_ohm=0.1, max_current_a=2.0):
        self.bus = bus
        self.addr = addr

        # calibration
        self.current_lsb = max_current_a / 32768.0
        self.power_lsb = 20.0 * self.current_lsb
        calib = int(0.04096 / (self.current_lsb * r_shunt_ohm))
        if calib <= 0 or calib > 0xFFFF:
            raise ValueError(f"Bad calibration={calib}. Check r_shunt/max_current.")

        self.write_u16(self.REG_CALIB, calib)

        # 32V range, PGA/ADC 기본값, continuous shunt+bus
        self.write_u16(self.REG_CONFIG, 0x399F)

    def swap16(self, v: int) -> int:
        return ((v & 0xFF) << 8) | ((v >> 8) & 0xFF)

    def to_int16(self, v: int) -> int:
        return v - 0x10000 if v & 0x8000 else v

    def read_u16(self, reg: int) -> int:
        raw = self.bus.read_word_data(self.addr, reg)
        return self.swap16(raw)

    def write_u16(self, reg: int, value: int):
        self.bus.write_word_data(self.addr, reg, self.swap16(value & 0xFFFF))

    @property
    def shunt_voltage(self) -> float:
        # volts
        raw = self.to_int16(self.read_u16(self.REG_SHUNT_V))
        return raw * 10e-6  # 10uV/bit

    @property
    def bus_voltage(self) -> float:
        # volts
        raw = self.read_u16(self.REG_BUS_V)
        raw = (raw >> 3) & 0x1FFF
        return raw * 4e-3  # 4mV/bit

    @property
    def current(self) -> float:
        # mA
        raw = self.to_int16(self.read_u16(self.REG_CURRENT))
        return (raw * self.current_lsb) * 1000.0

    @property
    def power(self) -> float:
        # W
        raw = self.read_u16(self.REG_POWER)
        return raw * self.power_lsb
