"""Описание UVC-параметров камеры и данные о Fifine K420.

Модуль не зависит от Qt, поэтому его легко тестировать.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import cv2

# Режимы, которые поддерживает Fifine K420 (2K/1440p, 1080p, 720p, 480p).
K420_RESOLUTIONS: list[tuple[int, int]] = [
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 480),
]
K420_FPS: list[int] = [30, 25, 20, 15, 10, 5]
# MJPG нужен для высоких разрешений при 30 fps, YUY2 — несжатый поток.
PIXEL_FORMATS: list[str] = ["MJPG", "YUY2"]

DEVICE_HINTS = ("k420", "fifine")


@dataclass(frozen=True)
class PropDef:
    key: str
    label: str
    cv_id: int
    minimum: int
    maximum: int
    default: int
    step: int = 1
    v4l2_names: tuple[str, ...] = ()
    # Диапазон для V4L2 (Linux), если он отличается от DirectShow.
    v4l2_range: tuple[int, int, int] | None = None  # (min, max, default)
    tooltip: str = ""


@dataclass(frozen=True)
class AutoDef:
    key: str
    label: str
    cv_id: int
    controls: str  # ключ ручного параметра, который отключается в авто-режиме
    v4l2_name: str = ""


PROPS: list[PropDef] = [
    PropDef("brightness", "Яркость", cv2.CAP_PROP_BRIGHTNESS, -64, 64, 0,
            v4l2_names=("brightness",)),
    PropDef("contrast", "Контрастность", cv2.CAP_PROP_CONTRAST, 0, 100, 50,
            v4l2_names=("contrast",)),
    PropDef("saturation", "Насыщенность", cv2.CAP_PROP_SATURATION, 0, 100, 64,
            v4l2_names=("saturation",)),
    PropDef("hue", "Оттенок", cv2.CAP_PROP_HUE, -180, 180, 0,
            v4l2_names=("hue",)),
    PropDef("sharpness", "Резкость", cv2.CAP_PROP_SHARPNESS, 0, 100, 50,
            v4l2_names=("sharpness",)),
    PropDef("gamma", "Гамма", cv2.CAP_PROP_GAMMA, 72, 500, 300,
            v4l2_names=("gamma",)),
    PropDef("gain", "Усиление (gain)", cv2.CAP_PROP_GAIN, 0, 100, 0,
            v4l2_names=("gain",),
            tooltip="Аппаратное усиление сенсора. Чем выше — тем светлее и шумнее."),
    PropDef("white_balance", "Баланс белого, K", cv2.CAP_PROP_WB_TEMPERATURE,
            2800, 6500, 4600, step=10,
            v4l2_names=("white_balance_temperature",)),
    PropDef("exposure", "Экспозиция", cv2.CAP_PROP_EXPOSURE, -13, -1, -6,
            v4l2_names=("exposure_time_absolute", "exposure_absolute"),
            v4l2_range=(1, 5000, 156),
            tooltip="В Windows (DirectShow) значение — степень двойки в секундах: "
                    "-6 ≈ 1/64 с, -5 ≈ 1/32 с."),
    PropDef("backlight", "Компенсация засветки", cv2.CAP_PROP_BACKLIGHT, 0, 2, 1,
            v4l2_names=("backlight_compensation",)),
    PropDef("zoom", "Зум (цифровой)", cv2.CAP_PROP_ZOOM, 100, 400, 100,
            v4l2_names=("zoom_absolute",)),
    PropDef("pan", "Панорама", cv2.CAP_PROP_PAN, -10, 10, 0,
            v4l2_names=("pan_absolute",)),
    PropDef("tilt", "Наклон", cv2.CAP_PROP_TILT, -10, 10, 0,
            v4l2_names=("tilt_absolute",)),
]

AUTOS: list[AutoDef] = [
    AutoDef("auto_exposure", "Автоэкспозиция", cv2.CAP_PROP_AUTO_EXPOSURE,
            "exposure", "auto_exposure"),
    AutoDef("auto_wb", "Автобаланс белого", cv2.CAP_PROP_AUTO_WB,
            "white_balance", "white_balance_automatic"),
]

PROP_BY_KEY = {p.key: p for p in PROPS}
AUTO_BY_KEY = {a.key: a for a in AUTOS}


def is_k420_name(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in DEVICE_HINTS)


def backend_choices() -> list[tuple[str, int]]:
    """Список бэкендов захвата для текущей ОС (первый — рекомендуемый)."""
    if sys.platform == "win32":
        return [("DirectShow (рекомендуется)", cv2.CAP_DSHOW),
                ("Media Foundation", cv2.CAP_MSMF)]
    if sys.platform == "darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION)]
    return [("V4L2", cv2.CAP_V4L2), ("Авто", cv2.CAP_ANY)]


def auto_value(backend: int, enabled: bool, auto_key: str) -> float:
    """Значение для включения/выключения авто-режима в зависимости от бэкенда."""
    if auto_key == "auto_exposure" and backend == cv2.CAP_V4L2:
        # V4L2: 3 = aperture priority (авто), 1 = manual
        return 3.0 if enabled else 1.0
    return 1.0 if enabled else 0.0


def is_auto_on(backend: int, raw: float, auto_key: str) -> bool:
    if auto_key == "auto_exposure" and backend == cv2.CAP_V4L2:
        return raw >= 2.5 or abs(raw - 0.75) < 1e-6
    return raw >= 0.5


def default_range(prop: PropDef, backend: int) -> tuple[int, int, int]:
    if backend == cv2.CAP_V4L2 and prop.v4l2_range:
        return prop.v4l2_range
    return prop.minimum, prop.maximum, prop.default


# ---------------------------------------------------------------- V4L2 (Linux)
@dataclass
class V4l2Ctrl:
    name: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    step: int | None = None
    default: int | None = None
    value: int | None = None
    extra: dict = field(default_factory=dict)


_CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def parse_v4l2_ctrls(text: str) -> dict[str, V4l2Ctrl]:
    """Разбирает вывод `v4l2-ctl --list-ctrls`."""
    result: dict[str, V4l2Ctrl] = {}
    for line in text.splitlines():
        m = _CTRL_RE.match(line)
        if not m:
            continue
        name, kind, rest = m.groups()
        ctrl = V4l2Ctrl(name=name, kind=kind)
        for k, v in re.findall(r"(\w+)=(-?\w+)", rest):
            try:
                num = int(v, 0)
            except ValueError:
                ctrl.extra[k] = v
                continue
            if k == "min":
                ctrl.minimum = num
            elif k == "max":
                ctrl.maximum = num
            elif k == "step":
                ctrl.step = num
            elif k == "default":
                ctrl.default = num
            elif k == "value":
                ctrl.value = num
            else:
                ctrl.extra[k] = num
        result[name] = ctrl
    return result


def query_v4l2_ranges(index: int) -> dict[str, tuple[int, int, int, int]]:
    """Реальные диапазоны параметров камеры через v4l2-ctl (если установлен).

    Возвращает {ключ параметра: (min, max, step, default)}.
    """
    exe = shutil.which("v4l2-ctl")
    if not exe or not sys.platform.startswith("linux"):
        return {}
    try:
        out = subprocess.run([exe, "-d", f"/dev/video{index}", "--list-ctrls"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    ctrls = parse_v4l2_ctrls(out)
    ranges: dict[str, tuple[int, int, int, int]] = {}
    for prop in PROPS:
        for name in prop.v4l2_names:
            c = ctrls.get(name)
            if c and c.minimum is not None and c.maximum is not None:
                ranges[prop.key] = (c.minimum, c.maximum, c.step or 1,
                                    c.default if c.default is not None else c.minimum)
                break
    return ranges


def fourcc_to_str(value: float) -> str:
    v = int(value)
    if v <= 0:
        return "?"
    chars = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return chars if chars.isprintable() else "?"
