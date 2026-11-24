"""Захват звука с микрофона камеры: индикатор уровня, усиление, шумодав, запись в WAV."""
from __future__ import annotations

import math
import threading
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:  # PortAudio может отсутствовать (например, Linux без libportaudio2)
    import sounddevice as sd
    SD_ERROR = ""
except Exception as exc:  # noqa: BLE001
    sd = None
    SD_ERROR = str(exc)

from .camera_props import is_k420_name

SAMPLE_RATES = [48000, 44100, 32000, 16000]


@dataclass
class MicSettings:
    device: str = ""          # имя устройства ("" — по умолчанию / K420)
    samplerate: int = 48000
    channels: int = 1
    gain_db: float = 0.0      # программное усиление, дБ
    gate_db: float = -90.0    # порог шумоподавителя; -90 = выключен
    monitor: bool = False     # прослушивание в наушниках

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MicSettings:
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base


def list_input_devices() -> list[tuple[int, str, int, float]]:
    """[(index, name, max_channels, default_samplerate)] — только входы."""
    if sd is None:
        return []
    result = []
    try:
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                api = hostapis[dev["hostapi"]]["name"] if hostapis else ""
                name = f"{dev['name']} [{api}]" if api else dev["name"]
                result.append((i, name, int(dev["max_input_channels"]),
                               float(dev.get("default_samplerate", 48000))))
    except Exception:  # noqa: BLE001
        return []
    return result


def pick_k420_device(devices: list[tuple[int, str, int, float]]) -> int | None:
    for idx, name, _ch, _sr in devices:
        if is_k420_name(name):
            return idx
    return None


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -120.0


def process_block(block: np.ndarray, gain_db: float, gate_db: float,
                  gate_state: float) -> tuple[np.ndarray, float, float, float]:
    """Усиление + мягкий шумоподавитель (noise gate).

    Возвращает (обработанный блок, rms_db, peak_db, новое состояние гейта 0..1).
    """
    data = block.astype(np.float32, copy=True)
    if gain_db:
        data *= db_to_lin(gain_db)
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    if gate_db > -89.9:
        target = 1.0 if lin_to_db(rms) >= gate_db else 0.0
        # плавное открытие/закрытие, чтобы не было щелчков
        n = data.shape[0]
        ramp = np.linspace(gate_state, target, n, dtype=np.float32)
        if data.ndim == 2:
            ramp = ramp[:, None]
        data *= ramp
        gate_state = target
    np.clip(data, -1.0, 1.0, out=data)
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return data, lin_to_db(rms), lin_to_db(peak), gate_state


class AudioEngine:
    """Постоянно открытый входной поток. Запись в WAV включается по требованию."""

    def __init__(self) -> None:
        self.settings = MicSettings()
        self._stream = None
        self._lock = threading.Lock()
        self._wav: wave.Wave_write | None = None
        self._gate_state = 1.0
        self.level_db = -120.0
        self.peak_db = -120.0
        self.clipped = False
        self.device_index: int | None = None
        self.last_error = ""

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self, device_index: int | None, settings: MicSettings) -> None:
        self.stop()
        if sd is None:
            raise RuntimeError(f"Звук недоступен: {SD_ERROR}")
        self.settings = settings
        self.device_index = device_index
        sd.check_input_settings(device=device_index, channels=settings.channels,
                                samplerate=settings.samplerate, dtype="float32")
        if settings.monitor:
            # дуплексный поток: вход камеры → выход по умолчанию (наушники)
            stream = sd.Stream(device=(device_index, None),
                               channels=(settings.channels, settings.channels),
                               samplerate=settings.samplerate, dtype="float32",
                               blocksize=0, latency="low", callback=self._duplex_callback)
        else:
            stream = sd.InputStream(device=device_index, channels=settings.channels,
                                    samplerate=settings.samplerate, dtype="float32",
                                    blocksize=0, latency="low", callback=self._callback)
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        self.end_record()
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # noqa: BLE001
                pass
        self.level_db = self.peak_db = -120.0

    def update_live(self, gain_db: float, gate_db: float) -> None:
        """Изменения, применяемые без перезапуска потока."""
        self.settings.gain_db = gain_db
        self.settings.gate_db = gate_db

    def begin_record(self, path: str | Path) -> None:
        wf = wave.open(str(path), "wb")  # noqa: SIM115 — закрывается в end_record()
        wf.setnchannels(self.settings.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.settings.samplerate)
        with self._lock:
            self._wav = wf

    def end_record(self) -> None:
        with self._lock:
            wf, self._wav = self._wav, None
        if wf is not None:
            wf.close()

    def _handle(self, indata) -> np.ndarray:
        data, rms_db, peak_db, self._gate_state = process_block(
            indata, self.settings.gain_db, self.settings.gate_db, self._gate_state)
        self.level_db = rms_db
        self.peak_db = peak_db
        if peak_db > -0.1:
            self.clipped = True
        pcm = (data * 32767.0).astype("<i2").tobytes()
        with self._lock:
            if self._wav is not None:
                self._wav.writeframesraw(pcm)
        return data

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        self._handle(indata)

    def _duplex_callback(self, indata, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        outdata[:] = self._handle(indata)
