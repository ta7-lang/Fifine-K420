"""Поиск FFmpeg, проверка кодеков и построение аргументов кодирования."""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

VIDEO_CODECS: dict[str, str] = {
    "libx264": "H.264 — процессор (x264)",
    "libx265": "H.265/HEVC — процессор (x265)",
    "h264_nvenc": "H.264 — NVIDIA NVENC",
    "hevc_nvenc": "H.265 — NVIDIA NVENC",
    "h264_qsv": "H.264 — Intel Quick Sync",
    "h264_amf": "H.264 — AMD AMF",
    "mpeg4": "MPEG-4 Part 2 (совместимость)",
}

AUDIO_CODECS: dict[str, str] = {
    "aac": "AAC",
    "libopus": "Opus",
    "libmp3lame": "MP3",
}

CONTAINERS = ["mp4", "mkv", "mov"]
PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower"]
RATE_CONTROLS = {
    "cbr": "CBR — постоянный битрейт (стриминг)",
    "vbr": "VBR — переменный битрейт",
    "cq": "CQ/CRF — постоянное качество",
}

_NVENC_PRESET = {"ultrafast": "p1", "superfast": "p2", "veryfast": "p3",
                 "faster": "p3", "fast": "p4", "medium": "p5",
                 "slow": "p6", "slower": "p7"}
_QSV_PRESET = {"ultrafast": "veryfast", "superfast": "veryfast"}
_AMF_QUALITY = {"ultrafast": "speed", "superfast": "speed", "veryfast": "speed",
                "faster": "balanced", "fast": "balanced", "medium": "balanced",
                "slow": "quality", "slower": "quality"}


@dataclass
class VideoSettings:
    codec: str = "libx264"
    rate_control: str = "cbr"
    bitrate_kbps: int = 6000
    quality: int = 23          # CRF/CQ, 0 (лучше) … 51 (хуже)
    preset: str = "veryfast"
    gop_seconds: float = 2.0
    container: str = "mp4"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VideoSettings:
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base


@dataclass
class AudioSettings:
    enabled: bool = True
    codec: str = "aac"
    bitrate_kbps: int = 160

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AudioSettings:
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base


def popen_flags() -> dict:
    """Не показывать консольное окно ffmpeg в Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    env = os.environ.get("K420_FFMPEG")
    if env and os.path.isfile(env):
        return env
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:  # noqa: BLE001 — пакет может отсутствовать или быть без бинарника
        pass
    return shutil.which("ffmpeg")


@functools.lru_cache(maxsize=32)
def encoder_works(ffmpeg: str, encoder: str) -> bool:
    """Пробное кодирование 1 кадра: проверяет и наличие кодека, и железа под него."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", "color=c=black:s=256x256:r=30", "-frames:v", "3",
           "-c:v", encoder, "-pix_fmt", pix_fmt_for(encoder), "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15, **popen_flags())
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@functools.lru_cache(maxsize=32)
def audio_encoder_works(ffmpeg: str, encoder: str) -> bool:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", "anullsrc=r=48000:cl=stereo", "-t", "0.2",
           "-c:a", encoder, "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15, **popen_flags())
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def available_video_codecs(ffmpeg: str | None) -> list[str]:
    if not ffmpeg:
        return []
    return [c for c in VIDEO_CODECS if encoder_works(ffmpeg, c)]


def available_audio_codecs(ffmpeg: str | None) -> list[str]:
    if not ffmpeg:
        return []
    return [c for c in AUDIO_CODECS if audio_encoder_works(ffmpeg, c)]


def pix_fmt_for(codec: str) -> str:
    return "nv12" if codec.endswith("_qsv") else "yuv420p"


def build_video_args(vs: VideoSettings, fps: float) -> list[str]:
    """Аргументы FFmpeg для видеокодека с учётом режима битрейта."""
    c = vs.codec
    k = max(100, int(vs.bitrate_kbps))
    b, maxb, buf = f"{k}k", f"{int(k * 1.5)}k", f"{k * 2}k"
    q = max(0, min(51, int(vs.quality)))
    rc = vs.rate_control
    args = ["-c:v", c]

    if c in ("libx264", "libx265"):
        args += ["-preset", vs.preset if vs.preset in PRESETS else "veryfast"]
        if rc == "cq":
            args += ["-crf", str(q)]
        elif rc == "cbr":
            args += ["-b:v", b, "-minrate", b, "-maxrate", b, "-bufsize", buf]
        else:
            args += ["-b:v", b, "-maxrate", maxb, "-bufsize", buf]
    elif c.endswith("_nvenc"):
        args += ["-preset", _NVENC_PRESET.get(vs.preset, "p4")]
        if rc == "cq":
            args += ["-rc", "vbr", "-cq", str(q), "-b:v", "0"]
        elif rc == "cbr":
            args += ["-rc", "cbr", "-b:v", b, "-maxrate", b, "-bufsize", buf]
        else:
            args += ["-rc", "vbr", "-b:v", b, "-maxrate", maxb, "-bufsize", buf]
    elif c.endswith("_qsv"):
        args += ["-preset", _QSV_PRESET.get(vs.preset, vs.preset)]
        if rc == "cq":
            args += ["-global_quality", str(max(1, q))]
        elif rc == "cbr":
            args += ["-b:v", b, "-maxrate", b, "-bufsize", buf]
        else:
            args += ["-b:v", b, "-maxrate", maxb, "-bufsize", buf]
    elif c.endswith("_amf"):
        args += ["-quality", _AMF_QUALITY.get(vs.preset, "balanced")]
        if rc == "cq":
            args += ["-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q)]
        elif rc == "cbr":
            args += ["-rc", "cbr", "-b:v", b, "-maxrate", b, "-bufsize", buf]
        else:
            args += ["-rc", "vbr_peak", "-b:v", b, "-maxrate", maxb, "-bufsize", buf]
    else:  # mpeg4 и прочие
        if rc == "cq":
            # шкала 0..51 → 2..31
            args += ["-q:v", str(2 + round(q * 29 / 51))]
        elif rc == "cbr":
            args += ["-b:v", b, "-minrate", b, "-maxrate", b, "-bufsize", buf]
        else:
            args += ["-b:v", b, "-maxrate", maxb, "-bufsize", buf]

    gop = max(1, round(vs.gop_seconds * fps))
    args += ["-g", str(gop), "-pix_fmt", pix_fmt_for(c)]
    return args


def build_audio_args(a: AudioSettings) -> list[str]:
    return ["-c:a", a.codec, "-b:a", f"{max(32, int(a.bitrate_kbps))}k"]


def container_args(container: str, codec: str) -> list[str]:
    args: list[str] = []
    if container in ("mp4", "mov"):
        args += ["-movflags", "+faststart"]
        if codec in ("libx265", "hevc_nvenc"):
            args += ["-tag:v", "hvc1"]  # чтобы HEVC открывался в плеерах Apple/Windows
    return args


def estimate_mb_per_minute(video_kbps: int, audio_kbps: int) -> float:
    return (video_kbps + audio_kbps) * 60 / 8 / 1024
