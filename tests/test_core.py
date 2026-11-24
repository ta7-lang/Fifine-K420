# ruff: noqa: E501
"""Тесты логики без камеры: аргументы FFmpeg, запись, звук, профили, разбор V4L2."""
from __future__ import annotations

import json
import math
import subprocess
import time
import wave

import numpy as np
import pytest

from k420studio.audio import MicSettings, process_block
from k420studio.camera_props import (PROPS, fourcc_to_str, is_k420_name,
                                     parse_v4l2_ctrls)
from k420studio.ffmpeg_tools import (AudioSettings, VideoSettings,
                                     build_video_args, container_args,
                                     estimate_mb_per_minute, find_ffmpeg)
from k420studio.profiles import BUILTIN_PROFILES, ProfileStore, deep_merge
from k420studio.recorder import Recorder, RecordingError, unique_path

FFMPEG = find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not found")


# ------------------------------------------------------------------ ffmpeg args
@pytest.mark.parametrize("codec", ["libx264", "libx265", "h264_nvenc", "hevc_nvenc",
                                   "h264_qsv", "h264_amf", "mpeg4"])
@pytest.mark.parametrize("rc", ["cbr", "vbr", "cq"])
def test_video_args_shape(codec, rc):
    args = build_video_args(VideoSettings(codec=codec, rate_control=rc,
                                          bitrate_kbps=6000, quality=20), 30)
    assert args[:2] == ["-c:v", codec]
    assert "-g" in args and args[args.index("-g") + 1] == "60"
    assert "-pix_fmt" in args
    if rc == "cbr":
        assert "6000k" in args
    assert len(args) % 2 == 0, "аргументы должны идти парами ключ/значение"


def test_cbr_sets_min_max_for_x264():
    a = build_video_args(VideoSettings(codec="libx264", rate_control="cbr",
                                       bitrate_kbps=4000), 30)
    assert a[a.index("-minrate") + 1] == "4000k"
    assert a[a.index("-maxrate") + 1] == "4000k"
    assert a[a.index("-bufsize") + 1] == "8000k"


def test_hevc_tag_only_for_mp4():
    assert "hvc1" in container_args("mp4", "libx265")
    assert "hvc1" not in container_args("mkv", "libx265")


def test_estimate():
    assert math.isclose(estimate_mb_per_minute(8192, 0), 60.0)


def test_settings_roundtrip():
    vs = VideoSettings(codec="libx265", bitrate_kbps=9000, gop_seconds=1.5)
    assert VideoSettings.from_dict(json.loads(json.dumps(vs.to_dict()))) == vs
    au = AudioSettings.from_dict({"bitrate_kbps": "192", "unknown": 1})
    assert au.bitrate_kbps == 192
    assert MicSettings.from_dict({"gain_db": 6}).gain_db == 6.0


# ------------------------------------------------------------------ camera props
def test_props_unique_keys():
    keys = [p.key for p in PROPS]
    assert len(keys) == len(set(keys))
    for p in PROPS:
        assert p.minimum <= p.default <= p.maximum, p.key


def test_k420_name():
    assert is_k420_name("FIFINE K420 Webcam")
    assert is_k420_name("Microphone (K420)")
    assert not is_k420_name("Logitech C920")


def test_fourcc():
    import cv2
    assert fourcc_to_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert fourcc_to_str(0) == "?"


V4L2_SAMPLE = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=5
                       contrast 0x00980901 (int)    : min=0 max=100 step=1 default=50 value=50
white_balance_automatic 0x0098090c (bool)   : default=1 value=1
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1 (50 Hz)
Camera Controls
                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=156 value=156 flags=inactive
"""


def test_parse_v4l2():
    c = parse_v4l2_ctrls(V4L2_SAMPLE)
    assert c["brightness"].minimum == -64 and c["brightness"].value == 5
    assert c["white_balance_temperature"].step == 10
    assert c["white_balance_temperature"].extra.get("flags") == "inactive"
    assert c["white_balance_automatic"].kind == "bool"
    assert c["exposure_time_absolute"].maximum == 5000
    assert c["power_line_frequency"].default == 1


# ------------------------------------------------------------------ audio DSP
def test_gain_and_clip():
    block = np.full((480, 1), 0.5, dtype=np.float32)
    out, rms_db, peak_db, _ = process_block(block, 12.0, -90.0, 1.0)
    assert np.all(out <= 1.0)
    assert peak_db == pytest.approx(0.0, abs=0.01)


def test_gate_closes_on_quiet():
    quiet = np.full((480, 2), 0.001, dtype=np.float32)
    out, *_rest, state = process_block(quiet, 0.0, -40.0, 0.0)
    assert state == 0.0
    assert float(np.max(np.abs(out))) == 0.0
    loud = np.full((480, 2), 0.3, dtype=np.float32)
    out, *_rest, state = process_block(loud, 0.0, -40.0, 0.0)
    assert state == 1.0
    assert float(out[-1, 0]) == pytest.approx(0.3)


# ------------------------------------------------------------------ profiles
def test_profiles(tmp_path):
    st = ProfileStore(tmp_path)
    assert set(BUILTIN_PROFILES) <= set(st.all_names())
    st.save("Мой", {"video": {"bitrate_kbps": 1234}})
    assert st.get("Мой")["video"]["bitrate_kbps"] == 1234
    with pytest.raises(ValueError):
        st.save(next(iter(BUILTIN_PROFILES)), {})
    assert st.delete("Мой") and st.get("Мой") is None
    st.save_last({"output_dir": "/x"})
    last = st.load_last()
    assert last["output_dir"] == "/x" and "camera" in last
    (tmp_path / "profiles.json").write_text("{broken", "utf-8")
    assert st.user_profiles() == {}


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_unique_path(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    assert unique_path(tmp_path, "a.mp4").name == "a_1.mp4"


# ------------------------------------------------------------------ recording
class FakeAudio:
    """Пишет синус в WAV — как AudioEngine, только без микрофона."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def begin_record(self, path):
        sr = 48000
        t = np.arange(int(sr * self.seconds)) / sr
        pcm = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype("<i2")
        self._wf = wave.open(str(path), "wb")  # noqa: SIM115
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(sr)
        self._wf.writeframes(pcm.tobytes())

    def end_record(self):
        self._wf.close()


def _probe(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    return r.stderr


def _record(tmp_path, container, codec, acodec="aac", audio=True, seconds=2.0, fps=30,
            size=(640, 360)):
    out = tmp_path / f"t_{codec}.{container}"
    rec = Recorder(FFMPEG)
    vs = VideoSettings(codec=codec, rate_control="cbr", bitrate_kbps=2000,
                       preset="ultrafast", container=container)
    au = AudioSettings(enabled=audio, codec=acodec, bitrate_kbps=128)
    rec.start(out, size[0], size[1], fps, vs, au, FakeAudio(seconds) if audio else None)
    w, h = size
    t0 = 1000.0
    # имитируем камеру, которая выдаёт только 15 к/с при заданных 30 —
    # рекордер должен дублировать кадры, сохраняя длительность
    n = int(seconds * 15)
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, : (i * 7) % w] = (0, 128, 255)
        rec.push_frame(frame, t0 + i / 15)
    time.sleep(0.2)
    path = rec.stop()
    return path, rec


@needs_ffmpeg
def test_record_mp4_with_audio(tmp_path):
    path, rec = _record(tmp_path, "mp4", "libx264")
    info = _probe(path)
    assert path.exists() and path.stat().st_size > 1000
    assert "h264" in info and "aac" in info
    # длительность ≈ 2 c, несмотря на 15 к/с от «камеры»
    dur = info.split("Duration: ")[1].split(",")[0]
    hh, mm, ss = dur.split(":")
    assert 1.7 <= float(ss) + int(mm) * 60 <= 2.3
    assert rec.frames_written >= 58


@needs_ffmpeg
def test_record_mkv_hevc_opus(tmp_path):
    from k420studio.ffmpeg_tools import audio_encoder_works, encoder_works
    if not (encoder_works(FFMPEG, "libx265") and audio_encoder_works(FFMPEG, "libopus")):
        pytest.skip("нет libx265/libopus в сборке ffmpeg")
    path, _ = _record(tmp_path, "mkv", "libx265", acodec="libopus")
    info = _probe(path)
    assert "hevc" in info and "opus" in info


@needs_ffmpeg
def test_record_without_audio_odd_size(tmp_path):
    path, _ = _record(tmp_path, "mov", "mpeg4", audio=False, size=(641, 361))
    info = _probe(path)
    assert "640x360" in info and "Audio" not in info


@needs_ffmpeg
def test_bad_codec_raises(tmp_path):
    rec = Recorder(FFMPEG)
    rec.start(tmp_path / "x.mp4", 320, 240, 30,
              VideoSettings(codec="nonexistent_codec"), AudioSettings(enabled=False), None)
    rec.push_frame(np.zeros((240, 320, 3), np.uint8), 1.0)
    time.sleep(0.3)
    with pytest.raises(RecordingError):
        rec.stop()


def test_audio_engine_writes_wav(tmp_path):
    """Колбэк AudioEngine пишет обработанный звук в WAV (без реального микрофона)."""
    from k420studio.audio import AudioEngine

    eng = AudioEngine()
    eng.settings = MicSettings(channels=2, samplerate=48000, gain_db=6.0)
    eng.begin_record(tmp_path / "a.wav")
    block = np.full((480, 2), 0.25, dtype=np.float32)
    for _ in range(100):
        eng._callback(block, 480, None, None)
    eng.end_record()
    with wave.open(str(tmp_path / "a.wav")) as wf:
        assert wf.getnchannels() == 2 and wf.getframerate() == 48000
        assert wf.getnframes() == 48000
        sample = np.frombuffer(wf.readframes(1), "<i2")[0]
    assert abs(sample / 32767 - 0.25 * 10 ** (6 / 20)) < 0.01
    assert -7 < eng.level_db < -5
