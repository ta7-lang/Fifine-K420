"""Запись видео: кадры OpenCV → FFmpeg (stdin), звук → WAV, затем сведение."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .ffmpeg_tools import (AudioSettings, VideoSettings, build_audio_args,
                           build_video_args, container_args, popen_flags)


class AudioSource(Protocol):
    def begin_record(self, path: str | Path) -> None: ...
    def end_record(self) -> None: ...


class RecordingError(RuntimeError):
    pass


class Recorder:
    """Пишет поток кадров с постоянной частотой кадров, синхронизированной с часами.

    Если камера отдаёт кадры медленнее заданного fps, кадры дублируются,
    если быстрее — лишние пропускаются. Так видео не «уезжает» от звука.
    """

    MAX_QUEUE = 90

    def __init__(self, ffmpeg: str) -> None:
        self.ffmpeg = ffmpeg
        self.active = False
        self.dropped = 0
        self.frames_written = 0
        self.out_path: Path | None = None
        self._q: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE)
        self._proc: subprocess.Popen | None = None
        self._writer: threading.Thread | None = None
        self._tmpdir: Path | None = None
        self._audio: AudioSource | None = None
        self._t0: float | None = None
        self._queued = 0
        self._size: tuple[int, int] = (0, 0)
        self._error: str = ""
        self._log_file = None
        self.started_at = 0.0

    # ------------------------------------------------------------------ start
    def start(self, out_path: str | Path, width: int, height: int, fps: float,
              video: VideoSettings, audio: AudioSettings,
              audio_source: AudioSource | None) -> None:
        if self.active:
            raise RecordingError("Запись уже идёт")
        width -= width % 2   # yuv420p требует чётных размеров
        height -= height % 2
        if width < 16 or height < 16:
            raise RecordingError("Некорректный размер кадра")
        self.fps = max(1.0, float(fps))
        self.video = video
        self.audio = audio
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._size = (width, height)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="k420_rec_"))
        self._tmp_video = self._tmpdir / "video.mkv"   # MKV не теряется при сбое
        self._tmp_audio = self._tmpdir / "audio.wav"
        self._log_path = self._tmpdir / "ffmpeg.log"
        self._log_file = open(self._log_path, "wb")  # noqa: SIM115

        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{self.fps:g}", "-i", "-",
               *build_video_args(video, self.fps), str(self._tmp_video)]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL,
                                          stderr=self._log_file, **popen_flags())
        except OSError as exc:
            self._cleanup()
            raise RecordingError(f"Не удалось запустить FFmpeg: {exc}") from exc

        self._audio = audio_source if audio.enabled else None
        if self._audio is not None:
            try:
                self._audio.begin_record(self._tmp_audio)
            except Exception as exc:  # noqa: BLE001
                self._audio = None
                self._error = f"Звук не записывается: {exc}"

        self.dropped = 0
        self.frames_written = 0
        self._queued = 0
        self._t0 = None
        self._q = queue.Queue(maxsize=self.MAX_QUEUE)
        self._writer = threading.Thread(target=self._write_loop, daemon=True,
                                        name="k420-writer")
        self._writer.start()
        self.started_at = time.monotonic()
        self.active = True

    # ------------------------------------------------------------------ frames
    def push_frame(self, frame: np.ndarray, now: float | None = None) -> None:
        if not self.active:
            return
        now = time.monotonic() if now is None else now
        if self._t0 is None:
            self._t0 = now
        target = int((now - self._t0) * self.fps) + 1
        n = target - self._queued
        if n <= 0:
            return
        n = min(n, int(self.fps * 2))
        w, h = self._size
        if frame.shape[1] != w or frame.shape[0] != h:
            import cv2  # локальный импорт: модуль удобно тестировать без cv2
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        try:
            self._q.put_nowait((np.ascontiguousarray(frame), n))
            self._queued += n
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        while True:
            item = self._q.get()
            if item is None:
                break
            frame, n = item
            data = frame.tobytes()
            try:
                for _ in range(n):
                    proc.stdin.write(data)
                    self.frames_written += 1
            except (BrokenPipeError, OSError, ValueError):
                self._error = "FFmpeg неожиданно завершился во время записи"
                # опустошаем очередь, чтобы stop() не завис
                while True:
                    try:
                        if self._q.get_nowait() is None:
                            return
                    except queue.Empty:
                        return

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.active else 0.0

    # ------------------------------------------------------------------ stop
    def stop(self) -> Path:
        """Останавливает запись и сводит звук с видео. Блокирующий вызов."""
        if not self.active:
            raise RecordingError("Запись не запущена")
        self.active = False
        try:
            self._q.put(None, timeout=10)
        except queue.Full:
            pass
        if self._writer:
            self._writer.join(timeout=30)
        if self._audio is not None:
            self._audio.end_record()
        proc = self._proc
        try:
            if proc and proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc:
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if self._log_file:
                self._log_file.close()
                self._log_file = None

            if not proc or proc.returncode != 0 or not self._tmp_video.exists() \
                    or self._tmp_video.stat().st_size == 0:
                raise RecordingError("Ошибка кодирования видео:\n" + self._log_tail())
            self._finalize()
            return self.out_path  # type: ignore[return-value]
        finally:
            self._cleanup()

    def _finalize(self) -> None:
        assert self.out_path is not None
        has_audio = (self._audio is not None and self._tmp_audio.exists()
                     and self._tmp_audio.stat().st_size > 44)
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(self._tmp_video)]
        if has_audio:
            cmd += ["-i", str(self._tmp_audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", *build_audio_args(self.audio), "-shortest"]
        else:
            cmd += ["-map", "0:v:0", "-c:v", "copy"]
        cmd += [*container_args(self.video.container, self.video.codec),
                str(self.out_path)]
        r = subprocess.run(cmd, capture_output=True, **popen_flags())
        if r.returncode != 0:
            # спасаем хотя бы видео без звука
            fallback = self.out_path.with_suffix(".mkv")
            if fallback == self.out_path:
                fallback = self.out_path.with_name(self.out_path.stem + "_video.mkv")
            shutil.copyfile(self._tmp_video, fallback)
            self.out_path = fallback
            msg = r.stderr.decode("utf-8", "replace")[-1500:]
            raise RecordingError(
                f"Не удалось свести звук и видео. Видео без звука сохранено: "
                f"{fallback}\n{msg}")

    def _log_tail(self) -> str:
        try:
            return self._log_path.read_text("utf-8", "replace")[-1500:]
        except OSError:
            return ""

    @property
    def warning(self) -> str:
        return self._error

    def _cleanup(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        self._proc = None
        self._writer = None


def default_filename(container: str) -> str:
    return time.strftime("K420_%Y-%m-%d_%H-%M-%S") + f".{container}"


def unique_path(folder: str | Path, name: str) -> Path:
    p = Path(folder) / name
    stem, suf, i = p.stem, p.suffix, 1
    while p.exists():
        p = p.with_name(f"{stem}_{i}{suf}")
        i += 1
    return p


def is_writable_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and os.access(p, os.W_OK)
