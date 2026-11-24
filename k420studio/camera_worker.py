"""Поток захвата видео. Все обращения к cv2.VideoCapture — только из этого потока."""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .camera_props import (AUTO_BY_KEY, PROP_BY_KEY, PROPS, auto_value,
                           fourcc_to_str, is_auto_on, query_v4l2_ranges)

PREVIEW_MAX_WIDTH = 1280


def save_image(path: str | Path, frame: np.ndarray) -> None:
    """cv2.imwrite не умеет пути с кириллицей в Windows — кодируем сами."""
    path = Path(path)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, frame, params)
    if not ok:
        raise OSError("Не удалось закодировать изображение")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    opened = Signal(dict)          # фактический режим + значения и диапазоны параметров
    props_read = Signal(dict)      # {ключ: значение}
    prop_result = Signal(str, bool, float)  # ключ, успех, фактическое значение
    error = Signal(str)
    closed = Signal()
    snapshot_saved = Signal(str)
    fps_measured = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cmds: queue.Queue = queue.Queue()
        self._running = True
        self._cap: cv2.VideoCapture | None = None
        self._backend = cv2.CAP_ANY
        self._rec_lock = threading.Lock()
        self._recorder = None
        self._preview_pending = False
        self.mirror = False
        self.rotate180 = False
        self.last_frame: np.ndarray | None = None
        self.frame_size = (0, 0)
        self.actual_fps = 0.0

    # ---------------------------------------------------------- API для UI
    def open_camera(self, index: int, backend: int, width: int, height: int,
                    fps: int, fourcc: str) -> None:
        self._cmds.put(("open", (index, backend, width, height, fps, fourcc)))

    def close_camera(self) -> None:
        self._cmds.put(("close", None))

    def set_prop(self, key: str, value: float) -> None:
        self._cmds.put(("prop", (key, float(value))))

    def set_auto(self, key: str, enabled: bool) -> None:
        self._cmds.put(("auto", (key, bool(enabled))))

    def read_props(self) -> None:
        self._cmds.put(("read", None))

    def driver_dialog(self) -> None:
        self._cmds.put(("dialog", None))

    def snapshot(self, path: str) -> None:
        self._cmds.put(("snapshot", path))

    def set_recorder(self, recorder) -> None:
        with self._rec_lock:
            self._recorder = recorder

    def preview_consumed(self) -> None:
        self._preview_pending = False

    def stop(self) -> None:
        self._running = False
        self.wait(5000)

    @property
    def backend(self) -> int:
        return self._backend

    # ---------------------------------------------------------- поток
    def run(self) -> None:
        fails = 0
        t_last = time.monotonic()
        n_frames = 0
        while self._running:
            self._process_commands()
            cap = self._cap
            if cap is None:
                time.sleep(0.03)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 50:
                    self._release()
                    self.error.emit("Камера перестала отдавать кадры "
                                    "(отключена или занята другой программой).")
                    fails = 0
                time.sleep(0.01)
                continue
            fails = 0
            now = time.monotonic()
            if self.rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self.last_frame = frame
            self.frame_size = (frame.shape[1], frame.shape[0])

            with self._rec_lock:
                rec = self._recorder
            if rec is not None:
                rec.push_frame(frame, now)

            n_frames += 1
            if now - t_last >= 1.0:
                self.actual_fps = n_frames / (now - t_last)
                self.fps_measured.emit(self.actual_fps)
                n_frames, t_last = 0, now

            if not self._preview_pending:
                self._preview_pending = True
                self.frame_ready.emit(self._to_qimage(frame))
        self._release()

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / w
            frame = cv2.resize(frame, (PREVIEW_MAX_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    def _process_commands(self) -> None:
        cmds = []
        while True:
            try:
                cmds.append(self._cmds.get_nowait())
            except queue.Empty:
                break
        if not cmds:
            return
        # при быстром движении слайдера оставляем только последнее значение
        latest_prop: dict[str, int] = {}
        for i, (name, arg) in enumerate(cmds):
            if name == "prop":
                latest_prop[arg[0]] = i
        for i, (name, arg) in enumerate(cmds):
            try:
                if name == "open":
                    self._open(*arg)
                elif name == "close":
                    self._release()
                    self.closed.emit()
                elif name == "prop" and latest_prop.get(arg[0]) == i:
                    self._set_prop(*arg)
                elif name == "auto":
                    self._set_auto(*arg)
                elif name == "read":
                    self.props_read.emit(self._read_all())
                elif name == "dialog":
                    if self._cap is not None:
                        self._cap.set(cv2.CAP_PROP_SETTINGS, 1)
                elif name == "snapshot":
                    if self.last_frame is None:
                        self.error.emit("Нет кадра для снимка")
                    else:
                        save_image(arg, self.last_frame)
                        self.snapshot_saved.emit(arg)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"{name}: {exc}")

    def _open(self, index: int, backend: int, width: int, height: int,
              fps: int, fourcc: str) -> None:
        self._release()
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            self.error.emit(f"Не удалось открыть камеру #{index}. "
                            "Возможно, она занята другой программой.")
            return
        # Порядок важен: сначала формат, потом размер и частота
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            # некоторые драйверы отдают первый кадр не сразу
            time.sleep(0.3)
            ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.error.emit(f"Камера #{index} не отдаёт видео в режиме "
                            f"{width}×{height} {fourcc} @ {fps}.")
            return
        self._cap = cap
        self._backend = backend
        self.last_frame = frame
        info = {
            "index": index,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "fps": cap.get(cv2.CAP_PROP_FPS) or fps,
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "backend": backend,
            "backend_name": cap.getBackendName(),
            "values": self._read_all(),
            "ranges": query_v4l2_ranges(index) if sys.platform.startswith("linux") else {},
        }
        self.opened.emit(info)

    def _read_all(self) -> dict:
        cap = self._cap
        if cap is None:
            return {}
        values = {p.key: cap.get(p.cv_id) for p in PROPS}
        for a in AUTO_BY_KEY.values():
            values[a.key] = is_auto_on(self._backend, cap.get(a.cv_id), a.key)
        return values

    def _set_prop(self, key: str, value: float) -> None:
        cap = self._cap
        prop = PROP_BY_KEY.get(key)
        if cap is None or prop is None:
            return
        ok = bool(cap.set(prop.cv_id, value))
        self.prop_result.emit(key, ok, cap.get(prop.cv_id))

    def _set_auto(self, key: str, enabled: bool) -> None:
        cap = self._cap
        a = AUTO_BY_KEY.get(key)
        if cap is None or a is None:
            return
        ok = bool(cap.set(a.cv_id, auto_value(self._backend, enabled, key)))
        self.prop_result.emit(key, ok, float(enabled))

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
