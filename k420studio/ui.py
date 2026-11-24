"""Главное окно приложения."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from . import APP_NAME, APP_VERSION
from .audio import (SAMPLE_RATES, SD_ERROR, AudioEngine, MicSettings,
                    list_input_devices, pick_k420_device)
from .camera_props import (AUTOS, K420_FPS, K420_RESOLUTIONS, PROPS, backend_choices, default_range,
                           is_k420_name)
from .camera_worker import CameraWorker
from .ffmpeg_tools import (AUDIO_CODECS, CONTAINERS, PRESETS, RATE_CONTROLS,
                           VIDEO_CODECS, AudioSettings, VideoSettings,
                           available_audio_codecs, available_video_codecs,
                           estimate_mb_per_minute, find_ffmpeg)
from .profiles import ProfileStore, deep_merge
from .recorder import Recorder, RecordingError, default_filename, unique_path

AUDIO_BITRATES = [64, 96, 128, 160, 192, 256, 320]


def video_input_names() -> list[str]:
    """Имена камер в системе (порядок обычно совпадает с индексами OpenCV)."""
    try:
        from PySide6.QtMultimedia import QMediaDevices

        return [d.description() for d in QMediaDevices.videoInputs()]
    except Exception:  # noqa: BLE001 — QtMultimedia может быть недоступен
        return []


class _Bridge(QObject):
    """Передаёт результат фоновой остановки записи в GUI-поток."""
    record_finished = Signal(str, str)  # путь, ошибка


class PropRow(QWidget):
    """Слайдер + поле ввода для одного UVC-параметра."""
    changed = Signal(str, int)

    def __init__(self, key: str, label: str, tooltip: str = "") -> None:
        super().__init__()
        self.key = key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setMinimumWidth(150)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.spin = QSpinBox()
        self.spin.setMinimumWidth(70)
        for w in (self.label, self.slider, self.spin):
            w.setToolTip(tooltip)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._mute = False
        self.default = 0

    def set_range(self, lo: int, hi: int, step: int, default: int) -> None:
        self._mute = True
        if lo > hi:
            lo, hi = hi, lo
        self.slider.setRange(lo, hi)
        self.spin.setRange(lo, hi)
        self.slider.setSingleStep(max(1, step))
        self.spin.setSingleStep(max(1, step))
        self.default = default
        self._mute = False

    def set_value(self, value: float, emit: bool = False) -> None:
        v = int(round(value))
        if v < self.slider.minimum() or v > self.slider.maximum():
            # реальное значение вне предполагаемого диапазона — расширяем диапазон
            self.set_range(min(v, self.slider.minimum()), max(v, self.slider.maximum()),
                           self.slider.singleStep(), self.default)
        self._mute = not emit
        self.slider.setValue(v)
        self.spin.setValue(v)
        self._mute = False

    def value(self) -> int:
        return self.spin.value()

    def _from_slider(self, v: int) -> None:
        if self.spin.value() != v:
            mute, self._mute = self._mute, True
            self.spin.setValue(v)
            self._mute = mute
        if not self._mute:
            self.changed.emit(self.key, v)

    def _from_spin(self, v: int) -> None:
        if self.slider.value() != v:
            mute, self._mute = self._mute, True
            self.slider.setValue(v)
            self._mute = mute
        if not self._mute:
            self.changed.emit(self.key, v)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1360, 820)

        cfg_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation) or str(Path.home() / ".k420studio")
        self.store = ProfileStore(cfg_dir)
        self.settings = self.store.load_last()

        self.ffmpeg = find_ffmpeg()
        self.audio = AudioEngine()
        self.recorder: Recorder | None = None
        self._stopping = False
        self._bridge = _Bridge()
        self._bridge.record_finished.connect(self._on_record_finished)
        self._last_image: QImage | None = None
        self._cam_open = False

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.opened.connect(self._on_opened)
        self.worker.props_read.connect(self._on_props_read)
        self.worker.prop_result.connect(self._on_prop_result)
        self.worker.error.connect(self._on_error)
        self.worker.closed.connect(lambda: self._set_cam_open(False))
        self.worker.snapshot_saved.connect(
            lambda p: self.statusBar().showMessage(f"Снимок сохранён: {p}", 6000))
        self.worker.fps_measured.connect(
            lambda f: self.lbl_fps.setText(f"{f:.1f} к/с"))

        self._build_ui()
        self._apply_settings_to_ui(self.settings)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(80)

        QTimer.singleShot(100, self._initial_start)

    # =================================================================== UI
    def _build_ui(self) -> None:
        split = QSplitter()
        self.setCentralWidget(split)

        # --- левая часть: превью и кнопки
        left = QWidget()
        lv = QVBoxLayout(left)
        self.preview = QLabel("Камера не подключена")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setStyleSheet("background:#000;color:#888;font-size:16px;")
        lv.addWidget(self.preview, 1)

        bar = QHBoxLayout()
        self.btn_rec = QPushButton("●  Начать запись")
        self.btn_rec.setMinimumHeight(38)
        self.btn_rec.clicked.connect(self._toggle_record)
        self.btn_snap = QPushButton("Снимок")
        self.btn_snap.setMinimumHeight(38)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_folder = QPushButton("Папка записей")
        self.btn_folder.setMinimumHeight(38)
        self.btn_folder.clicked.connect(self._open_folder)
        self.lbl_rec = QLabel("")
        self.lbl_rec.setStyleSheet("color:#ff5555;font-weight:bold;font-size:15px;")
        self.lbl_fps = QLabel("— к/с")
        for w in (self.btn_rec, self.btn_snap, self.btn_folder):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.lbl_rec)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_fps)
        lv.addLayout(bar)
        split.addWidget(left)

        # --- правая часть: вкладки
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(440)
        self.tabs.addTab(self._tab_camera(), "Камера")
        self.tabs.addTab(self._scroll(self._tab_image()), "Изображение")
        self.tabs.addTab(self._tab_mic(), "Микрофон")
        self.tabs.addTab(self._scroll(self._tab_record()), "Запись")
        self.tabs.addTab(self._tab_profiles(), "Профили")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        if not self.ffmpeg:
            self.btn_rec.setEnabled(False)
            self.btn_rec.setToolTip("FFmpeg не найден — запись недоступна")

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(w)
        return sa

    def _tab_camera(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        form = QFormLayout()

        row = QHBoxLayout()
        self.cb_device = QComboBox()
        btn_refresh = QPushButton("⟳")
        btn_refresh.setFixedWidth(36)
        btn_refresh.setToolTip("Обновить список камер")
        btn_refresh.clicked.connect(self._refresh_cameras)
        row.addWidget(self.cb_device, 1)
        row.addWidget(btn_refresh)
        form.addRow("Камера:", row)

        self.cb_backend = QComboBox()
        for name, val in backend_choices():
            self.cb_backend.addItem(name, val)
        form.addRow("Драйвер захвата:", self.cb_backend)

        self.cb_res = QComboBox()
        for wd, ht in K420_RESOLUTIONS:
            tag = {1440: " (2K)", 1080: " (Full HD)", 720: " (HD)", 480: " (VGA)"}.get(ht, "")
            self.cb_res.addItem(f"{wd} × {ht}{tag}", f"{wd}x{ht}")
        form.addRow("Разрешение:", self.cb_res)

        self.cb_fps = QComboBox()
        for f in K420_FPS:
            self.cb_fps.addItem(f"{f} к/с", f)
        form.addRow("Частота кадров:", self.cb_fps)

        self.cb_fourcc = QComboBox()
        self.cb_fourcc.addItem("MJPG — сжатый (нужен для 2K/1080p@30)", "MJPG")
        self.cb_fourcc.addItem("YUY2 — несжатый (меньше артефактов, ниже fps)", "YUY2")
        form.addRow("Формат потока:", self.cb_fourcc)

        self.chk_mirror = QCheckBox("Зеркально (по горизонтали)")
        self.chk_mirror.toggled.connect(lambda on: setattr(self.worker, "mirror", on))
        self.chk_rotate = QCheckBox("Повернуть на 180°")
        self.chk_rotate.toggled.connect(lambda on: setattr(self.worker, "rotate180", on))
        form.addRow("", self.chk_mirror)
        form.addRow("", self.chk_rotate)
        v.addLayout(form)

        hb = QHBoxLayout()
        self.btn_apply_cam = QPushButton("Применить / переподключить")
        self.btn_apply_cam.clicked.connect(self._open_camera)
        btn_close = QPushButton("Отключить")
        btn_close.clicked.connect(self.worker.close_camera)
        hb.addWidget(self.btn_apply_cam, 1)
        hb.addWidget(btn_close)
        v.addLayout(hb)

        self.lbl_cam_info = QLabel("")
        self.lbl_cam_info.setWordWrap(True)
        self.lbl_cam_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.lbl_cam_info)

        note = QLabel(
            "<small>Fifine K420: сенсор 2K (2560×1440 @ 30 к/с), фиксированный фокус, "
            "встроенный микрофон. Для 1440p и 1080p при 30 к/с используйте формат MJPG. "
            "Сам поток UVC-камеры не имеет настраиваемого битрейта — битрейт задаётся "
            "при кодировании записи (вкладка «Запись»).</small>")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        self._refresh_cameras()
        return w

    def _tab_image(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.auto_checks: dict[str, QCheckBox] = {}
        self.prop_rows: dict[str, PropRow] = {}
        autos_for = {a.controls: a for a in AUTOS}

        box = QGroupBox("Параметры изображения (UVC)")
        bl = QVBoxLayout(box)
        for p in PROPS:
            a = autos_for.get(p.key)
            if a is not None:
                chk = QCheckBox(a.label)
                chk.toggled.connect(lambda on, k=a.key: self._on_auto_toggled(k, on))
                self.auto_checks[a.key] = chk
                bl.addWidget(chk)
            row = PropRow(p.key, p.label, p.tooltip)
            lo, hi, d = default_range(p, self._backend())
            row.set_range(lo, hi, p.step, d)
            row.set_value(d)
            row.changed.connect(self._on_prop_changed)
            self.prop_rows[p.key] = row
            bl.addWidget(row)
        v.addWidget(box)

        grid = QGridLayout()
        b1 = QPushButton("Считать с камеры")
        b1.clicked.connect(self.worker.read_props)
        b2 = QPushButton("Сбросить по умолчанию")
        b2.clicked.connect(self._reset_props)
        self.btn_driver = QPushButton("Окно настроек драйвера…")
        self.btn_driver.setToolTip("Системное окно DirectShow со всеми параметрами камеры, "
                                   "включая частоту электросети (50/60 Гц) против мерцания.")
        self.btn_driver.clicked.connect(self.worker.driver_dialog)
        self.btn_driver.setVisible(sys.platform == "win32")
        grid.addWidget(b1, 0, 0)
        grid.addWidget(b2, 0, 1)
        grid.addWidget(self.btn_driver, 1, 0, 1, 2)
        v.addLayout(grid)

        hint = QLabel("<small>Значения применяются сразу. Если параметр не поддерживается "
                      "драйвером, его подпись станет серой. Диапазоны зависят от драйвера; "
                      "в Linux они считываются через v4l2-ctl.</small>")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)
        return w

    def _tab_mic(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        form = QFormLayout()
        row = QHBoxLayout()
        self.cb_mic = QComboBox()
        btn = QPushButton("⟳")
        btn.setFixedWidth(36)
        btn.clicked.connect(self._refresh_mics)
        row.addWidget(self.cb_mic, 1)
        row.addWidget(btn)
        form.addRow("Микрофон:", row)

        self.cb_sr = QComboBox()
        for sr in SAMPLE_RATES:
            self.cb_sr.addItem(f"{sr} Гц", sr)
        form.addRow("Частота дискретизации:", self.cb_sr)

        self.cb_ch = QComboBox()
        self.cb_ch.addItem("Моно", 1)
        self.cb_ch.addItem("Стерео", 2)
        form.addRow("Каналы:", self.cb_ch)

        self.sl_gain = QSlider(Qt.Orientation.Horizontal)
        self.sl_gain.setRange(-20, 30)
        self.lbl_gain = QLabel("0 дБ")
        self.sl_gain.valueChanged.connect(self._on_mic_live)
        g = QHBoxLayout()
        g.addWidget(self.sl_gain, 1)
        g.addWidget(self.lbl_gain)
        form.addRow("Усиление:", g)

        self.sl_gate = QSlider(Qt.Orientation.Horizontal)
        self.sl_gate.setRange(-90, -20)
        self.lbl_gate = QLabel("выкл")
        self.sl_gate.valueChanged.connect(self._on_mic_live)
        gg = QHBoxLayout()
        gg.addWidget(self.sl_gate, 1)
        gg.addWidget(self.lbl_gate)
        form.addRow("Шумоподавитель:", gg)

        self.chk_monitor = QCheckBox("Слушать себя (в наушниках)")
        form.addRow("", self.chk_monitor)
        v.addLayout(form)

        btn_apply = QPushButton("Применить / перезапустить микрофон")
        btn_apply.clicked.connect(self._start_mic)
        v.addWidget(btn_apply)

        self.meter = QProgressBar()
        self.meter.setRange(-60, 0)
        self.meter.setFormat("%v дБFS")
        self.meter.setTextVisible(True)
        v.addWidget(QLabel("Уровень сигнала:"))
        v.addWidget(self.meter)
        self.lbl_clip = QLabel("")
        v.addWidget(self.lbl_clip)

        hint = QLabel("<small>Рекомендуемый уровень речи: от −18 до −6 дБFS. Если индикатор "
                      "краснеет — уменьшите усиление. Шумоподавитель глушит звук тише "
                      "порога (полезно против гула вентиляторов). Аппаратная громкость "
                      "микрофона K420 настраивается в системе: Параметры звука → Ввод.</small>")
        hint.setWordWrap(True)
        v.addWidget(hint)
        if not self.audio.available:
            v.addWidget(QLabel(f"<b>Звук недоступен:</b> {SD_ERROR}"))
        v.addStretch(1)
        self._refresh_mics()
        return w

    def _tab_record(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        box_out = QGroupBox("Файл")
        fo = QFormLayout(box_out)
        row = QHBoxLayout()
        self.lbl_outdir = QLabel()
        self.lbl_outdir.setWordWrap(True)
        b = QPushButton("Выбрать…")
        b.clicked.connect(self._choose_outdir)
        row.addWidget(self.lbl_outdir, 1)
        row.addWidget(b)
        fo.addRow("Папка:", row)
        self.cb_container = QComboBox()
        for c in CONTAINERS:
            self.cb_container.addItem(c.upper(), c)
        self.cb_container.setToolTip("MKV не повреждается при сбое; MP4 — максимальная совместимость.")
        fo.addRow("Контейнер:", self.cb_container)
        v.addWidget(box_out)

        box_v = QGroupBox("Видео")
        fv = QFormLayout(box_v)
        self.cb_vcodec = QComboBox()
        codecs = available_video_codecs(self.ffmpeg)
        for c in codecs:
            self.cb_vcodec.addItem(VIDEO_CODECS[c], c)
        if not codecs:
            self.cb_vcodec.addItem("FFmpeg не найден", "")
        fv.addRow("Кодек:", self.cb_vcodec)

        self.cb_rc = QComboBox()
        for k, label in RATE_CONTROLS.items():
            self.cb_rc.addItem(label, k)
        fv.addRow("Режим битрейта:", self.cb_rc)

        self.sp_bitrate = QSpinBox()
        self.sp_bitrate.setRange(300, 80000)
        self.sp_bitrate.setSingleStep(500)
        self.sp_bitrate.setSuffix(" кбит/с")
        fv.addRow("Битрейт видео:", self.sp_bitrate)

        self.sp_quality = QSpinBox()
        self.sp_quality.setRange(0, 51)
        self.sp_quality.setToolTip("0 — без потерь, 18 — визуально без потерь, 23 — по умолчанию, "
                                   "28+ — заметное сжатие")
        fv.addRow("Качество (CRF/CQ):", self.sp_quality)

        self.cb_preset = QComboBox()
        for p in PRESETS:
            self.cb_preset.addItem(p, p)
        self.cb_preset.setToolTip("Быстрее — меньше нагрузка на процессор, медленнее — "
                                  "лучше качество при том же битрейте.")
        fv.addRow("Пресет скорости:", self.cb_preset)

        self.sp_gop = QDoubleSpinBox()
        self.sp_gop.setRange(0.5, 10.0)
        self.sp_gop.setSingleStep(0.5)
        self.sp_gop.setSuffix(" с")
        fv.addRow("Интервал ключевых кадров:", self.sp_gop)
        v.addWidget(box_v)

        box_a = QGroupBox("Звук")
        fa = QFormLayout(box_a)
        self.chk_audio = QCheckBox("Записывать звук с микрофона")
        fa.addRow(self.chk_audio)
        self.cb_acodec = QComboBox()
        for c in available_audio_codecs(self.ffmpeg) or ["aac"]:
            self.cb_acodec.addItem(AUDIO_CODECS.get(c, c), c)
        fa.addRow("Кодек:", self.cb_acodec)
        self.cb_abitrate = QComboBox()
        for br in AUDIO_BITRATES:
            self.cb_abitrate.addItem(f"{br} кбит/с", br)
        fa.addRow("Битрейт звука:", self.cb_abitrate)
        v.addWidget(box_a)

        self.lbl_estimate = QLabel()
        v.addWidget(self.lbl_estimate)
        # сигналы подключаем после создания всех виджетов вкладки
        self.cb_rc.currentIndexChanged.connect(self._update_rc_widgets)
        self.sp_bitrate.valueChanged.connect(self._update_estimate)
        self.cb_abitrate.currentIndexChanged.connect(self._update_estimate)
        self.chk_audio.toggled.connect(self._update_estimate)
        v.addWidget(QLabel(f"<small>FFmpeg: {self.ffmpeg or 'не найден'}</small>"))
        v.addStretch(1)
        return w

    def _tab_profiles(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.lst_profiles = QListWidget()
        self.lst_profiles.itemDoubleClicked.connect(lambda _i: self._load_profile())
        v.addWidget(self.lst_profiles, 1)
        grid = QGridLayout()
        buttons = [("Загрузить", self._load_profile), ("Сохранить как…", self._save_profile),
                   ("Удалить", self._delete_profile), ("Экспорт в файл…", self._export_profile),
                   ("Импорт из файла…", self._import_profile)]
        for i, (text, fn) in enumerate(buttons):
            b = QPushButton(text)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
        v.addLayout(grid)
        v.addWidget(QLabel("<small>Профиль хранит режим камеры, параметры изображения, "
                           "настройки микрофона и записи. Двойной щелчок — загрузить.</small>"))
        self._refresh_profiles()
        return w

    # ============================================================ helpers
    def _backend(self) -> int:
        if hasattr(self, "cb_backend") and self.cb_backend.count():
            return int(self.cb_backend.currentData())
        return backend_choices()[0][1]

    def _resolution(self) -> tuple[int, int]:
        try:
            wd, ht = str(self.cb_res.currentData()).split("x")
            return int(wd), int(ht)
        except ValueError:
            return 1920, 1080

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _refresh_cameras(self) -> None:
        current = self.cb_device.currentText()
        names = video_input_names()
        self.cb_device.clear()
        # если имена получить не удалось — предлагаем первые 4 индекса
        labels = names or [f"Камера #{i}" for i in range(4)]
        for i, label in enumerate(labels):
            self.cb_device.addItem(f"{i}: {label}", i)
        if current:
            idx = self.cb_device.findText(current)
            if idx >= 0:
                self.cb_device.setCurrentIndex(idx)
                return
        for i, n in enumerate(names):
            if is_k420_name(n):
                self.cb_device.setCurrentIndex(i)
                break

    def _refresh_mics(self) -> None:
        current = self.cb_mic.currentText()
        self.cb_mic.clear()
        self.cb_mic.addItem("Системный по умолчанию", None)
        devices = list_input_devices()
        for idx, name, _ch, _sr in devices:
            self.cb_mic.addItem(name, idx)
        if current and self.cb_mic.findText(current) >= 0:
            self.cb_mic.setCurrentIndex(self.cb_mic.findText(current))
            return
        k = pick_k420_device(devices)
        if k is not None:
            self._select_data(self.cb_mic, k)

    def _refresh_profiles(self) -> None:
        self.lst_profiles.clear()
        self.lst_profiles.addItems(self.store.all_names())

    # ====================================================== settings <-> UI
    def _collect_settings(self) -> dict:
        res = self._resolution()
        return {
            "camera": {"device": self.cb_device.currentText(),
                       "backend": self.cb_backend.currentIndex(),
                       "width": res[0], "height": res[1],
                       "fps": self.cb_fps.currentData(),
                       "fourcc": self.cb_fourcc.currentData(),
                       "mirror": self.chk_mirror.isChecked(),
                       "rotate180": self.chk_rotate.isChecked()},
            "image": {k: r.value() for k, r in self.prop_rows.items()},
            "auto": {k: c.isChecked() for k, c in self.auto_checks.items()},
            "mic": self._mic_settings().to_dict(),
            "video": self._video_settings().to_dict(),
            "audio": self._audio_settings().to_dict(),
            "output_dir": self._output_dir(),
        }

    def _apply_settings_to_ui(self, s: dict) -> None:
        cam = s.get("camera", {})
        if cam.get("device"):
            i = self.cb_device.findText(cam["device"])
            if i >= 0:
                self.cb_device.setCurrentIndex(i)
        if isinstance(cam.get("backend"), int) and 0 <= cam["backend"] < self.cb_backend.count():
            self.cb_backend.setCurrentIndex(cam["backend"])
        if "width" in cam and "height" in cam:
            self._select_data(self.cb_res, f"{cam['width']}x{cam['height']}")
        if "fps" in cam:
            self._select_data(self.cb_fps, cam["fps"])
        if "fourcc" in cam:
            self._select_data(self.cb_fourcc, cam["fourcc"])
        if "mirror" in cam:
            self.chk_mirror.setChecked(bool(cam["mirror"]))
        if "rotate180" in cam:
            self.chk_rotate.setChecked(bool(cam["rotate180"]))

        for k, val in s.get("image", {}).items():
            if k in self.prop_rows:
                self.prop_rows[k].set_value(val)
        for k, on in s.get("auto", {}).items():
            if k in self.auto_checks:
                chk = self.auto_checks[k]
                chk.blockSignals(True)
                chk.setChecked(bool(on))
                chk.blockSignals(False)
                self._update_auto_enabled(k, bool(on))

        mic = MicSettings.from_dict(s.get("mic", {}))
        if mic.device:
            i = self.cb_mic.findText(mic.device)
            if i >= 0:
                self.cb_mic.setCurrentIndex(i)
        self._select_data(self.cb_sr, mic.samplerate)
        self._select_data(self.cb_ch, mic.channels)
        self.sl_gain.setValue(int(mic.gain_db))
        self.sl_gate.setValue(int(mic.gate_db))
        self.chk_monitor.setChecked(mic.monitor)

        vs = VideoSettings.from_dict(s.get("video", {}))
        self._select_data(self.cb_vcodec, vs.codec)
        self._select_data(self.cb_rc, vs.rate_control)
        self.sp_bitrate.setValue(vs.bitrate_kbps)
        self.sp_quality.setValue(vs.quality)
        self._select_data(self.cb_preset, vs.preset)
        self.sp_gop.setValue(vs.gop_seconds)
        self._select_data(self.cb_container, vs.container)

        au = AudioSettings.from_dict(s.get("audio", {}))
        self.chk_audio.setChecked(au.enabled)
        self._select_data(self.cb_acodec, au.codec)
        self._select_data(self.cb_abitrate, au.bitrate_kbps)

        out = s.get("output_dir") or self._default_outdir()
        self.lbl_outdir.setText(out)
        self._update_rc_widgets()
        self._update_estimate()

    def _mic_settings(self) -> MicSettings:
        return MicSettings(device=self.cb_mic.currentText() if self.cb_mic.currentIndex() > 0 else "",
                           samplerate=int(self.cb_sr.currentData() or 48000),
                           channels=int(self.cb_ch.currentData() or 1),
                           gain_db=float(self.sl_gain.value()),
                           gate_db=float(self.sl_gate.value()),
                           monitor=self.chk_monitor.isChecked())

    def _video_settings(self) -> VideoSettings:
        return VideoSettings(codec=self.cb_vcodec.currentData() or "libx264",
                             rate_control=self.cb_rc.currentData() or "cbr",
                             bitrate_kbps=self.sp_bitrate.value(),
                             quality=self.sp_quality.value(),
                             preset=self.cb_preset.currentData() or "veryfast",
                             gop_seconds=self.sp_gop.value(),
                             container=self.cb_container.currentData() or "mp4")

    def _audio_settings(self) -> AudioSettings:
        return AudioSettings(enabled=self.chk_audio.isChecked(),
                             codec=self.cb_acodec.currentData() or "aac",
                             bitrate_kbps=int(self.cb_abitrate.currentData() or 160))

    @staticmethod
    def _default_outdir() -> str:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation) \
            or str(Path.home())
        return str(Path(base) / "K420")

    def _output_dir(self) -> str:
        return self.lbl_outdir.text() or self._default_outdir()

    # ============================================================ camera
    def _initial_start(self) -> None:
        self._open_camera()
        self._start_mic(silent=True)

    def _open_camera(self) -> None:
        if self.recorder and self.recorder.active:
            QMessageBox.information(self, APP_NAME, "Остановите запись перед сменой режима камеры.")
            return
        res = self._resolution()
        self.lbl_cam_info.setText("Подключение…")
        self.worker.open_camera(int(self.cb_device.currentData() or 0), self._backend(),
                                res[0], res[1], int(self.cb_fps.currentData() or 30),
                                self.cb_fourcc.currentData() or "MJPG")

    def _set_cam_open(self, on: bool) -> None:
        self._cam_open = on
        if not on:
            self.preview.clear()
            self.preview.setText("Камера не подключена")
            self.lbl_fps.setText("— к/с")
            self.lbl_cam_info.setText("Камера отключена")

    def _on_opened(self, info: dict) -> None:
        self._set_cam_open(True)
        req = self._resolution()
        text = (f"Открыто: {info['width']}×{info['height']} @ {info['fps']:.0f} к/с, "
                f"формат {info['fourcc']}, драйвер {info['backend_name']}")
        if (info["width"], info["height"]) != req:
            text += ("<br><span style='color:#e0a030'>Камера не поддержала запрошенное "
                     "разрешение — выбрано ближайшее.</span>")
        self.lbl_cam_info.setText(text)
        for key, (lo, hi, step, default) in info.get("ranges", {}).items():
            if key in self.prop_rows:
                self.prop_rows[key].set_range(lo, hi, step, default)
        # применяем сохранённые параметры изображения к камере
        autos = {k: c.isChecked() for k, c in self.auto_checks.items()}
        for k, on in autos.items():
            self.worker.set_auto(k, on)
        manual_blocked = {a.controls for a in AUTOS if autos.get(a.key)}
        saved = self.settings.get("image", {})
        for key in self.prop_rows:
            if key in saved and key not in manual_blocked:
                self.worker.set_prop(key, saved[key])
        self.worker.read_props()

    def _on_props_read(self, values: dict) -> None:
        for k, v in values.items():
            if k in self.prop_rows:
                self.prop_rows[k].set_value(v)
            elif k in self.auto_checks:
                chk = self.auto_checks[k]
                chk.blockSignals(True)
                chk.setChecked(bool(v))
                chk.blockSignals(False)
                self._update_auto_enabled(k, bool(v))

    def _on_prop_changed(self, key: str, value: int) -> None:
        self.settings.setdefault("image", {})[key] = value
        self.worker.set_prop(key, value)

    def _on_prop_result(self, key: str, ok: bool, actual: float) -> None:
        row = self.prop_rows.get(key)
        if row is not None:
            row.label.setStyleSheet("" if ok else "color:#777;")
            if not ok:
                row.label.setToolTip("Драйвер не поддерживает этот параметр")

    def _on_auto_toggled(self, key: str, on: bool) -> None:
        self._update_auto_enabled(key, on)
        self.worker.set_auto(key, on)
        if not on:
            # передаём ручное значение сразу после выключения авто-режима
            a = next(a for a in AUTOS if a.key == key)
            row = self.prop_rows[a.controls]
            self.worker.set_prop(a.controls, row.value())

    def _update_auto_enabled(self, key: str, on: bool) -> None:
        a = next((a for a in AUTOS if a.key == key), None)
        if a and a.controls in self.prop_rows:
            self.prop_rows[a.controls].setEnabled(not on)

    def _reset_props(self) -> None:
        for key, row in self.prop_rows.items():
            row.set_value(row.default, emit=True)
            self.settings.setdefault("image", {})[key] = row.default
        for chk in self.auto_checks.values():
            chk.setChecked(True)

    def _on_frame(self, img: QImage) -> None:
        self._last_image = img
        self._show_image()
        self.worker.preview_consumed()

    def _show_image(self) -> None:
        if self._last_image is None:
            return
        pm = QPixmap.fromImage(self._last_image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pm)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 10000)
        if "открыть камеру" in msg or "не отдаёт" in msg or "перестала" in msg:
            self._set_cam_open(False)
            self.lbl_cam_info.setText(msg)

    # ============================================================ mic
    def _start_mic(self, silent: bool = False) -> None:
        if not self.audio.available:
            return
        ms = self._mic_settings()
        try:
            self.audio.start(self.cb_mic.currentData(), ms)
            self.statusBar().showMessage(f"Микрофон: {self.cb_mic.currentText()}", 4000)
        except Exception as exc:  # noqa: BLE001
            if not silent:
                QMessageBox.warning(self, APP_NAME, f"Не удалось открыть микрофон:\n{exc}")
            else:
                self.statusBar().showMessage(f"Микрофон не запущен: {exc}", 8000)

    def _on_mic_live(self) -> None:
        g = self.sl_gain.value()
        t = self.sl_gate.value()
        self.lbl_gain.setText(f"{g:+d} дБ")
        self.lbl_gate.setText("выкл" if t <= -90 else f"{t} дБ")
        self.audio.update_live(float(g), float(t))

    # ============================================================ record
    def _update_rc_widgets(self) -> None:
        cq = self.cb_rc.currentData() == "cq"
        self.sp_quality.setEnabled(cq)
        self.sp_bitrate.setEnabled(not cq)
        self._update_estimate()

    def _update_estimate(self) -> None:
        if not hasattr(self, "lbl_estimate"):
            return
        if self.cb_rc.currentData() == "cq":
            self.lbl_estimate.setText("Размер файла зависит от сцены (режим постоянного качества).")
            return
        a = int(self.cb_abitrate.currentData() or 0) if self.chk_audio.isChecked() else 0
        mb = estimate_mb_per_minute(self.sp_bitrate.value(), a)
        self.lbl_estimate.setText(f"Примерный размер: {mb:.0f} МБ/мин, {mb * 60 / 1024:.1f} ГБ/час")

    def _choose_outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Папка для записей", self._output_dir())
        if d:
            self.lbl_outdir.setText(d)

    def _toggle_record(self) -> None:
        if self._stopping:
            return
        if self.recorder and self.recorder.active:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if not self.ffmpeg:
            QMessageBox.warning(self, APP_NAME, "FFmpeg не найден.")
            return
        if not self._cam_open or self.worker.frame_size == (0, 0):
            QMessageBox.warning(self, APP_NAME, "Сначала подключите камеру.")
            return
        vs = self._video_settings()
        au = self._audio_settings()
        if au.enabled and not self.audio.running:
            r = QMessageBox.question(self, APP_NAME,
                                     "Микрофон не запущен. Записать видео без звука?")
            if r != QMessageBox.StandardButton.Yes:
                return
            au.enabled = False
        folder = self._output_dir()
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Нет доступа к папке:\n{exc}")
            return
        path = unique_path(folder, default_filename(vs.container))
        w, h = self.worker.frame_size
        fps = int(self.cb_fps.currentData() or 30)
        rec = Recorder(self.ffmpeg)
        try:
            rec.start(path, w, h, fps, vs, au, self.audio if au.enabled else None)
        except RecordingError as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        self.recorder = rec
        self.worker.set_recorder(rec)
        self.btn_rec.setText("■  Остановить запись")
        self.btn_rec.setStyleSheet("background:#a02020;color:white;font-weight:bold;")
        self._lock_controls(True)
        self.statusBar().showMessage(f"Запись: {path}")

    def _stop_record(self) -> None:
        rec = self.recorder
        if rec is None:
            return
        self.worker.set_recorder(None)
        self._stopping = True
        self.btn_rec.setEnabled(False)
        self.btn_rec.setText("Сохранение…")

        def job() -> None:
            try:
                out = rec.stop()
                self._bridge.record_finished.emit(str(out), rec.warning)
            except Exception as exc:  # noqa: BLE001
                self._bridge.record_finished.emit("", str(exc))

        threading.Thread(target=job, daemon=True, name="k420-stop").start()

    def _on_record_finished(self, path: str, err: str) -> None:
        self._stopping = False
        self.recorder = None
        self.btn_rec.setEnabled(bool(self.ffmpeg))
        self.btn_rec.setText("●  Начать запись")
        self.btn_rec.setStyleSheet("")
        self.lbl_rec.setText("")
        self._lock_controls(False)
        if path and not err:
            self.statusBar().showMessage(f"Сохранено: {path}", 10000)
        elif path:
            self.statusBar().showMessage(f"Сохранено с предупреждением: {path}", 10000)
            QMessageBox.warning(self, APP_NAME, err)
        else:
            QMessageBox.critical(self, APP_NAME, f"Запись не сохранена:\n{err}")

    def _lock_controls(self, lock: bool) -> None:
        for w in (self.cb_res, self.cb_fps, self.cb_fourcc, self.cb_device, self.cb_backend,
                  self.btn_apply_cam, self.cb_vcodec, self.cb_rc, self.sp_bitrate,
                  self.sp_quality, self.cb_preset, self.sp_gop, self.cb_container,
                  self.chk_audio, self.cb_acodec, self.cb_abitrate, self.cb_mic,
                  self.cb_sr, self.cb_ch):
            w.setEnabled(not lock)
        if not lock:
            self._update_rc_widgets()

    def _snapshot(self) -> None:
        if not self._cam_open:
            return
        folder = self._output_dir()
        path = unique_path(folder, default_filename("png").replace(".png", "_photo.png"))
        self.worker.snapshot(str(path))

    def _open_folder(self) -> None:
        folder = self._output_dir()
        Path(folder).mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])  # noqa: S603,S607
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ============================================================ profiles
    def _selected_profile(self) -> str | None:
        item = self.lst_profiles.currentItem()
        return item.text() if item else None

    def _apply_profile(self, data: dict) -> None:
        merged = deep_merge(self._collect_settings(), data)
        self.settings = merged
        self._apply_settings_to_ui(merged)
        if not (self.recorder and self.recorder.active):
            self._open_camera()

    def _load_profile(self) -> None:
        name = self._selected_profile()
        if not name:
            return
        data = self.store.get(name)
        if data is None:
            return
        self._apply_profile(data)
        self.statusBar().showMessage(f"Профиль «{name}» загружен", 5000)

    def _save_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Сохранить профиль", "Название профиля:")
        if not ok or not name.strip():
            return
        try:
            self.store.save(name, self._collect_settings())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self._refresh_profiles()

    def _delete_profile(self) -> None:
        name = self._selected_profile()
        if name and not self.store.delete(name):
            QMessageBox.information(self, APP_NAME, "Встроенные профили удалить нельзя.")
        self._refresh_profiles()

    def _export_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт профиля", "k420_profile.json",
                                              "JSON (*.json)")
        if path:
            try:
                self.store.export_to(path, self._collect_settings())
            except OSError as exc:
                QMessageBox.warning(self, APP_NAME, str(exc))

    def _import_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Импорт профиля", "", "JSON (*.json)")
        if not path:
            return
        try:
            self._apply_profile(self.store.import_from(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Не удалось импортировать:\n{exc}")

    # ============================================================ misc
    def _tick(self) -> None:
        if self.audio.running:
            lvl = max(-60, int(self.audio.level_db))
            self.meter.setValue(lvl)
            color = "#d03030" if self.audio.peak_db > -1 else (
                "#d0a020" if self.audio.peak_db > -6 else "#30a050")
            self.meter.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}")
            if self.audio.clipped:
                self.lbl_clip.setText("<span style='color:#d03030'>Были перегрузки (клиппинг) — "
                                      "уменьшите усиление</span>")
                self.audio.clipped = False
        else:
            self.meter.setValue(-60)
        rec = self.recorder
        if rec and rec.active:
            t = int(rec.elapsed)
            extra = f"  (пропущено кадров: {rec.dropped})" if rec.dropped else ""
            self.lbl_rec.setText(f"● REC {t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}{extra}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._show_image()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.recorder and self.recorder.active:
            r = QMessageBox.question(self, APP_NAME, "Идёт запись. Остановить и выйти?")
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.set_recorder(None)
            try:
                self.recorder.stop()
            except RecordingError as exc:
                QMessageBox.warning(self, APP_NAME, str(exc))
        try:
            self.store.save_last(self._collect_settings())
        except OSError:
            pass
        self.timer.stop()
        self.audio.stop()
        self.worker.stop()
        super().closeEvent(event)
