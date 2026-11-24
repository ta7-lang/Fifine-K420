"""Профили настроек (JSON)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_SETTINGS: dict = {
    "camera": {"device": "", "backend": 0, "width": 1920, "height": 1080,
               "fps": 30, "fourcc": "MJPG", "mirror": False, "rotate180": False},
    "image": {},          # {ключ параметра: значение}
    "auto": {"auto_exposure": True, "auto_wb": True},
    "mic": {},            # MicSettings.to_dict()
    "video": {},          # VideoSettings.to_dict()
    "audio": {},          # AudioSettings.to_dict()
    "output_dir": "",
}

BUILTIN_PROFILES: dict[str, dict] = {
    "Стрим 1080p30 (6 Мбит/с CBR)": {
        "camera": {"width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG"},
        "video": {"codec": "libx264", "rate_control": "cbr", "bitrate_kbps": 6000,
                  "preset": "veryfast", "gop_seconds": 2.0, "container": "mp4"},
        "audio": {"enabled": True, "codec": "aac", "bitrate_kbps": 160},
    },
    "Максимальное качество 1440p": {
        "camera": {"width": 2560, "height": 1440, "fps": 30, "fourcc": "MJPG"},
        "video": {"codec": "libx264", "rate_control": "cq", "quality": 18,
                  "bitrate_kbps": 16000, "preset": "fast", "gop_seconds": 2.0,
                  "container": "mkv"},
        "audio": {"enabled": True, "codec": "aac", "bitrate_kbps": 256},
    },
    "Видеозвонок 720p": {
        "camera": {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"},
        "video": {"codec": "libx264", "rate_control": "vbr", "bitrate_kbps": 2500,
                  "preset": "veryfast", "gop_seconds": 2.0, "container": "mp4"},
        "audio": {"enabled": True, "codec": "aac", "bitrate_kbps": 128},
    },
    "Слабый ПК 720p": {
        "camera": {"width": 1280, "height": 720, "fps": 25, "fourcc": "MJPG"},
        "video": {"codec": "libx264", "rate_control": "cbr", "bitrate_kbps": 3000,
                  "preset": "ultrafast", "gop_seconds": 2.0, "container": "mp4"},
        "audio": {"enabled": True, "codec": "aac", "bitrate_kbps": 128},
    },
}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ProfileStore:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.file = self.folder / "profiles.json"
        self.last_file = self.folder / "last_settings.json"

    def _read(self, path: Path) -> dict:
        try:
            data = json.loads(path.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, path: Path, data: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)

    def user_profiles(self) -> dict[str, dict]:
        return self._read(self.file)

    def all_names(self) -> list[str]:
        return list(BUILTIN_PROFILES) + [n for n in self.user_profiles()
                                         if n not in BUILTIN_PROFILES]

    def get(self, name: str) -> dict | None:
        if name in BUILTIN_PROFILES:
            return copy.deepcopy(BUILTIN_PROFILES[name])
        return self.user_profiles().get(name)

    def save(self, name: str, settings: dict) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Пустое имя профиля")
        if name in BUILTIN_PROFILES:
            raise ValueError("Нельзя перезаписать встроенный профиль")
        data = self.user_profiles()
        data[name] = settings
        self._write(self.file, data)

    def delete(self, name: str) -> bool:
        data = self.user_profiles()
        if name in data:
            del data[name]
            self._write(self.file, data)
            return True
        return False

    def load_last(self) -> dict:
        return deep_merge(DEFAULT_SETTINGS, self._read(self.last_file))

    def save_last(self, settings: dict) -> None:
        self._write(self.last_file, settings)

    def export_to(self, path: str | Path, settings: dict) -> None:
        Path(path).write_text(json.dumps(settings, ensure_ascii=False, indent=2), "utf-8")

    def import_from(self, path: str | Path) -> dict:
        data = json.loads(Path(path).read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Файл не является профилем")
        return data
