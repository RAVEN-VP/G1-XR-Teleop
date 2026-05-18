import csv
import json
from datetime import datetime, timezone
from pathlib import Path


class TeleopLogger:
    def __init__(self, root_dir="logs", session_name=None, metadata=None, fieldnames=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"

        self.session_dir = Path(root_dir) / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.session_dir / "metadata.json"
        self.trajectory_path = self.session_dir / "trajectory.csv"

        self.fieldnames = fieldnames or []
        self._csv_file = None
        self._writer = None

        metadata = metadata or {}
        metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("session_name", self.session_name)

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._open_csv()

    def _open_csv(self):
        self._csv_file = self.trajectory_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()

    def write_row(self, row):
        if self._writer is None:
            raise RuntimeError("Logger is closed.")

        self._writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def flatten_vec(prefix, values):
    if values is None:
        return {
            f"{prefix}_x": "",
            f"{prefix}_y": "",
            f"{prefix}_z": "",
        }

    return {
        f"{prefix}_x": values[0],
        f"{prefix}_y": values[1],
        f"{prefix}_z": values[2],
    }


def make_joint_fields(prefix, joint_indices):
    fields = []

    for joint in joint_indices:
        fields.append(f"{prefix}_{joint}_q")

    return fields


def flatten_joint_values(prefix, joint_indices, values_by_joint):
    row = {}

    for joint in joint_indices:
        row[f"{prefix}_{joint}_q"] = values_by_joint.get(joint, "")

    return row
