import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OBSERVATION_PREFIXES = [
    "left_",
    "right_",
    "left_delta_",
    "right_delta_",
    "actual_",
]

DEFAULT_ACTION_PREFIXES = [
    "cmd_",
]


def load_metadata(session_dir: Path):
    metadata_path = session_dir / "metadata.json"

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_columns(df, prefixes):
    columns = []

    for column in df.columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                columns.append(column)
                break

    return columns


def dataframe_to_float_array(df, columns):
    if not columns:
        return np.empty((len(df), 0), dtype=np.float32)

    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(0.0)

    return numeric.to_numpy(dtype=np.float32)


def convert_session(session_dir: Path, output_dir: Path):
    trajectory_path = session_dir / "trajectory.csv"

    if not trajectory_path.exists():
        print(f"Skipping {session_dir}: no trajectory.csv")
        return None

    df = pd.read_csv(trajectory_path)

    if df.empty:
        print(f"Skipping {session_dir}: empty trajectory.csv")
        return None

    metadata = load_metadata(session_dir)

    observation_columns = select_columns(df, DEFAULT_OBSERVATION_PREFIXES)
    action_columns = select_columns(df, DEFAULT_ACTION_PREFIXES)

    observations = dataframe_to_float_array(df, observation_columns)
    actions = dataframe_to_float_array(df, action_columns)

    timestamps = (
        pd.to_numeric(df["timestamp"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.arange(len(df), dtype=np.float64)
    )

    deadman = (
        pd.to_numeric(df["deadman"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
        if "deadman" in df.columns
        else np.zeros(len(df), dtype=np.int8)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_dir.name}.npz"

    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        deadman=deadman,
        observation_columns=np.array(observation_columns),
        action_columns=np.array(action_columns),
        metadata=json.dumps(metadata),
        source_session=str(session_dir),
    )

    print(f"Converted {session_dir} -> {output_path}")
    print(f"  observations: {observations.shape}")
    print(f"  actions:      {actions.shape}")
    print(f"  obs columns:  {len(observation_columns)}")
    print(f"  act columns:  {len(action_columns)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert teleop session logs into NPZ training datasets."
    )

    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing session folders.",
    )

    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory to write NPZ datasets.",
    )

    parser.add_argument(
        "--session",
        default=None,
        help="Optional single session folder name to convert.",
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    if args.session:
        session_dirs = [logs_dir / args.session]
    else:
        session_dirs = sorted(
            path for path in logs_dir.glob("session_*") if path.is_dir()
        )

    if not session_dirs:
        print(f"No session folders found in {logs_dir}")
        return

    for session_dir in session_dirs:
        convert_session(session_dir, output_dir)


if __name__ == "__main__":
    main()
