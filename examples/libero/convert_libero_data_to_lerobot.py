"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

The raw RLDS input can live either on local disk or on GCS. For example:
- /path/to/your/data
- gs://your-bucket/modified_libero_rlds

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --source_dir /path/to/your/data --output_dir /tmp/libero

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --source_dir gs://your-bucket/modified_libero_rlds --repo_id your_hf_username/libero --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
By default the converted LeRobot dataset is written under $HF_LEROBOT_HOME/<repo_id>.
If `output_dir` is provided, the final dataset is moved to that local path instead.
Running this conversion script will take approximately 30 minutes.
"""

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

DEFAULT_REPO_ID = "your_hf_username/libero"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(
    source_dir: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: str | None = None,
    raw_dataset_names: list[str] = RAW_DATASET_NAMES,
    push_to_hub: bool = False,
):
    # `tfds.load` accepts both local paths and `gs://` sources.
    source_dir = source_dir.rstrip("/")
    if output_dir is not None and output_dir.startswith("gs://"):
        raise ValueError(
            "LeRobotDataset writes to a local filesystem path, not a raw gs:// URI. "
            "Mount the bucket with gcsfuse and pass the mounted path to --output_dir instead."
        )

    root_path = Path(output_dir).expanduser() if output_dir is not None else HF_LEROBOT_HOME / repo_id
    if root_path.exists():
        shutil.rmtree(root_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        root=root_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=source_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
