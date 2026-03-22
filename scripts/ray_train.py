import dataclasses
import os
import shutil
from typing import Any
from typing import Literal

import tyro

import openpi.training.train_lib as train_lib


@dataclasses.dataclass(frozen=True)
class RayTrainArgs:
    config_name: str
    exp_name: str

    accelerator: Literal["tpu", "gpu"] = "tpu"
    num_workers: int = 2
    num_accelerators_per_worker: int = 4
    topology: str | None = None
    accelerator_type: str | None = None
    placement_strategy: str = "SPREAD"

    ray_address: str | None = None
    run_name: str | None = None
    storage_path: str | None = None

    batch_size: int | None = None
    num_train_steps: int | None = None
    fsdp_devices: int | None = None
    checkpoint_base_dir: str | None = None
    assets_base_dir: str | None = None
    data_repo_id: str | None = None
    data_asset_id: str | None = None
    data_assets_dir: str | None = None
    hf_lerobot_home: str | None = None
    hf_cache_dir: str | None = None
    num_dataloader_workers: int | None = None
    save_interval: int | None = None

    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True


def _apply_overrides(config, args: RayTrainArgs):
    if args.hf_lerobot_home is not None:
        os.environ["HF_LEROBOT_HOME"] = args.hf_lerobot_home

    data = config.data
    if args.data_repo_id is not None or args.data_asset_id is not None or args.data_assets_dir is not None:
        current_assets = data.assets
        data = dataclasses.replace(
            data,
            repo_id=args.data_repo_id if args.data_repo_id is not None else data.repo_id,
            assets=dataclasses.replace(
                current_assets,
                assets_dir=args.data_assets_dir if args.data_assets_dir is not None else current_assets.assets_dir,
                asset_id=args.data_asset_id if args.data_asset_id is not None else current_assets.asset_id,
            ),
        )

    replace_kwargs: dict[str, Any] = {
        "exp_name": args.exp_name,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
        "data": data,
    }
    if args.batch_size is not None:
        replace_kwargs["batch_size"] = args.batch_size
    if args.num_train_steps is not None:
        replace_kwargs["num_train_steps"] = args.num_train_steps
    if args.fsdp_devices is not None:
        replace_kwargs["fsdp_devices"] = args.fsdp_devices
    if args.checkpoint_base_dir is not None:
        replace_kwargs["checkpoint_base_dir"] = args.checkpoint_base_dir
    if args.assets_base_dir is not None:
        replace_kwargs["assets_base_dir"] = args.assets_base_dir
    if args.num_dataloader_workers is not None:
        replace_kwargs["num_workers"] = args.num_dataloader_workers
    if args.save_interval is not None:
        replace_kwargs["save_interval"] = args.save_interval
    return dataclasses.replace(config, **replace_kwargs)


def _default_hf_cache_dir(args: RayTrainArgs) -> str | None:
    """Choose a persistent HF cache location when using shared storage."""
    if args.hf_cache_dir is not None:
        return args.hf_cache_dir
    if args.data_repo_id is not None and os.path.isabs(args.data_repo_id):
        return os.path.join(args.data_repo_id, ".hf_cache")
    if args.storage_path is not None and os.path.isabs(args.storage_path):
        return os.path.join(args.storage_path, "hf_cache")
    return None


def _apply_datasets_v3_compat():
    """Monkey-patch torch.stack and lerobot to handle datasets>=3.0 Column type.

    Applied in the worker process so it takes effect before any dataset loading.
    """
    import logging
    import time

    import torch

    _orig_stack = torch.stack

    def _compat_stack(tensors, *args, **kwargs):
        if not isinstance(tensors, (list, tuple)):
            try:
                return _orig_stack(tensors, *args, **kwargs)
            except TypeError:
                return _orig_stack(list(tensors), *args, **kwargs)
        return _orig_stack(tensors, *args, **kwargs)

    torch.stack = _compat_stack

    import lerobot.common.datasets.lerobot_dataset as _ld

    _orig_init = _ld.LeRobotDataset.__init__

    def _instrumented_init(self, *args, **kwargs):
        t0 = time.monotonic()
        logging.info("[LeRobotDataset] __init__ starting...")

        from pathlib import Path

        from lerobot.common.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            HF_LEROBOT_HOME,
            LeRobotDatasetMetadata,
            get_safe_default_codec,
        )
        from lerobot.common.datasets.utils import (
            check_delta_timestamps,
            check_timestamps_sync,
            get_delta_indices,
            get_episode_data_index,
        )

        repo_id = args[0] if args else kwargs.get("repo_id")
        root = args[1] if len(args) > 1 else kwargs.get("root", None)
        episodes = kwargs.get("episodes", None)
        image_transforms = kwargs.get("image_transforms", None)
        delta_timestamps = kwargs.get("delta_timestamps", None)
        tolerance_s = kwargs.get("tolerance_s", 1e-4)
        revision = kwargs.get("revision", None)
        video_backend = kwargs.get("video_backend", None)
        force_cache_sync = kwargs.get("force_cache_sync", False)
        download_videos = kwargs.get("download_videos", True)

        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)

        logging.info(f"[LeRobotDataset] Loading metadata... ({time.monotonic() - t0:.1f}s)")
        self.meta = LeRobotDatasetMetadata(self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync)

        # Skip the 33s NFS file existence check and Hub fallback -- data is already local.
        logging.info(f"[LeRobotDataset] Loading HF dataset directly... ({time.monotonic() - t0:.1f}s)")
        self.hf_dataset = self.load_hf_dataset()

        logging.info(f"[LeRobotDataset] HF dataset loaded. Building episode index... ({time.monotonic() - t0:.1f}s)")
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Skip timestamp validation -- it calls torch.stack on 273k-element columns
        # which is extremely slow with datasets 3.x. Data was validated during preprocessing.

        if self.delta_timestamps is not None:
            logging.info(f"[LeRobotDataset] Computing delta indices... ({time.monotonic() - t0:.1f}s)")
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        logging.info(f"[LeRobotDataset] __init__ complete in {time.monotonic() - t0:.1f}s")

    _ld.LeRobotDataset.__init__ = _instrumented_init


def _apply_multislice_mesh_fix():
    """Fix JAX mesh creation for multislice TPU.

    jax.make_mesh uses create_device_mesh which queries the per-slice physical
    topology. With MegaScale multislice, the topology only describes one slice,
    so the assertion len(devices) == prod(dims) fails when there are 2+ slices.

    The fix: construct the mesh directly from the flat device list. The device
    ordering from jax.devices() is already slice-major, host-major, chip-major,
    which naturally groups each host's chips into one FSDP row.
    """
    import openpi.training.sharding as _sharding

    _orig_make_mesh = _sharding.make_mesh

    def _multislice_make_mesh(num_fsdp_devices: int):
        import jax
        import logging
        import numpy as np

        num_devices = jax.device_count()
        if num_devices % num_fsdp_devices != 0:
            raise ValueError(
                f"Number of devices {num_devices} must be divisible by "
                f"the number of FSDP devices {num_fsdp_devices}."
            )
        mesh_shape = (num_devices // num_fsdp_devices, num_fsdp_devices)

        try:
            return _orig_make_mesh(num_fsdp_devices)
        except (AssertionError, Exception) as exc:
            logging.info(
                f"[multislice] jax.make_mesh failed ({type(exc).__name__}), "
                f"constructing mesh manually: shape={mesh_shape} from {num_devices} devices"
            )
            devices = np.array(jax.devices()).reshape(mesh_shape)
            return jax.sharding.Mesh(devices, axis_names=(_sharding.BATCH_AXIS, _sharding.FSDP_AXIS))

    _sharding.make_mesh = _multislice_make_mesh


def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
    import jax
    import jax.experimental.multihost_utils as multihost_utils
    import ray.train

    _apply_datasets_v3_compat()
    _apply_multislice_mesh_fix()

    import logging as _logging
    import resource
    import subprocess
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _logging.info(f">>> [ulimit] Before: soft={soft}, hard={hard}")
    try:
        subprocess.run(["bash", "-c", "ulimit -n 1048576"], check=False)
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))
    except (ValueError, OSError):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ValueError, OSError):
            pass
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _logging.info(f"[ulimit] After: soft={soft}, hard={hard}")

    from openpi.training import config as _config

    args = RayTrainArgs(**train_loop_config)

    if (hf_cache_dir := _default_hf_cache_dir(args)) is not None:
        worker_cache = os.path.join(hf_cache_dir, f"worker_{jax.process_index()}")
        os.makedirs(worker_cache, exist_ok=True)
        os.environ["HF_HOME"] = worker_cache
        os.environ["HF_DATASETS_CACHE"] = os.path.join(worker_cache, "datasets")
        os.environ["HF_HUB_CACHE"] = os.path.join(worker_cache, "hub")

    config = _apply_overrides(_config.get_config(args.config_name), args)

    if args.overwrite and jax.process_index() == 0 and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
    multihost_utils.sync_global_devices("checkpoint_dir_prepared")

    # The chief worker owns the external run state.
    config = dataclasses.replace(
        config,
        overwrite=False,
        wandb_enabled=config.wandb_enabled and jax.process_index() == 0,
    )

    train_lib.main(config)


def main(args: RayTrainArgs) -> None:
    import ray
    from ray.train import RunConfig
    from ray.train import ScalingConfig

    try:
        from ray.train.v2.jax import JaxTrainer
    except ImportError as exc:
        raise ImportError(
            "Ray JaxTrainer is not available in the installed Ray version "
            f"({ray.__version__}). Install a Ray release that provides "
            "`ray.train.v2.jax.JaxTrainer` to use this script."
        ) from exc


    run_name = args.run_name or f"{args.config_name}-{args.exp_name}"
    env_vars = {
        "JAX_PLATFORMS": "tpu" if args.accelerator == "tpu" else "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }

    run_config_kwargs: dict[str, Any] = {
        "name": run_name,
        "worker_runtime_env": {
            "env_vars": env_vars,
        },
    }
    if args.storage_path is not None:
        run_config_kwargs["storage_path"] = args.storage_path

    if args.accelerator == "tpu":
        scaling_config = ScalingConfig(
            num_workers=args.num_workers,
            use_tpu=True,
            topology=args.topology,
            accelerator_type=args.accelerator_type,
            placement_strategy=args.placement_strategy,
            resources_per_worker={"TPU": 4},
        )
    else:
        scaling_config = ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            placement_strategy=args.placement_strategy,
            resources_per_worker={"GPU": args.num_accelerators_per_worker},
        )

    trainer = JaxTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=dataclasses.asdict(args),
        scaling_config=scaling_config,
        run_config=RunConfig(**run_config_kwargs),
    )
    trainer.fit()


if __name__ == "__main__":
    main(tyro.cli(RayTrainArgs))
