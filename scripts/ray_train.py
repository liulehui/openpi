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


def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
    import jax
    import jax.experimental.multihost_utils as multihost_utils
    import ray.train

    from openpi.training import config as _config

    args = RayTrainArgs(**train_loop_config)
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
    if (hf_cache_dir := _default_hf_cache_dir(args)) is not None:
        env_vars.update(
            {
                "HF_HOME": hf_cache_dir,
                "HF_DATASETS_CACHE": os.path.join(hf_cache_dir, "datasets"),
                "HF_HUB_CACHE": os.path.join(hf_cache_dir, "hub"),
            }
        )

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
