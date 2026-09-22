import glob
from functools import partial
import random
import webdataset as wds
import numpy as np
import torch
from torch.utils.data import DataLoader


def dict_collation_fn(samples):
    if not samples:
        return {}

    keys = set(samples[0].keys())
    for sample in samples[1:]:
        keys &= set(sample.keys())
    batched_dict = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            batched_dict[key].append(sample[key])

    for key in keys:
        if isinstance(batched_dict[key][0], torch.Tensor):
            batched_dict[key] = torch.stack(batched_dict[key])

    return batched_dict


def rename_sample(sample, include_first_frame_rgb=False):
    missing = [k for k in ("latent.pt", "embed.pt", "prompt.txt") if k not in sample]
    if missing:
        raise KeyError(f"incomplete sample {sample.get('__key__')} in {sample.get('__url__')}: missing {missing}")
    renamed = {
        "latents": sample["latent.pt"],
        "t5_text_embeddings": sample["embed.pt"],
        "prompts": sample["prompt.txt"],
    }
    if include_first_frame_rgb and "first_frame_rgb.pt" in sample:
        renamed["first_frame_rgb"] = sample["first_frame_rgb.pt"]
    return renamed


def create_dataloader(
    tar_path_pattern,  # e.g., "/path/to/dataset/shard_*.tar"
    batch_size,
    num_workers=4,
    shuffle_buffer=64,
    prefetch_factor=2,
    seed=None,
    include_first_frame_rgb=False,
):
    # sorted() keeps the shard list order consistent across runs
    shards = sorted(glob.glob(tar_path_pattern))
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    # Key constraint: webdataset uses split_by_node to assign whole shards per rank
    # (it does not split an individual tar). If the number of shards < number of
    # GPUs (ranks), the trailing ranks get no shard and next(dataloader) blocks
    # forever, silently deadlocking multi-GPU training at the first FSDP collective.
    # Fail early here.
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _ws = _dist.get_world_size()
            if len(shards) < _ws:
                raise ValueError(
                    f"number of tar shards ({len(shards)}) < world_size ({_ws}). "
                    f"webdataset assigns whole shards per rank, so trailing ranks get no data and hang. "
                    f"Ensure the number of shards >= number of GPUs (split/add shards, or reduce GPUs). "
                    f"pattern='{tar_path_pattern}'"
                )
    except (ImportError, RuntimeError):
        pass

    # Fix the global random state (wds.shuffle uses Python random internally)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        # this shuffles the shards
        wds.shuffle(1000, seed=seed),
        wds.split_by_node,
        wds.split_by_worker,
        # warn_and_continue: skip corrupt/incomplete samples instead of killing the run
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        # this shuffles the samples in memory
        wds.shuffle(shuffle_buffer, seed=seed),
        wds.decode(wds.handle_extension("pt", wds.torch_loads), handler=wds.warn_and_continue),
        wds.map(partial(rename_sample, include_first_frame_rgb=include_first_frame_rgb), handler=wds.warn_and_continue),
        wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
    )

    # Independent but deterministic seed per DataLoader worker
    def _worker_init_fn(worker_id):
        if seed is not None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn,
    )

    return dataloader
