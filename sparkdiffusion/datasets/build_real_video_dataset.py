"""
Preprocess real video datasets into webdataset tar format.

Reads a metadata file (CSV/JSONL) + video files, encodes them with
Wan2.1 VAE and umT5-XXL text encoder, and writes tar shards compatible
with the training pipeline's webdataset loader.

Usage (single GPU):
  python sparkdiffusion/datasets/build_real_video_dataset.py \
      --data_root datasets/OpenVid-1M \
      --output_dir datasets/distill/OpenVid-1M-tar-480p

Usage (multi-GPU):
  torchrun --nproc_per_node=8 sparkdiffusion/datasets/build_real_video_dataset.py \
      --data_root datasets/OpenVid-1M \
      --output_dir datasets/distill/OpenVid-1M-tar-480p
"""

import argparse
import csv
import io
import json
import math
import os
import tarfile
import time
from collections import defaultdict

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from sparkdiffusion.datasets.utils import VIDEO_RES_SIZE_INFO
from sparkdiffusion.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from sparkdiffusion.utils.umt5 import get_umt5_embedding, clear_umt5_memory

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}


def read_metadata(data_root, metadata_file):
    """Read (video_path, caption) pairs from CSV or JSONL."""
    meta_path = os.path.join(data_root, metadata_file)
    entries = []

    if metadata_file.endswith(".jsonl"):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                video_path = os.path.join(data_root, obj["file_name"])
                caption = obj.get("caption", obj.get("text", ""))
                entries.append((video_path, caption))
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = os.path.join(data_root, row["file_name"])
                caption = row.get("caption", row.get("text", ""))
                entries.append((video_path, caption))

    return entries


def load_and_preprocess_video(video_path, num_frames, target_h, target_w):
    """Load video, uniformly sample frames, resize, normalize to [-1, 1].

    Returns: [1, C, T, H, W] float32 tensor in [-1, 1], or None on failure.
    """
    import decord
    decord.bridge.set_bridge("torch")

    try:
        vr = decord.VideoReader(video_path, num_threads=1)
    except Exception as e:
        print(f"  [WARN] Failed to open {video_path}: {e}")
        return None

    total_frames = len(vr)
    if total_frames < 2:
        print(f"  [WARN] Video too short ({total_frames} frames): {video_path}")
        return None

    # Uniformly sample num_frames indices
    if total_frames >= num_frames:
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
    else:
        # Repeat last frame if video is shorter than requested
        indices = torch.arange(total_frames)
        pad = num_frames - total_frames
        indices = torch.cat([indices, indices[-1:].expand(pad)])

    try:
        frames = vr.get_batch(indices.tolist())  # [T, H, W, C] uint8
    except Exception as e:
        print(f"  [WARN] Failed to decode frames from {video_path}: {e}")
        return None

    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float()

    # Resize each frame to target resolution
    frames = torch.stack([TF.resize(f, [target_h, target_w], antialias=True) for f in frames])

    # Normalize to [-1, 1]
    frames = frames / 127.5 - 1.0

    # [T, C, H, W] -> [1, C, T, H, W]
    video = frames.permute(1, 0, 2, 3).unsqueeze(0)

    return video


def write_to_tar(tar, key, data_bytes):
    ti = tarfile.TarInfo(key)
    ti.size = len(data_bytes)
    tar.addfile(ti, io.BytesIO(data_bytes))


def is_shard_done(shard_path):
    return os.path.exists(shard_path) and os.path.getsize(shard_path) > 0


def get_processed_indices(tmp_path):
    """Check which samples are already in the tmp tar."""
    processed = set()
    if not os.path.exists(tmp_path):
        return processed
    try:
        with tarfile.open(tmp_path, "r") as existing_tar:
            files_in_tar = defaultdict(list)
            for member in existing_tar.getmembers():
                parts = member.name.split(".")
                if len(parts) == 3 and parts[0].isdigit():
                    index_str, file_type, _ = parts
                    files_in_tar[int(index_str)].append(file_type)
            needs_schema_rebuild = False
            for index, types in files_in_tar.items():
                has_core = "latent" in types and "embed" in types and "prompt" in types
                if has_core and "first_frame_rgb" not in types:
                    needs_schema_rebuild = True
                    break
                if has_core and "first_frame_rgb" in types:
                    processed.add(index)
            if needs_schema_rebuild:
                print("  [WARN] Tmp file uses old schema without first_frame_rgb, rebuilding shard.")
                os.remove(tmp_path)
                processed.clear()
    except (tarfile.ReadError, EOFError) as e:
        print(f"  [WARN] Tmp file corrupted ({e}), starting over.")
        os.remove(tmp_path)
        processed.clear()
    return processed


def main(args):
    # Distributed setup (optional)
    rank, world_size = 0, 1
    try:
        from imaginaire.utils import distributed
        distributed.init()
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    except Exception:
        pass

    entries = read_metadata(args.data_root, args.metadata_file)
    if args.max_samples > 0:
        entries = entries[:args.max_samples]

    total = len(entries)
    num_shards = math.ceil(total / args.samples_per_shard)
    my_shards = [i for i in range(num_shards) if i % world_size == rank]

    print(f"[Rank {rank}] {total} samples, {num_shards} shards, processing {len(my_shards)} shards.")

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    # Load VAE
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    tokenizer.model.model = tokenizer.model.model.to("cuda")

    # Pre-encode all text embeddings (T5 is loaded once, singleton cached)
    print(f"[Rank {rank}] Encoding text embeddings...")
    all_captions = [e[1] for e in entries]

    # Encode captions in batches to avoid OOM
    t5_batch_size = args.t5_batch_size
    all_text_embs = []
    for batch_start in tqdm(range(0, len(all_captions), t5_batch_size),
                            desc="T5 encoding", disable=(rank != 0)):
        batch_captions = all_captions[batch_start:batch_start + t5_batch_size]
        embs = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=batch_captions,
            tokenizer_path=args.tokenizer_path,
        ).to(dtype=torch.bfloat16).cpu()
        all_text_embs.append(embs)

    all_text_embs = torch.cat(all_text_embs, dim=0)  # [N, 512, 4096]
    clear_umt5_memory()
    print(f"[Rank {rank}] Text embeddings done: {all_text_embs.shape}")

    # Process shards
    os.makedirs(args.output_dir, exist_ok=True)
    skipped = 0

    for shard_id in my_shards:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id:06d}.tar")

        if is_shard_done(shard_path):
            print(f"[Rank {rank}] Shard {shard_id} done, skipping.")
            continue

        start = shard_id * args.samples_per_shard
        end = min(total, start + args.samples_per_shard)

        tmp_path = shard_path + ".tmp"
        processed_indices = get_processed_indices(tmp_path)
        if processed_indices:
            print(f"[Rank {rank}] Shard {shard_id}: {len(processed_indices)} already done.")

        print(f"[Rank {rank}] Building shard {shard_id}, items {start}..{end-1}")

        for idx in tqdm(range(start, end), desc=f"Shard {shard_id}", disable=(rank != 0)):
            if idx in processed_indices:
                continue

            video_path, caption = entries[idx]

            video = load_and_preprocess_video(video_path, args.num_frames, h, w)
            if video is None:
                skipped += 1
                continue
            first_frame_rgb = video[0, :, 0].clone().contiguous().cpu().float()  # [3, H, W], resized/normalized to [-1, 1]

            # VAE encode: [1, C, T, H, W] -> [1, 16, T_lat, H/8, W/8]
            with torch.no_grad():
                latent = tokenizer.encode(video.to("cuda")).cpu().float()
            latent = latent.squeeze(0).clone().contiguous()  # [16, T_lat, H/8, W/8]

            text_emb = all_text_embs[idx].clone().contiguous()  # [512, 4096]

            key_prefix = f"{idx:09d}"
            try:
                with tarfile.open(tmp_path, "a") as tar:
                    latent_buf = io.BytesIO()
                    torch.save(latent, latent_buf)
                    write_to_tar(tar, f"{key_prefix}.latent.pt", latent_buf.getvalue())

                    first_frame_buf = io.BytesIO()
                    torch.save(first_frame_rgb, first_frame_buf)
                    write_to_tar(tar, f"{key_prefix}.first_frame_rgb.pt", first_frame_buf.getvalue())

                    embed_buf = io.BytesIO()
                    torch.save(text_emb, embed_buf)
                    write_to_tar(tar, f"{key_prefix}.embed.pt", embed_buf.getvalue())

                    write_to_tar(tar, f"{key_prefix}.prompt.txt", caption.encode("utf-8"))
            except Exception as e:
                print(f"[Rank {rank}] Failed to write idx {idx}: {e}")
                continue

        # Finalize shard
        if os.path.exists(tmp_path):
            os.rename(tmp_path, shard_path)
            print(f"[Rank {rank}] Finished shard {shard_id}")
        else:
            print(f"[Rank {rank}] Shard {shard_id} empty (all skipped)")

    print(f"[Rank {rank}] All done. Skipped {skipped} videos.")

    # Wait for all ranks
    try:
        torch.distributed.barrier()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess real videos into webdataset tar")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset (contains metadata + videos)")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv",
                        help="Metadata file name (CSV or JSONL) relative to data_root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for tar shards")
    parser.add_argument("--samples_per_shard", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of samples to process (-1 = all)")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to sample from each video (must be 4n+1)")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--vae_path", type=str,
                        default="pretrain_weights/Wan2.1-T2V-1.3B-Diffusers/vae")
    parser.add_argument("--text_encoder_path", type=str,
                        default="pretrain_weights/Wan2.1-T2V-1.3B-Diffusers/text_encoder")
    parser.add_argument("--t5_batch_size", type=int, default=64,
                        help="Batch size for T5 text encoding")
    parser.add_argument("--tokenizer_path", type=str,
                        default="pretrain_weights/Wan2.1-T2V-1.3B-Diffusers/tokenizer",
                        help="Local path to umt5 tokenizer directory")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0, \
        f"num_frames must be 4n+1, got {args.num_frames}"

    main(args)
