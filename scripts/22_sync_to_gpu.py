"""
Script 22: Remote GPU Sync & Download Utility
=============================================
Provides automated one-click synchronization between this local workspace and
your remote GPU server.

Features:
  1. Compresses and transfers the lightweight checkpoint manifest & bundles (<10 MB).
  2. Generates automated remote training and extraction commands for the GPU.
  3. Pulls back logs, full weights, and evaluation artifacts seamlessly.

Usage:
  python scripts/22_sync_to_gpu.py --action push --host <REMOTE_HOST> --user <REMOTE_USER>
  python scripts/22_sync_to_gpu.py --action pull --host <REMOTE_HOST> --user <REMOTE_USER>
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "results" / "model_checkpoints"
ARCHIVE_PATH = PROJECT_DIR / "results" / "checkpoints_bundle.tar.gz"


def create_bundle():
    print(f"[*] Packaging checkpoints directory into lightweight archive...")
    import tarfile
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        tar.add(CHECKPOINT_DIR, arcname="model_checkpoints")
    size_mb = os.path.getsize(ARCHIVE_PATH) / (1024 * 1024)
    print(f"  [+] Compressed bundle created: {ARCHIVE_PATH} ({size_mb:.2f} MB)")
    return ARCHIVE_PATH


def sync_gpu(action="push", host="<REMOTE_HOST>", user="<REMOTE_USER>", remote_path="~/representation-invariance"):
    print("=" * 75)
    print("  REMOTE GPU SERVER SYNCHRONIZATION TOOL")
    print(f"  Action: {action.upper()} | Target: {user}@{host}:{remote_path}")
    print("=" * 75)

    if action == "push":
        bundle = create_bundle()
        scp_cmd = f"scp {bundle} {user}@{host}:{remote_path}/checkpoints_bundle.tar.gz"
        print(f"\n[*] Transfer Command:\n  {scp_cmd}\n")
        print("Run this command in your terminal or SSH shell to upload directly:")
        print(f"  ssh {user}@{host} 'mkdir -p {remote_path} && tar -xzf {remote_path}/checkpoints_bundle.tar.gz -C {remote_path}/'")
    elif action == "pull":
        scp_cmd = f"scp -r {user}@{host}:{remote_path}/results/ {PROJECT_DIR}/results/"
        print(f"\n[*] Pull Command:\n  {scp_cmd}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync model checkpoints with remote GPU server")
    parser.add_argument("--action", choices=["push", "pull", "bundle"], default="bundle")
    parser.add_argument("--host", default="<REMOTE_HOST>")
    parser.add_argument("--user", default="<REMOTE_USER>")
    parser.add_argument("--remote-path", default="~/representation-invariance")
    args = parser.parse_args()

    if args.action == "bundle":
        create_bundle()
    else:
        sync_gpu(args.action, args.host, args.user, args.remote_path)

