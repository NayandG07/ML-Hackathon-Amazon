"""
Upload project files (excluding dataset) to Hugging Face Hub.
Supports uploading large artifacts, embeddings, and code cleanly.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


def upload(repo_id: str, repo_type: str = "model", private: bool = True, token: str | None = None):
    project_root = Path(__file__).resolve().parent.parent
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name", "Unknown")
    print(f"[*] Authenticated as HF user: {username}")

    # Check token write permissions
    auth_role = user_info.get("auth", {}).get("accessToken", {}).get("role", "")
    if auth_role == "read":
        print(
            "\n[!] ERROR: Current Hugging Face token has 'read' permission only.\n"
            "    Please provide a token with 'write' permission from https://huggingface.co/settings/tokens\n"
            "    You can pass it via: python utils/upload_to_hf.py --token <YOUR_WRITE_TOKEN>\n"
            "    Or set the HF_TOKEN environment variable.\n"
        )
        sys.exit(1)

    print(f"[*] Ensuring repository '{repo_id}' exists (type: {repo_type}, private: {private})...")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    print(f"[+] Repository ready: https://huggingface.co/{repo_id if repo_type == 'model' else f'{repo_type}s/{repo_id}'}")

    ignore_patterns = [
        "dataset/**",
        ".git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.DS_Store",
    ]

    print(f"\n[*] Starting upload from: {project_root}")
    print("[*] Excluding patterns:", ignore_patterns)
    print("[*] Uploading to Hugging Face Hub (this may take some time for ~23 GB)...")

    api.upload_folder(
        folder_path=str(project_root),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore_patterns,
        commit_message="Upload embeddings, artifacts, and pipeline code",
    )

    print(f"\n[+] Successfully uploaded project files to Hugging Face: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload project to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Ndg07/ML-Hackathon-Amazon",
        help="Hugging Face repository ID (e.g. Ndg07/ML-Hackathon-Amazon)",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset"],
        help="Repository type on Hugging Face (default: model)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token with write permission",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (default is private)",
    )

    args = parser.parse_args()
    upload(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=not args.public,
        token=args.token,
    )
