"""Create or update the always-on demo on a free Hugging Face Space (CPU), from deploy/hf-space.

    setx HF_TOKEN "hf_..."          # a Write token (huggingface.co/settings/tokens), once
    python scripts/deploy_hf_space.py [--name stutterclear] [--rebuild]

Docker Spaces on free CPU need a Hugging Face PRO subscription (since 2026); without it Hugging Face
answers 402 Payment Required. The public demo runs on Azure instead (docs/DEPLOY.md).

The Space builds the app from the GitHub repo (deploy/hf-space/Dockerfile). Keys go in as Space
secrets from this computer's environment (NEBIUS_API_KEY, NVIDIA_API_KEY, TAVILY_API_KEY if set):
they are never written to a file or printed. --rebuild restarts the build after a git push.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ("NEBIUS_API_KEY", "NVIDIA_API_KEY", "TAVILY_API_KEY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stutterclear")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN (a Write token from huggingface.co/settings/tokens) first.")
    api = HfApi()
    owner = api.whoami()["name"]
    repo = f"{owner}/{args.name}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT / "deploy" / "hf-space"),
                      commit_message="Deploy StutterClear (Docker, from GitHub)")
    for key in SECRETS:
        if os.environ.get(key):
            api.add_space_secret(repo, key, os.environ[key])
            print(f"secret set: {key}")
    if args.rebuild:
        api.restart_space(repo, factory_reboot=True)
    sub = repo.lower().replace("/", "-").replace("_", "-")
    print(f"Space: https://huggingface.co/spaces/{repo}\nApp:   https://{sub}.hf.space")


if __name__ == "__main__":
    main()
