"""Pin the published revision and fetch params only (not the old optimizer state)."""

import argparse
import json
import pathlib

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    repo = "NathanWu7/pi0_lora_tacfield_tabero"
    revision = HfApi().model_info(repo, revision=args.revision).sha
    prefix = "checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
    print(f"Downloading {repo}@{revision} params only", flush=True)
    snapshot_download(
        repo,
        revision=revision,
        local_dir=args.output_dir,
        allow_patterns=[f"{prefix}/params/**", f"{prefix}/assets/**", "README.md"],
        max_workers=3,
    )
    manifest = {"repo_id": repo, "revision": revision, "params_path": str(args.output_dir / prefix / "params")}
    (args.output_dir / "source_revision.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
