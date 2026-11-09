"""核验B1源码、资产、运行配置和可选本机检查点的冻结身份。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash one source, asset or checkpoint using bounded memory.

    Args:
        path: Existing file whose exact bytes define a frozen artifact.

    Returns:
        Lowercase SHA256 digest.

    Raises:
        OSError: The file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Compare the checkout with the archived live-run identity.

    Raises:
        AssertionError: Source, asset, configuration or a present checkpoint differs.
        OSError: A required manifest or tracked runtime file cannot be read.

    Note:
        Prints a JSON result without changing files or contacting the training host.
        Checkpoint bytes are intentionally outside Git and checked only if present.
    """
    directory = Path(__file__).resolve().parent
    project_root = directory.parents[2]
    snapshot = json.loads((directory / "remote_snapshot.json").read_text())
    source_root = project_root / "UniFP_v3"
    identity = snapshot["metadata"]["identity"]
    counts = {}
    for group in ("source_manifest", "asset_manifest"):
        for relative_path, expected_hash in identity[group].items():
            assert file_sha256(source_root / relative_path) == expected_hash, relative_path
        counts[group] = len(identity[group])
    for relative_path, record in snapshot["additional_runtime_files"].items():
        assert file_sha256(source_root / relative_path) == record["sha256"], relative_path
    checkpoint = project_root / "checkpoints/B1/model_017800.pt"
    checkpoint_present = checkpoint.exists()
    if checkpoint_present:
        assert file_sha256(checkpoint) == snapshot["frozen_checkpoint"]["sha256"]
    print(json.dumps({"branch": "B1", "source_and_asset_match": True, **counts,
        "additional_runtime_files": len(snapshot["additional_runtime_files"]),
        "checkpoint_iteration": snapshot["frozen_checkpoint"]["completed_iterations"],
        "local_checkpoint_present": checkpoint_present,
        "local_checkpoint_matches": True if checkpoint_present else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
