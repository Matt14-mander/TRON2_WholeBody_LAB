"""Export a strict source/carry/robust checkpoint without launching Isaac Sim."""
import argparse
import json
from pathlib import Path

import torch

from sfyg_asset_reference import load_reference
from sfyg_policy import load_checkpoint, sha256


def export(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new export directory")
    model, cfg, contract = load_checkpoint(checkpoint)
    reference = load_reference()
    scripted = torch.jit.script(model)
    generator = torch.Generator().manual_seed(42)
    maximum_error = 0.
    with torch.inference_mode():
        for batch in (1, 7, 32):
            obs = torch.randn(batch, 42, generator=generator)
            history = torch.randn(batch, 10, 42, generator=generator)
            commands = torch.randn(batch, 3, generator=generator) * .2
            expected = model.actor(torch.cat((model.encoder(history.flatten(1)), obs, commands), dim=-1))
            actual = scripted(obs, history, commands)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            maximum_error = max(maximum_error, (actual - expected).abs().max().item())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "policy.pt").open("wb") as stream:
        torch.jit.save(scripted, stream)
    with (output / "policy.pt").open("rb") as stream:
        restored = torch.jit.load(stream).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(obs, history, commands), expected, rtol=1e-5, atol=1e-6)
    cfg.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               policy_sha256=sha256(output / "policy.pt"), source_commit=contract["source_commit"],
               usd_sources=reference["sources"], usd_mass_kg=reference["total_mass_kg"],
               export_max_abs_error=maximum_error)
    (output / "deployment.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[EXPORT] {output}; actor+encoder parity max_abs={maximum_error:.3g}; 42+10x42+3 -> 10")
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output_dir", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    export(args.checkpoint, args.output_dir or args.checkpoint.parent / "exported" / args.checkpoint.stem)
