"""Convert the training checkpoint of the released run into HyPDE's weights (model.safetensors + config.json).

The scored weights are the RAW weights of step 27,280 of run main2d_w64_wnorm_v1 (`state["model"]`, not the EMA
shadow), the checkpoint every recorded number was measured on. The training code built one stem, head, refiner,
fusion and output projection per dimensionality in ModuleDicts keyed "2" and "3"; HyPDE is the 2D model, so the
"2" entries are renamed to plain module paths and the never-trained 3D entries are dropped. The token-grid
depthwise conv keeps its 2D kernel only (`mlp.conv.dw2` -> `mlp.dw`). Everything else keeps its name, and the
result is loaded with strict=True, so a missed or extra key fails.

    python tools/convert_checkpoint.py <step dir with state.pt> <out dir>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hypde.api import sha256  # noqa: E402
from hypde.model import HyPDE, HyPDEConfig  # noqa: E402

PER_DIM = ("stem", "head", "refiner", "fuse", "out_proj", "pad_emb")


def convert(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = ".".join(p for p in k.split(".") if p != "_orig_mod")
        top, _, rest = k.partition(".")
        if top in PER_DIM:
            dim, _, tail = rest.partition(".")
            if dim == "3":
                continue
            assert dim == "2", k
            k = f"{top}.{tail}" if tail else top
        if k.endswith(".mlp.conv.dw3"):
            continue
        k = re.sub(r"\.mlp\.conv\.dw2$", ".mlp.dw", k)
        out[k] = v.detach().to(torch.float32).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="directory holding state.pt")
    ap.add_argument("out")
    a = ap.parse_args()
    state = torch.load(Path(a.ckpt) / "state.pt", map_location="cpu", weights_only=False)
    cfg_run = state["config"]
    assert cfg_run["model"]["core"] == "F" and cfg_run["model"]["name"] == "F321", cfg_run["model"]
    assert cfg_run["norm"]["mode"] == "walrus_full", cfg_run["norm"]
    sd = convert(state["model"])
    model = HyPDE(HyPDEConfig())
    model.load_state_dict(sd, strict=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = out / "model.safetensors"
    save_file(sd, str(st), metadata={"format": "pt", "source_step": str(state.get("step"))})
    cfg = {"architecture": HyPDEConfig().to_dict(),
           "parameters": model.num_params(), "processor_parameters": model.num_params("processor"),
           "weights": "raw (not EMA)", "source_step": state.get("step"), "source_commit": state.get("commit"),
           "source_checkpoint_sha256": sha256(Path(a.ckpt) / "state.pt"), "sha256": sha256(st),
           "dtype": "float32"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
