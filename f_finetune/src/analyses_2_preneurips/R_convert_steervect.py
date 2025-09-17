"""
python3 f_finetune/src/analyses_2_preneurips/R_convert_steervect.py \
  --adapter_dir f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar \
  --layer_idx 12 \
  --which attn_out \
  --out_vec f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar/rank1_o_proj_A_attn_out.npy

python3 f_finetune/src/analyses_2_preneurips/R_convert_steervect.py \
  --adapter_dir f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar \
  --layer_idx 12 \
  --which mlp_out \
  --out_vec f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar/rank1_mlp_down_B.npy
"""
import argparse, re, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

def find_keys(sd, layer0, which):
    # layer0 is 0-based
    layer_pat = re.compile(rf"\.layers\.{layer0}\.")
    keys = [k for k in sd.keys() if layer_pat.search(k)]
    if which == "attn_out":
        # prefer self_attn.o_proj lora_B
        kb = [k for k in keys if "self_attn" in k and "o_proj" in k and ".lora_B.weight" in k]
        ka = [k for k in keys if "self_attn" in k and "o_proj" in k and ".lora_A.weight" in k]
        label = "self_attn.o_proj"
    else:
        # mlp.down_proj lora_B
        kb = [k for k in keys if ".mlp." in k and "down_proj" in k and ".lora_B.weight" in k]
        ka = [k for k in keys if ".mlp." in k and "down_proj" in k and ".lora_A.weight" in k]
        label = "mlp.down_proj"
    return ka, kb, label, keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--layer_idx", type=int, default=12, help="1-based layer index")
    ap.add_argument("--which", default="mlp_out", choices=["attn_out","mlp_out"])
    ap.add_argument("--out_vec", default="rank1_vec.npy")
    ap.add_argument("--use_alpha_scale", action="store_true",
                    help="Multiply by lora_alpha/r from adapter_config.json (defaults to unit-norm otherwise).")
    args = ap.parse_args()

    layer0 = args.layer_idx - 1
    st_path = Path(args.adapter_dir) / "adapter_model.safetensors"
    if not st_path.exists():
        raise SystemExit(f"Not found: {st_path}")

    sd = load_file(str(st_path))

    ka, kb, label, all_keys = find_keys(sd, layer0, args.which)
    if not kb:
        # Helpful error printout
        print(f"Available keys under layer {args.layer_idx}:")
        for k in all_keys:
            print("  ", k)
        raise SystemExit(f"Couldn't find {label}.lora_B.weight at layer {args.layer_idx}. "
                         f"(You pointed at A earlier; for residual direction we need B.)")

    B = sd[kb[0]].detach().cpu()   # expect [hidden, 1]
    if B.ndim != 2 or min(B.shape) != 1:
        raise SystemExit(f"Expected rank-1 B with shape [hidden,1] or [1,hidden], got {tuple(B.shape)} at {kb[0]}")

    # If it came as [1, hidden], transpose
    if B.shape[0] == 1:
        B = B.T

    v = B[:, 0].float().numpy()  # [hidden]

    # Optional scaling by lora_alpha / r
    if args.use_alpha_scale:
        cfg_path = Path(args.adapter_dir) / "adapter_config.json"
        scale = 1.0
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            r = float(cfg.get("r", 1))
            alpha = float(cfg.get("lora_alpha", 1))
            scale = alpha / max(r, 1.0)
        v = v * scale

    # Unit-normalise (directional)
    n = np.linalg.norm(v) + 1e-8
    v = v / n

    outp = Path(args.out_vec)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, v.astype(np.float32))
    print(f"Saved {outp} with shape {v.shape} (unit-norm).")

if __name__ == "__main__":
    main()
