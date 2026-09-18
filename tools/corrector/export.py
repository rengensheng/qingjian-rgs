# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0", "safetensors>=0.4", "numpy>=1.0"]
# ///
"""把拼音纠错的训练检查点导出成青简能加载的两件套（`config.json` + `model.safetensors`）。

权重名原样转存（torch 层名），`qingjian-neural::corrector` 按名取张量；
字表写死在 Rust 侧（26 个小写字母 + 4 个特殊标记），这里只导出结构配置。

用法：
  uv run tools/corrector/export.py pinyin_correction/data/corrector.pt -o data/corrector
  cargo run -p qingjian-cli -- --neural-corrector data/corrector <错拼拼音>...

对拍（Rust 与 Python 输出一致）：
  python3 -m pinyin_correction.infer <错拼>          # 训练侧贪心解码
  cargo run -p qingjian-cli -- --neural-corrector data/corrector <错拼>  # 同一串应给出同一纠正
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
VOCAB_SIZE = 30  # PAD / BOS / EOS / UNK + a-z，见 qingjian-neural corrector::codec


def main(argv=None):
    ap = argparse.ArgumentParser(description="导出拼音纠错模型（ckpt -> config.json + model.safetensors）")
    ap.add_argument("ckpt", help="训练写出的 corrector.pt")
    ap.add_argument("-o", "--out-dir", default="data/corrector", help="导出目录")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = dict(ckpt.get("args") or {})
    state = ckpt["state_dict"]
    val_acc = ckpt.get("val_acc")

    config = {
        "d_model": int(train_args.get("d_model", 192)),
        "nhead": int(train_args.get("nhead", 4)),
        "enc_layers": int(train_args.get("enc_layers", 3)),
        "dec_layers": int(train_args.get("dec_layers", 3)),
        "dim_ff": int(train_args.get("dim_ff", 512)),
        "max_len": 128,
        "vocab_size": VOCAB_SIZE,
    }
    if config["d_model"] % config["nhead"] != 0:
        sys.exit(f"d_model={config['d_model']} 不能被 nhead={config['nhead']} 整除，检查点不对")
    out_bias = state.get("out.bias")
    if out_bias is None or list(out_bias.shape) != [VOCAB_SIZE]:
        sys.exit("检查点里没有 out.bias / 形状不是 [30]，不是拼音纠错模型吧")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, CONFIG_NAME), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    save_file({k: v.contiguous().float() for k, v in state.items()},
              os.path.join(args.out_dir, WEIGHTS_NAME))

    params = sum(v.numel() for v in state.values())
    print(f"config: {config}")
    print(f"val_acc: {val_acc}  params: {params / 1e6:.2f}M")
    print(f"已导出 -> {args.out_dir}/{{{CONFIG_NAME},{WEIGHTS_NAME}}}")


if __name__ == "__main__":
    main()
