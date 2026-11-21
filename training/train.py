"""Command-line entry point.

    python -m training.train --config training/configs/l23_two_stage.yaml
    python -m training.train --config ... --stages subthreshold          # one stage only
    python -m training.train --config ... --resume                       # continue after a crash / timeout
    python -m training.train --config ... --set training.batch_size=128 data.files.all=[a.txt]
    python -m training.train --config ... --eval-only --split test       # closed-loop score of finished stages
"""
from __future__ import annotations

import argparse
import json
import logging
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config import load_config


def _parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw) if yaml else json.loads(raw)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Staged training of CNN-LSTM neuron surrogates")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stages", help="comma-separated stage names to run (default: all)")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(a.config, _parse_overrides(a.set))
    from .trainer import Trainer  # import TF only after arg parsing

    if a.eval_only:
        from .evaluate import evaluate_closed_loop
        tr = Trainer(cfg, resume=True)
        for s in cfg["stages"]:
            p = os.path.join(tr.run_dir, s["name"], "best.weights.h5")
            if os.path.exists(p):
                tr.model.load_weights(p)
                print(s["name"], json.dumps(evaluate_closed_loop(tr.model, tr.ds, cfg, a.split), indent=2))
        return
    tr = Trainer(cfg, resume=a.resume)
    res = tr.run(a.stages.split(",") if a.stages else None)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
