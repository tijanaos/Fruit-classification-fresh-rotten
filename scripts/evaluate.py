import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from fruitcls.config import Config
from fruitcls.data.loader import build_test_loader
from fruitcls.eval.evaluate import evaluate_model


def _default_model_path(cfg: Config, model_name: str) -> Path:
    if model_name == "mobilenet":
        return cfg.models_dir / "mobilenet" / "best.keras"
    if model_name == "custom":
        return cfg.models_dir / "custom_cnn" / "best.keras"
    raise ValueError(f"Unknown model: {model_name}")


def main():
    cfg = Config()

    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mobilenet", "custom"], default="mobilenet")
    p.add_argument("--model-path", type=str, default=None, help="Optional override path to .keras model file")
    p.add_argument("--test-dir", type=str, default=str(cfg.data_test_dir))
    p.add_argument("--img-size", type=int, default=cfg.img_size)
    p.add_argument("--batch", type=int, default=cfg.batch_size)
    p.add_argument("--save", action="store_true", help="Save metrics + confusion matrix to reports/metrics")
    args = p.parse_args()

    model_path = Path(args.model_path) if args.model_path else _default_model_path(cfg, args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_ds, class_names = build_test_loader(Path(args.test_dir), args.img_size, args.batch)

    print(f"\n✅ Evaluating model: {args.model}")
    print(f"   Model path: {model_path}")
    print(f"   Test dir:   {args.test_dir}")
    print(f"   Classes:    {class_names}\n")

    model = tf.keras.models.load_model(model_path, compile=False) 

    result = evaluate_model(model, test_ds, class_names)

    print(f"TEST loss: {result.loss:.4f}")
    print(f"TEST acc : {result.accuracy:.4f}")
    print(f"Macro precision: {result.macro_precision:.4f}")
    print(f"Macro recall   : {result.macro_recall:.4f}\n")

    print("Per-class precision / recall:")
    for c in class_names:
        print(f" - {c:14s}  P={result.per_class_precision[c]:.4f}  R={result.per_class_recall[c]:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(result.confusion_matrix)

    if args.save:
        out_dir = cfg.reports_metrics_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "model": args.model,
            "model_path": str(model_path),
            "test_dir": args.test_dir,
            "loss": result.loss,
            "accuracy": result.accuracy,
            "macro_precision": result.macro_precision,
            "macro_recall": result.macro_recall,
            "per_class_precision": result.per_class_precision,
            "per_class_recall": result.per_class_recall,
            "class_names": class_names,
        }

        metrics_path = out_dir / f"test_metrics_{args.model}.json"
        cm_path = out_dir / f"confusion_matrix_{args.model}.csv"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        np.savetxt(cm_path, result.confusion_matrix, delimiter=",", fmt="%d")

        print(f"\n💾 Saved metrics to: {metrics_path}")
        print(f"💾 Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()