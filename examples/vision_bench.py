"""Images in, certified decisions out, in microseconds on a CPU: a reproducible benchmark for shad0w.vision.

    pip install "shad0wllm[compile,vision]" datasets
    make                                                   # the C core (pixels -> decision in one call)
    python examples/vision_bench.py --dataset fashion_mnist
    python examples/vision_bench.py --dataset cifar10 --labels vlm_labels.jsonl     # answers of a real vision LLM
    python examples/vision_bench.py --dataset cifar10 --labels vlm_labels.jsonl --embed openai/clip-vit-base-patch32
                                                       # a cascade: ferns (us) -> CLIP table (ms) -> the vision LLM

The teacher is the dataset's labels by default (an upper bound: a perfectly consistent teacher). With --labels, a
JSON Lines file of {"i": <test-split index>, "vlm": "<answer>"} rows (for example from a local VLM through ollama)
is the teacher instead, which is the real use: the table learns what YOUR vision model answers.

Reports: agreement with the teacher on unseen images, the share certified at alpha, the realised disagreement on that
share (the certificate says it is at most alpha with 90% confidence), table size, and pixels -> decision latency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shad0w import native  # noqa: E402
from shad0w.mathutil import softmax  # noqa: E402
from shad0w.shadow import shadow_compile  # noqa: E402
from shad0w.vision import ImageSpec, image_items  # noqa: E402

DATASETS = {"fashion_mnist": ("zalando-datasets/fashion_mnist", "image"), "cifar10": ("uoft-cs/cifar10", "img"),
            "mnist": ("ylecun/mnist", "image")}


def load(name: str, n: int, labels_file: str | None):
    from datasets import load_dataset
    hub, col = DATASETS[name]
    if labels_file:  # the teacher is a vision model's answers on the test split
        ds = load_dataset(hub, split="test")
        names = ds.features["label"].names
        rows = [json.loads(l) for l in open(labels_file)]
        rows = [r for r in rows if r.get("vlm") in names][:n]
        X = [np.asarray(ds[r["i"]][col]) for r in rows]
        return X, [r["vlm"] for r in rows], [names[ds[r["i"]]["label"]] for r in rows], names
    ds = load_dataset(hub, split="train").shuffle(seed=0).select(range(n))
    names = ds.features["label"].names
    X = [np.asarray(im) for im in ds[col]]
    y = [names[v] for v in ds["label"]]
    return X, y, y, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion_mnist", choices=sorted(DATASETS))
    ap.add_argument("--n", type=int, default=12000, help="logged (image, answer) pairs, then test images")
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--labels", help="JSON Lines of a vision model's answers (the teacher); default: dataset labels")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ferns", type=int, default=256)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--channels", type=int, default=0, help="1 or 3 (default: the dataset's)")
    ap.add_argument("--embed", help="add a second tier: this image encoder (a Hugging Face CLIP/SigLIP id), certified "
                                    "jointly with the ferns tier (needs torch + transformers)")
    ap.add_argument("--vlm-ms", type=float, default=2600.0, help="the vision LLM's latency, for the cost estimate")
    a = ap.parse_args()

    X, teacher, truth, names = load(a.dataset, a.n, a.labels)
    ch = a.channels or (3 if X[0].ndim == 3 else 1)
    spec = ImageSpec(channels=ch, ferns=a.ferns, bits=a.bits, radius=a.radius)
    n_test = min(a.test, len(X) // 4)
    log_ix, test_ix = np.arange(len(X) - n_test), np.arange(len(X) - n_test, len(X))
    t = time.time()
    its = [image_items(x, spec) for x in X]
    feat_s = time.time() - t
    rows = [{"items": its[i].tolist(), "label": teacher[i]} for i in log_ix]
    schema = {"label": {"type": "choice", "criteria": {k: None for k in names}}}
    t = time.time()
    model, cert = shadow_compile(schema, rows, alpha=a.alpha, image=spec.to_dict())
    train_s = time.time() - t
    q = model.questions["label"]
    c = cert["questions"]["label"]

    P = np.stack([softmax(q.rx.logits(its[i])) for i in test_ix])
    pred = [names[k] for k in P.argmax(1)]
    served = P.max(1) >= q.threshold
    t_ans = np.array([teacher[i] for i in test_ix])
    agree = np.array(pred) == t_ans
    truth_t = np.array([truth[i] for i in test_ix])

    print(f"\n{a.dataset}: {len(log_ix)} logged answers, {n_test} unseen test images, teacher = "
          f"{'vision LLM answers' if a.labels else 'dataset labels'}, alpha = {a.alpha}")
    if a.labels:
        print(f"  teacher accuracy vs truth on test images:    {(t_ans == truth_t).mean():.1%}")
    print(f"  table agrees with teacher (all test images): {agree.mean():.1%}")
    print(f"  certified on calibration / on test:          {c['certified_share_on_calibration']:.1%} / {served.mean():.1%}")
    if served.any():
        print(f"  disagreement on certified test answers:      {(~agree[served]).mean():.1%}  (certificate: <= {a.alpha:.0%})")
    print(f"  table size: {q.rx.nbytes / 1e6:.2f} MB   items/image: {spec.n_items}   train: {train_s:.0f}s   "
          f"featurise (python loop): {feat_s / len(X) * 1e6:.0f} us/image")

    if a.embed:
        cascade(a, X, teacher, names, spec, its, log_ix, test_ix)

    if native.available():
        with tempfile.TemporaryDirectory() as d:
            model.save(d)
            table = native.NativeReflex(os.path.join(d, "label.s0"))
            nat = native.NativeImage(spec)
            sample = [np.ascontiguousarray(X[i]) for i in test_ix[:500] if X[i].shape == X[test_ix[0]].shape]
            ns = nat.bench(table, sample, reps=50)
            big = [np.ascontiguousarray(np.kron(s, np.ones((7, 7) + ((1,) if s.ndim == 3 else ()), np.uint8)))
                   for s in sample[:100]]
            ns_big = nat.bench(table, big, reps=20)
        print(f"  C core, pixels -> decision: {ns / 1e3:.1f} us at {sample[0].shape}, "
              f"{ns_big / 1e3:.1f} us at {big[0].shape} (one core, no GPU; decoding a JPEG/PNG is extra)")


def cascade(a, X, teacher, names, spec, its, log_ix, test_ix):
    """Ferns tier -> embedding tier -> the teacher, one certificate (shad0w.shadow.cascade_compile)."""
    from shad0w.shadow import cascade_compile
    from shad0w.vision import EmbedSpec, embed_vector
    t = time.time()
    emb = [embed_vector(x, a.embed) for x in X]
    enc_ms = (time.time() - t) / len(X) * 1e3
    rows = [{"items": its[i].tolist(), "embed": emb[i].tolist(), "label": teacher[i]} for i in log_ix]
    schema = {"type": "choice", "criteria": {k: None for k in names}}
    model, cert = cascade_compile("label", schema, [spec, EmbedSpec(a.embed)], rows, alpha=a.alpha)
    tiers = model.meta["tiers"]
    fitted = EmbedSpec.from_dict(tiers[1]["spec"])
    by, wrong = [0, 0, 0], 0
    for i in test_ix:
        for k, t_ in enumerate(tiers):
            q = model.questions[t_["question"]]
            p = softmax(q.rx.logits(its[i] if k == 0 else fitted.items(emb[i])))
            if p.max() >= q.threshold:
                by[k] += 1
                wrong += names[int(p.argmax())] != teacher[i]
                break
        else:
            by[2] += 1
    n = len(test_ix)
    served = by[0] + by[1]
    cost = (by[0] * 0.01 + by[1] * enc_ms + by[2] * (enc_ms + a.vlm_ms)) / n
    print(f"\n  cascade (one certificate, alpha = {a.alpha}): ferns {by[0] / n:.1%} | {a.embed.split('/')[-1]} table "
          f"{by[1] / n:.1%} | teacher {by[2] / n:.1%}")
    if served:
        print(f"  disagreement on everything the cascade served: {wrong / served:.1%}")
    print(f"  mean cost per image ~{cost:.0f} ms (encoder {enc_ms:.0f} ms, teacher {a.vlm_ms:.0f} ms) "
          f"vs {a.vlm_ms:.0f} ms asking the teacher every time")


if __name__ == "__main__":
    main()
