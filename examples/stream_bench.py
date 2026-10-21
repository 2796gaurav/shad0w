"""Video frames in, certified decisions out, reusing the previous answer when that is safe: a reproducible benchmark
for shad0w.stream on a folder of frame images (one sub-folder per clip) or a video file, labelled by your vision LLM.

    pip install "shad0wllm[compile,vision]"
    python examples/stream_bench.py --frames UCSDped2 --labels labels.jsonl \
        --options empty,pedestrians,wheels,unclear --step 3 --block 5 --split 12,10,6

`--labels` is JSON Lines of {"clip": "<clip name>", "frame": <0-based index in the clip>, "vlm": "<answer>"}: the
teacher's answer on every `step`-th frame (null / missing = the teacher gave none). The stream runs on exactly those
frames (frame 0, step, 2 step, ...), so the teacher's answer is known wherever the stream would call it. Clips are
dealt into fitting / calibration / test clips (`--split`, stratified by the rarest answer); the tables are fitted on
the fitting clips, the certificate on one row per block of the calibration clips, and everything reported below is
measured on the test clips, which touch neither.

Reports, with and without the reuse tier: the share of frames answered by reuse / the frame table / the teacher,
end-to-end disagreement with the teacher on certified frames, disagreement on transition frames (where the teacher's
answer changed since the previous labelled frame), microseconds per frame (numpy; and an estimate for the C core),
the teacher's own flicker rate on near-identical consecutive frames, and the certificates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shad0w.stream import Stream, _popcount, frame_state, stream_compile  # noqa: E402
from shad0w.vision import ImageSpec  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_folder(root: str, step: int, exclude: tuple[str, ...]):
    """{clip name: [frames at 0, step, 2 step, ...]} from every leaf folder of images under root."""
    from PIL import Image
    clips = {}
    for d, _, files in sorted(os.walk(root, followlinks=True)):
        names = sorted(f for f in files if f.lower().endswith(IMAGE_EXT))
        name = os.path.basename(d)
        if len(names) < 2 or any(x in name for x in exclude):
            continue
        frames = []
        for i in range(0, len(names), step):
            im = Image.open(os.path.join(d, names[i]))
            frames.append(np.asarray(im.convert("L") if im.mode in ("L", "I;16", "I") else im.convert("RGB")))
        clips[name] = frames
    return clips


def load_video(path: str, step: int, clip_frames: int):
    """{"<video>#<k>": frames} with the video cut into clips of `clip_frames` source frames (needs cv2 or av)."""
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(np.ascontiguousarray(f[:, :, ::-1]))
            i += 1
    except ImportError:
        try:
            import av
        except ImportError as e:
            raise SystemExit("reading a video needs opencv-python or av: pip install opencv-python") from e
        with av.open(path) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i % step == 0:
                    frames.append(fr.to_ndarray(format="rgb24"))
    per = max(1, clip_frames // step)
    base = os.path.splitext(os.path.basename(path))[0]
    return {f"{base}#{k}": frames[k * per:(k + 1) * per] for k in range((len(frames) + per - 1) // per)}


def load_labels(path: str, clips: dict, step: int, options: list[str]):
    """answers per clip aligned with the subsampled frames (None where the teacher gave no answer)."""
    answers = {name: [None] * len(fr) for name, fr in clips.items()}
    n, n_unknown = 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            name, i = r.get("clip"), r.get("frame")
            if name not in answers or i is None or i % step:
                continue
            j = i // step
            if j >= len(answers[name]):
                continue
            a = r.get("vlm", r.get("answer"))
            if a is not None and a not in options:
                n_unknown += 1
                a = None
            answers[name][j] = a
            n += 1
    return answers, n, n_unknown


def deal(names: list[str], answers: dict, split: tuple[int, int, int], options: list[str], rotate: int = 0):
    """Stratified deal of clips into (fit, cal, test): clips sorted by their share of the rarest answer, then dealt
    in a fixed interleaving so that every split sees the rare answer; `rotate` shifts the interleaving, which
    deals a different set of clips to each split (for repeated runs on other test clips)."""
    counts = {o: sum(a == o for n in names for a in answers[n]) for o in options}
    rare = min((o for o in options if counts[o]), key=counts.get, default=options[0])
    order = sorted(names, key=lambda n: -np.mean([a == rare for a in answers[n]] or [0]))
    total = sum(split)
    seq, given = [], [0, 0, 0]
    for k in range(len(order)):
        want = [split[j] * (k + 1) / total - given[j] for j in range(3)]
        j = int(np.argmax(want))
        given[j] += 1
        seq.append(j)
    r = rotate % len(seq)
    seq = seq[r:] + seq[:r]
    out = ([], [], [])
    for n, j in zip(order, seq):
        out[j].append(n)
    return out


def run_stream(model, clips: dict, answers: dict, names: list[str], reuse: bool):
    """Play the test clips through a Stream whose teacher is the recorded answer. Returns the metrics."""
    cur = {"clip": None}

    def teacher(frame):
        i = cur["i"]
        return answers[cur["clip"]][i]

    s = Stream(model, reuse=reuse, teacher=teacher)
    rows = []
    for name in names:
        s.reset()
        prev = None
        for i, f in enumerate(clips[name]):
            cur["clip"], cur["i"] = name, i
            d = s.decide(f)
            y = answers[name][i]
            rows.append((d, y, y is not None and prev is not None and y != prev))
            if y is not None:
                prev = y
    lab = [(d, y, t) for d, y, t in rows if y is not None]
    served = [(d, y, t) for d, y, t in lab if d.certified]
    by = {src: [(d, y) for d, y, t in served if d.source == src] for src in ("reuse", "table")}
    trans = [(d, y) for d, y, t in lab if t]
    trans_served = [(d, y) for d, y in trans if d.certified]
    n = len(rows)
    out = {"frames": n, "labelled": len(lab),
           "share": s.stats()["share"],
           "certified_share": len(served) / max(len(lab), 1),
           "disagreement_certified": (sum(d.answer != y for d, y, t in served) / len(served)) if served else None,
           "disagreement_by_source": {k: (sum(d.answer != y for d, y in v) / len(v) if v else None)
                                      for k, v in by.items()},
           "transition_frames": len(trans), "transition_served": len(trans_served),
           "transition_disagreement": ((sum(d.answer != y for d, y in trans_served) / len(trans_served))
                                       if trans_served else None),
           "transition_served_by": {src: sum(d.source == src for d, y in trans_served) for src in ("reuse", "table")},
           "mean_latency_us": float(np.mean([d.latency_us for d, _, _ in rows])),
           "p50_latency_us": float(np.median([d.latency_us for d, _, _ in rows])),
           "answer_changes": s.stats()["answer_changes"], "keyframes": s.stats()["keyframes"]}
    return out


def teacher_flicker(states: dict, answers: dict, names: list[str], max_hd: int):
    n = d = 0
    for name in names:
        st, ans = states[name], answers[name]
        for i in range(1, len(st)):
            if ans[i] is not None and ans[i - 1] is not None and _popcount(st[i].dhash ^ st[i - 1].dhash) <= max_hd:
                n += 1
                d += ans[i] != ans[i - 1]
    return {"pairs": n, "flicker": d / n if n else None}


def native_estimate(spec: ImageSpec, frame) -> float | None:
    """Microseconds of the C core's pixels -> items on one frame (the stream's delta items add a few more)."""
    try:
        from shad0w import native
        if not native.available():
            return None
        nat = native.NativeImage(spec)
    except Exception:
        return None
    for _ in range(50):
        nat.items(frame)
    t = time.perf_counter()
    for _ in range(500):
        nat.items(frame)
    return (time.perf_counter() - t) / 500 * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder with one sub-folder of frame images per clip")
    src.add_argument("--video", help="a video file (cut into clips of --clip-frames source frames)")
    ap.add_argument("--labels", required=True, help="JSON Lines {clip, frame, vlm}: the vision model's answers")
    ap.add_argument("--options", required=True, help="comma-separated answer options")
    ap.add_argument("--step", type=int, default=1, help="the stream runs on every step-th source frame")
    ap.add_argument("--clip-frames", type=int, default=300, help="--video: source frames per clip")
    ap.add_argument("--exclude", default="_gt", help="comma-separated substrings of clip folders to skip")
    ap.add_argument("--split", default="", help="clips for fit,cal,test (e.g. 12,10,6); default 45/35/20 percent")
    ap.add_argument("--block", type=int, default=15,
                    help="frames per block: one calibration row, max age of a reused answer")
    ap.add_argument("--blocks", default="", help="also certify at these block sizes (comma-separated)")
    ap.add_argument("--hamming", type=int, default=12, help="dHash bits that end a keyframe's reign")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--channels", type=int, default=0, help="1 or 3 (default: the frames')")
    ap.add_argument("--ferns", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate", type=int, default=0, help="deal other clips to fit / cal / test (see deal())")
    ap.add_argument("--procedure", default="auto", choices=["auto", "single", "cascade"],
                    help="certificate candidates (shad0w.stream.certify_rows); pick before looking at results")
    ap.add_argument("--student", default="linear", help="the frame table's student (linear / deep / auto)")
    ap.add_argument("--out", default="", help="write results.json here")
    a = ap.parse_args()
    options = [o.strip() for o in a.options.split(",") if o.strip()]

    t0 = time.time()
    clips = (load_folder(a.frames, a.step, tuple(x for x in a.exclude.split(",") if x)) if a.frames
             else load_video(a.video, a.step, a.clip_frames))
    if len(clips) < 3:
        raise SystemExit(f"{len(clips)} clips; need at least 3 (fit, calibrate, test)")
    answers, n_lab, n_unknown = load_labels(a.labels, clips, a.step, options)
    names = sorted(clips)
    print(f"{len(clips)} clips, {sum(map(len, clips.values()))} frames at step {a.step}, {n_lab} labelled "
          f"({n_unknown} outside the options), loaded in {time.time() - t0:.1f}s")
    ch = a.channels or (3 if next(iter(clips.values()))[0].ndim == 3 else 1)
    spec = ImageSpec(channels=ch, ferns=a.ferns)
    if a.split:
        split = tuple(int(x) for x in a.split.split(","))
    else:
        n = len(names)
        split = (n - round(0.35 * n) - max(1, round(0.2 * n)), round(0.35 * n), max(1, round(0.2 * n)))
    fit, cal, test = deal(names, answers, split, options, a.rotate)
    print(f"fit {len(fit)} clips, calibrate {len(cal)} clips, test {len(test)} clips")
    ordered = fit + cal
    cal_ix = list(range(len(fit), len(ordered)))

    t = time.time()
    states = {n: [frame_state(f, spec) for f in clips[n]] for n in names}
    feat_us = (time.time() - t) / sum(map(len, clips.values())) * 1e6
    flick = {f"hd<={h}": teacher_flicker(states, answers, names, h) for h in (0, 2, 4)}
    results = {"dataset": a.frames or a.video, "step": a.step, "procedure": a.procedure, "rotate": a.rotate,
               "block": a.block, "hamming": a.hamming, "alpha": a.alpha, "delta": a.delta, "options": options,
               "spec": spec.to_dict(), "clips": {"fit": fit, "cal": cal, "test": test},
               "n_frames": sum(map(len, clips.values())), "n_labelled": n_lab, "n_unknown": n_unknown,
               "answer_counts": {o: sum(x == o for v in answers.values() for x in v) for o in options},
               "teacher_flicker": flick, "frame_state_us": feat_us,
               "native_items_us": native_estimate(spec, clips[names[0]][0]), "runs": {}}
    for label, reuse in (("with_reuse", True), ("no_reuse", False)):
        t = time.time()
        model, cert = stream_compile([clips[n] for n in ordered], [answers[n] for n in ordered], options, spec=spec,
                                     alpha=a.alpha, delta=a.delta, block=a.block, hamming=a.hamming, cal_clips=cal_ix,
                                     reuse=reuse, seed=a.seed, teacher=os.path.basename(a.labels),
                                     procedure=a.procedure, student=a.student)
        comp_s = time.time() - t
        ev = run_stream(model, clips, answers, test, reuse)
        e = cert["questions"]["frame"]
        results["runs"][label] = {"certificate": e, "test": ev, "compile_s": comp_s}
        print(f"\n== {label} (compiled in {comp_s:.0f}s) ==")
        print(f"certificate: {e['n_calibration']} rows from {e['n_blocks']} blocks of {a.block}; certified share "
              f"on calibration {e['certified_share_on_calibration']:.1%}, realised "
              f"{e['disagreement_on_certified_calibration']}; tiers "
              + ", ".join(f"{t_['source']} {t_['share']:.1%} (thr {t_['threshold']})" for t_ in e["tiers"]))
        sh = ev["share"]
        print(f"test clips: reuse {sh['reuse']:.1%}  table {sh['table']:.1%}  teacher {sh['teacher']:.1%}"
              f"  uncertified {ev['share']['uncertified']:.1%}")
        print(f"disagreement with the teacher on certified frames: {ev['disagreement_certified']}  (reuse "
              f"{ev['disagreement_by_source']['reuse']}, table {ev['disagreement_by_source']['table']})")
        print(f"transition frames: {ev['transition_frames']}, served {ev['transition_served']}, "
              f"disagreement {ev['transition_disagreement']}")
        print(f"latency: mean {ev['mean_latency_us']:.0f} us, median {ev['p50_latency_us']:.0f} us per frame (numpy)")
    if a.blocks:
        results["block_sweep"] = {}
        for b in (int(x) for x in a.blocks.split(",")):
            model, cert = stream_compile([clips[n] for n in ordered], [answers[n] for n in ordered], options, spec=spec,
                                         alpha=a.alpha, delta=a.delta, block=b, hamming=a.hamming, cal_clips=cal_ix,
                                         seed=a.seed,
                                         procedure=a.procedure, student=a.student)
            e = cert["questions"]["frame"]
            ev = run_stream(model, clips, answers, test, True)
            results["block_sweep"][str(b)] = {"certificate": e, "test": ev}
            print(f"block {b}: {e['n_calibration']} rows, certified {e['certified_share_on_calibration']:.1%} "
                  f"(reuse {e['tiers'][0]['share']:.1%}); test reuse {ev['share']['reuse']:.1%} "
                  f"teacher {ev['share']['teacher']:.1%} "
                  f"disagreement {ev['disagreement_certified']}")
    print(f"\nteacher flicker on near-identical consecutive frames: {flick}")
    print(f"frame featurisation: {feat_us:.0f} us/frame numpy; C core items: {results['native_items_us']} us")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
