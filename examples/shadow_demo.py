"""Real-data shadow demo: reproduces the shape of the headline result on public data in ~5-10 minutes.

    pip install -e ".[demo]"
    python examples/shadow_demo.py [--teacher bge-small|minilm|gemma-300m] [--dataset banking77|clinc150]

Teacher = a frozen open sentence encoder used zero-shot (option-name similarity) — a stand-in for "the model you
already run". The demo logs the teacher's answers on the train split (shadow), compiles and certifies the table
against the teacher on a held-out slice, then measures on the official test split:
agreement with the teacher, the share answered locally, realised disagreement, and cascade accuracy vs gold.
Numbers differ by teacher; the published table in docs/BENCHMARKS.md used EmbeddingGemma-300m and Kev-0.8B.
"""
import argparse
import time

import numpy as np

from shad0w.compiler.encoders import Encoder
from shad0w.shadow import shadow_compile


def load(name):
    from datasets import load_dataset
    if name == "banking77":
        d = load_dataset("mteb/banking77")
        labels = sorted(set(zip(d["train"]["label"], d["train"]["label_text"])))
        opts = [t.replace("_", " ") for _, t in labels]
        return (list(d["train"]["text"]), list(d["test"]["text"]), np.array(d["test"]["label"]), opts)
    if name == "clinc150":
        d = load_dataset("clinc/clinc_oos", "plus")
        names = d["train"].features["intent"].names
        keep = [i for i, n in enumerate(names) if n != "oos"]
        remap = {i: j for j, i in enumerate(keep)}
        tr = [t for t, y in zip(d["train"]["text"], d["train"]["intent"]) if y in remap]
        te = [(t, remap[y]) for t, y in zip(d["test"]["text"], d["test"]["intent"]) if y in remap]
        return tr, [t for t, _ in te], np.array([y for _, y in te]), [names[i].replace("_", " ") for i in keep]
    raise SystemExit(f"unknown dataset {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="bge-small")
    ap.add_argument("--dataset", default="banking77")
    a = ap.parse_args()
    train, test, y_test, options = load(a.dataset)
    enc = Encoder(a.teacher)
    L = enc.encode(options, "option")
    t0 = time.time()
    teacher_train = (enc.encode(train) @ L.T).argmax(1)            # shadow: log the teacher's answers
    teacher_test = (enc.encode(test) @ L.T).argmax(1)
    print(f"teacher ({a.teacher}) zero-shot accuracy on test: {(teacher_test == y_test).mean():.1%}  [{time.time()-t0:.0f}s]")
    schema = {"intent": {"type": "choice", "criteria": {o: None for o in options}}}
    log = [{"text": t, "intent": options[k]} for t, k in zip(train, teacher_train)]
    for alpha in (0.02, 0.05):
        t0 = time.time()
        m, cert = shadow_compile(schema, log, alpha=alpha, teacher=a.teacher)
        served = dis = 0
        casc = []
        for text, tk, gold in zip(test, teacher_test, y_test):
            r = m.decide(text)["answers"]["intent"]
            pred = options.index(r["choice"])
            if r["certified"]:
                served += 1; dis += pred != tk; casc.append(pred == gold)
            else:
                casc.append(tk == gold)                              # deferred to the teacher
        print(f"alpha={alpha:.0%}: answered locally {served/len(test):.1%}, realised disagreement "
              f"{(dis/served if served else 0):.1%}, cascade accuracy {np.mean(casc):.1%} "
              f"(teacher alone {(teacher_test == y_test).mean():.1%})  [compile+certify {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
