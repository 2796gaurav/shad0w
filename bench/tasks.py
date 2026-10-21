"""The benchmark's tasks: public fixed-option decision datasets, each split the same way for every system.

Every task gives (train, calibration, test) rows of (text, label index) and its option names. The split is fixed by
seed 0 and never depends on the system being measured:

  test         a fixed random sample of the dataset's own test split (`n_test` rows)
  calibration  a fixed random sample of the train split, held out from training (`n_cal` rows)
  train        the rest of the train split

Systems that learn (shad0w, the classic baselines) fit on `train`; every system's confidence is certified on
`calibration` by the same procedure; all numbers are reported on `test`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@dataclass
class Task:
    name: str
    title: str
    question: str  # the instruction a decision model gets
    options: list[str]  # option ids (what systems answer)
    descriptions: dict[str, str]  # option id -> short description for decision models
    train: list[tuple[str, int]]
    cal: list[tuple[str, int]]
    test: list[tuple[str, int]]
    source: str


def _humanize(label: str) -> str:
    return re.sub(r"[_\-]+", " ", label).strip()


def _split(train_rows, test_rows, n_cal: int, n_test: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    tr = list(train_rows)
    order = rng.permutation(len(tr))
    cal = [tr[i] for i in order[:n_cal]]
    fit = [tr[i] for i in order[n_cal:]]
    te = list(test_rows)
    te = [te[i] for i in np.random.default_rng(seed + 1).permutation(len(te))[:n_test]]
    return fit, cal, te


def load(name: str, n_cal: int = 1000, n_test: int = 1000) -> Task:
    from datasets import load_dataset
    if name == "banking77":
        d = load_dataset("mteb/banking77")
        names = sorted({(r["label"], r["label_text"]) for r in d["train"]})
        opts = [t for _, t in names]
        rows = lambda s: [(r["text"], r["label"]) for r in d[s]]  # noqa: E731
        fit, cal, te = _split(rows("train"), rows("test"), n_cal, n_test)
        return Task(name, "BANKING77 intents", "Which banking customer-support intent does this message express?", opts,
                    {o: _humanize(o) for o in opts}, fit, cal, te, "mteb/banking77 (Casanueva et al. 2020), CC BY 4.0")
    if name == "clinc150":
        d = load_dataset("DeepPavlov/clinc_oos", "plus")
        names = dict(sorted({(r["label"], r["label_text"]) for r in d["train"]}))
        opts = [names[i] for i in range(len(names))]
        rows = lambda s: [(r["text"], r["label"]) for r in d[s]]  # noqa: E731
        fit, cal, te = _split(rows("train"), rows("test"), n_cal, n_test)
        desc = {o: ("anything outside the other intents" if o == "oos" else _humanize(o)) for o in opts}
        return Task(name, "CLINC150 intents + out-of-scope", "Which assistant intent does this request express?", opts,
                    desc, fit, cal, te, "DeepPavlov/clinc_oos 'plus' (Larson et al. 2019), CC BY 3.0")
    if name == "sst2":
        d = load_dataset("stanfordnlp/sst2")
        opts = ["negative", "positive"]
        rows = [(r["sentence"], r["label"]) for r in d["train"]]
        # SST-2's test labels are hidden: its validation split (872 rows) is the test set, as is standard
        fit, cal, te = _split(rows, [(r["sentence"], r["label"]) for r in d["validation"]], n_cal, n_test)
        return Task(name, "SST-2 sentiment", "Is the sentiment of this movie-review phrase positive or negative?", opts,
                    {"negative": "negative sentiment", "positive": "positive sentiment"}, fit, cal, te,
                    "stanfordnlp/sst2 (Socher et al. 2013); test = validation split")
    if name == "ag_news":
        d = load_dataset("fancyzhx/ag_news")
        opts = ["world", "sports", "business", "sci_tech"]
        fit, cal, te = _split([(r["text"], r["label"]) for r in d["train"]], [(r["text"], r["label"]) for r in d["test"]],
                              n_cal, n_test)
        fit = fit[:30000]  # a realistic log size; keeps every system's training time comparable
        return Task(name, "AG News topics", "Which topic is this news article about?", opts,
                    {"world": "world news, politics", "sports": "sports", "business": "business, economy",
                     "sci_tech": "science and technology"}, fit, cal, te, "fancyzhx/ag_news (Zhang et al. 2015)")
    if name == "sms_spam":
        d = load_dataset("ucirvine/sms_spam")["train"]
        opts = ["ham", "spam"]
        allr = [(r["sms"], r["label"]) for r in d]
        order = np.random.default_rng(7).permutation(len(allr))  # no official test split: a fixed 25% holdout
        cut = len(allr) * 3 // 4
        fit, cal, te = _split([allr[i] for i in order[:cut]], [allr[i] for i in order[cut:]], n_cal, n_test)
        return Task(name, "SMS spam", "Is this SMS message spam?", opts,
                    {"ham": "a normal message", "spam": "spam or unsolicited advertising"}, fit, cal, te,
                    "ucirvine/sms_spam (Almeida et al. 2011); fixed 25% holdout as test")
    if name == "prompt_injection":
        d = load_dataset("xTRam1/safe-guard-prompt-injection")
        opts = ["safe", "injection"]
        fit, cal, te = _split([(r["text"], r["label"]) for r in d["train"]], [(r["text"], r["label"]) for r in d["test"]],
                              n_cal, n_test)
        return Task(name, "Prompt injection", "Is this user input a prompt-injection or jailbreak attempt?", opts,
                    {"safe": "a normal request", "injection": "tries to override instructions or jailbreak the model"},
                    fit, cal, te, "xTRam1/safe-guard-prompt-injection")
    raise KeyError(name)


TASKS = ["banking77", "clinc150", "sst2", "ag_news", "sms_spam", "prompt_injection"]
