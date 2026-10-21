"""Images as decisions: pixels -> items (C and numpy agree), a certified table over them, and the Shadow wrapper."""
import json

import numpy as np
import pytest

from shad0w import native
from shad0w.vision import ImageSpec, image_digest, image_items, image_items_py, pixels, thumbnail

COLOURS = {"red": (200, 30, 30), "green": (30, 180, 40), "blue": (30, 40, 200), "dark": (25, 25, 25)}


def synth_images(n, seed, noise=0.02):
    """Shapes on a noisy background; the class is the shape's colour. The 'teacher' is right 98% of the time."""
    rng = np.random.default_rng(seed)
    keys = list(COLOURS)
    imgs, labels = [], []
    for _ in range(n):
        y = keys[rng.integers(len(keys))]
        h, w = rng.integers(24, 64, 2)
        im = rng.integers(150, 256, (h, w, 3)).astype(np.uint8)
        y0, x0 = rng.integers(0, h // 2), rng.integers(0, w // 2)
        im[y0:y0 + h // 2, x0:x0 + w // 2] = np.clip(np.array(COLOURS[y]) + rng.integers(-20, 21, 3), 0, 255)
        imgs.append(im)
        labels.append(y if rng.random() > noise else keys[rng.integers(len(keys))])
    return imgs, labels


SPECS = [ImageSpec(), ImageSpec(size=16, channels=1, ferns=64, bits=6, radius=3, grid=2, levels=8, margin=0, seed=7),
         ImageSpec(size=64, ferns=4096, bits=16, grid=16, levels=256), ImageSpec(version=1, dhash=False),
         ImageSpec(size=9, channels=1, ferns=8, grid=0), ImageSpec(dhash=False, channels=1)]
SHAPES = [(32, 32, 3), (28, 28), (1, 1, 3), (3, 500, 4), (480, 640, 3), (7, 5), (224, 224, 4)]


@pytest.mark.skipif(not native.available(), reason="C core not built")
@pytest.mark.parametrize("spec", SPECS)
def test_c_and_numpy_items_agree(spec):
    rng = np.random.default_rng(0)
    nat = native.NativeImage(spec)
    for shape in SHAPES:
        a = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(image_items_py(a, spec), nat.items(a)), shape


def test_items_shape_determinism_and_identity():
    spec = ImageSpec()
    a = np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    its = image_items(a, spec)
    assert its.dtype == np.uint32 and len(its) == spec.n_items and (its < 2**22).all()
    assert np.array_equal(its, image_items(a.copy(), spec))
    assert not np.array_equal(its, image_items(a, ImageSpec(seed=1)))  # another seed is another featurizer
    assert image_digest(a) == image_digest(a.copy()) != image_digest(a[::-1])
    assert thumbnail(a, spec).shape == (32, 32, 3)
    # floats in [0, 1] and single-channel arrays are accepted
    assert pixels(a / 255.0).dtype == np.uint8 and pixels(a[:, :, :1]).ndim == 2


def test_ferns1_bundles_still_load_and_dhash_marks_duplicates():
    old = ImageSpec.from_dict({"kind": "ferns/1", "size": 32, "channels": 3, "ferns": 256, "bits": 8, "radius": 2,
                               "grid": 4, "levels": 4, "margin": 2, "seed": 0})
    assert old.version == 1 and not old.dhash and old.n_items == 304 and old.to_dict()["kind"] == "ferns/1"
    with pytest.raises(ValueError):
        ImageSpec(version=1, dhash=True)
    rng = np.random.default_rng(4)
    a = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    noisy = np.clip(a.astype(int) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    band = lambda x: set(image_items(x)[-4:].tolist())
    assert len(band(a) & band(noisy)) >= 2 and not band(a) & band(other)


def test_spec_validation_and_roundtrip():
    with pytest.raises(ValueError):
        ImageSpec(size=65)
    with pytest.raises(ValueError):
        ImageSpec(channels=2)
    s = ImageSpec(ferns=100, seed=3)
    assert ImageSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    with pytest.raises(ValueError):
        ImageSpec.from_dict({"kind": "clip/9"})


def test_certified_image_table_end_to_end(tmp_path):
    """Logged (image, answer) pairs -> a certified table that answers images; the bound holds on fresh images."""
    import shad0w

    imgs, labels = synth_images(1600, 0)
    test, truth = synth_images(800, 1, noise=0.0)
    answers = {**dict(zip(map(image_digest, test), truth)), **dict(zip(map(image_digest, imgs), labels))}
    calls = []

    def vision_llm(img):  # stands in for a vision model; Shadow hands it the image itself
        calls.append(1)
        return answers[image_digest(img)]

    sh = shad0w.Shadow(str(tmp_path / "bundle"), teacher=vision_llm, question="colour", log=str(tmp_path / "log.jsonl"),
                       schema={"type": "choice", "criteria": {k: None for k in COLOURS}}, image=True, audit_rate=0.0)
    for im in imgs:
        assert sh(im).source == "teacher"
    row = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert "text" not in row and len(row["items"]) == ImageSpec().n_items and len(row["image"]) == 64
    cert = sh.train()
    assert cert["accepted"] and cert["certified_share_on_calibration"] > 0.5

    ds = [sh(im) for im in test]
    served = [d for d in ds if d.source == "table"]
    assert len(served) > 0.5 * len(test)
    wrong = sum(d.answer != t for d, t in zip(ds, truth) if d.source == "table")
    assert wrong / len(served) <= 0.05
    # a reloaded bundle remembers its featurizer and refuses a different one
    again = shad0w.Shadow(str(tmp_path / "bundle"), question="colour")
    assert again.image_spec == ImageSpec() and again(test[0]).source == "table"
    with pytest.raises(ValueError, match="featurises images"):
        shad0w.Shadow(str(tmp_path / "bundle"), question="colour", image=ImageSpec(seed=9))
    with pytest.raises(TypeError, match="takes images"):
        again("some text")


@pytest.mark.skipif(not native.available(), reason="C core not built")
def test_native_image_decision_matches_python(tmp_path):
    from shad0w.shadow import shadow_compile

    imgs, labels = synth_images(600, 2)
    spec = ImageSpec(ferns=64)
    rows = [{"items": image_items(im, spec).tolist(), "colour": y} for im, y in zip(imgs, labels)]
    m, _ = shadow_compile({"colour": {"type": "choice", "criteria": {k: None for k in COLOURS}}}, rows,
                          image=spec.to_dict())
    m.save(str(tmp_path))
    q = m.questions["colour"]
    table = native.NativeReflex(str(tmp_path / "colour.s0"))
    nat = native.NativeImage(spec)
    for im in imgs[:50]:
        its = image_items_py(im, spec)
        y_py = int(np.argmax(q.rx.logits(its)))
        y_c, p_c, _ = nat.decide(table, im)
        y_i, p_i, _ = table.decide_items(its)
        assert y_c == y_i == y_py and np.allclose(p_c, p_i)


def test_teacher_receives_image_parts():
    from shad0w.llm import LLMTeacher, image_data_url

    pytest.importorskip("PIL")
    t = LLMTeacher({"colour": {"type": "choice", "criteria": {k: None for k in COLOURS}}}, "openai/x", question="colour",
                   api_key="sk-test", complete=lambda msgs: "red")
    im = np.zeros((8, 8, 3), np.uint8)
    msgs = t.messages(im)
    part = msgs[-1]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    assert t(im) == "red"
    assert image_data_url(b"\xff\xd8\xff\xe0rest").startswith("data:image/jpeg;base64,")


def _colour_embedder(img):
    """Stands in for CLIP in tests: a normalised colour histogram (fast, deterministic)."""
    a = pixels(img).reshape(-1, 3).astype(np.float32)
    h = np.concatenate([np.histogram(a[:, c], bins=16, range=(0, 256))[0] for c in range(3)]).astype(np.float32)
    return h / (np.linalg.norm(h) + 1e-9)


def test_image_cascade_one_certificate(tmp_path):
    """ferns (us) -> embedding tier (ms) -> teacher, one certificate; the embedding tier takes what ferns pass on."""
    import shad0w
    from shad0w.vision import EmbedSpec, register_embedder

    register_embedder("colour-hist", _colour_embedder)
    imgs, labels = synth_images(1800, 5)
    test, truth = synth_images(600, 6, noise=0.0)
    answers = {**dict(zip(map(image_digest, test), truth)), **dict(zip(map(image_digest, imgs), labels))}
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=lambda im: answers[image_digest(im)], question="colour",
                       log=str(tmp_path / "log.jsonl"), schema={"type": "choice", "criteria": {k: None for k in COLOURS}},
                       image=[ImageSpec(ferns=32), EmbedSpec("colour-hist")], audit_rate=0.0)
    for im in imgs:
        sh(im)
    row = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert len(row["embed"]) == 48 and len(row["items"]) == ImageSpec(ferns=32).n_items
    cert = sh.train()
    tiers = cert["tiers"]
    assert cert["accepted"] and len(tiers) == 2 and cert["procedure"].endswith("cascade-fixed-sequence")
    ds = [sh(im) for im in test]
    served = [(d, t) for d, t in zip(ds, truth) if d.source == "table"]
    assert len(served) > 0.5 * len(test)
    assert sum(d.answer != t for d, t in served) / len(served) <= 0.05
    # reloaded from disk: the cascade, its fitted bucket edges and its thresholds come back
    again = shad0w.Shadow(str(tmp_path / "b"), question="colour")
    assert len(again._serve_tiers) == 2 and again._serve_tiers[1][1].fitted
    assert [again(im).answer for im in test[:20]] == [d.answer for d in ds[:20]]
