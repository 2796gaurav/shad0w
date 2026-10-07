PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
LIB := shad0w/libreflex.so

.PHONY: all test demo lint build clean

all: $(LIB)

# the C core, in place, for a source checkout (pip wheels ship it prebuilt as shad0w._reflex)
$(LIB): shad0w/_native/reflex.c
	cc -O3 -Wall -shared -fPIC -o $@ $< -lm

test: $(LIB)
	$(PY) -m pytest -q tests

demo: $(LIB)
	$(PY) examples/shadow_demo.py

lint:
	$(PY) -m ruff check shad0w tests examples

build:
	$(PY) -m build && $(PY) -m twine check dist/*

clean:
	rm -f shad0w/libreflex.* shad0w/_reflex*.so shad0w/_reflex*.pyd
	rm -rf build dist *.egg-info .pytest_cache examples/out
