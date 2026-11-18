PY = .venv/bin/python
T = $(PY) scripts/timed.py

setup:
	uv venv --python 3.12 .venv
	uv pip install -p .venv/bin/python -r requirements.txt

all: test envcheck demos train eval summarize ddim onnx readme

test:
	$(PY) -m pytest -q tests
envcheck:
	$(T) envcheck "$(PY) scripts/check_env.py > /dev/null"
demos:
	$(T) demos "$(PY) scripts/gen_demos.py"
train:  # three seeds in parallel processes (each uses 2 threads)
	$(T) train "sh scripts/train_all.sh"
eval:
	$(T) eval "$(PY) scripts/evaluate.py --n 200"
summarize:
	$(PY) scripts/summarize.py
ddim:
	$(T) ddim "$(PY) scripts/ddim_steps.py"
onnx:
	$(T) onnx "$(PY) scripts/bench_onnx.py"
readme:
	$(PY) scripts/render_readme.py
