# -------- Configurable defaults --------
ENV_NAME ?= dia
PY ?= python
PCG ?= notears          # notears | variational | simple
STEPS ?= 300
FIT_EVERY ?= 10
PCG_EPOCHS ?= 200
MIN_BUFFER ?= 256
BUF_RECENT ?= 1024
LOGDIR ?= runs

# Goal variable per env (safe defaults)
GOAL_MC2D ?= diamond
GOAL_COIN ?= coin_collected
GOAL_CW ?= tower_height
GOAL_CP ?= x
GOAL_MONTE ?= has_key

# -------- Environments --------
env-cpu:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Activate with: conda activate dia"

env-gpu:
	conda env create -f environment-gpu.yml || conda env update -f environment-gpu.yml
	@echo "Activate with: conda activate dia-gpu"

env-sb3-legacy:
	conda env create -f environment-sb3-legacy.yml || conda env update -f environment-sb3-legacy.yml
	@echo "Activate with: conda activate dia-sb3-legacy"

# -------- Install package (editable) --------
install:
	$(PY) -m pip install -e .

# -------- Runners via CLI --------
run:
	$(PY) scripts/dia_cli.py run --env $(ENV) \
	  --pcg $(PCG) --steps $(STEPS) \
	  --fit-every $(FIT_EVERY) --pcg-epochs $(PCG_EPOCHS) \
	  --min-buffer $(MIN_BUFFER) --buffer-recent $(BUF_RECENT) \
	  --logdir $(LOGDIR) $(EXTRA)

mc2d:
	$(PY) scripts/dia_cli.py run --env minecraft2d --pcg $(PCG) --steps $(STEPS) --logdir $(LOGDIR)/mc2d \
	  --goal $(GOAL_MC2D)

coinrun:
	$(PY) scripts/dia_cli.py run --env coinrun --pcg $(PCG) --steps $(STEPS) --logdir $(LOGDIR)/coinrun \
	  --goal $(GOAL_COIN)

causalworld:
	$(PY) scripts/dia_cli.py run --env causalworld --pcg $(PCG) --steps $(STEPS) --logdir $(LOGDIR)/cw \
	  --goal $(GOAL_CW)

cartpole:
	$(PY) scripts/dia_cli.py run --env cartpole --pcg $(PCG) --steps $(STEPS) --logdir $(LOGDIR)/cp \
	  --goal $(GOAL_CP)

montezuma:
	$(PY) scripts/dia_cli.py run --env montezuma --pcg $(PCG) --steps $(STEPS) --logdir $(LOGDIR)/monte \
	  --goal $(GOAL_MONTE)

# --- Interactive menu (select envs/params) ---
menu:
	$(PY) scripts/dia_menu.py

# --- Long sweep over multiple envs ---
sweep-long:
	$(PY) scripts/run_sweep.py --preset long --logroot runs/sweeps

# --- Install Atari ROMs (ALE) ---
atari-roms:
	$(PY) -m autorom --accept-license


# -------- TensorBoard & utils --------
tb:
	tensorboard --logdir $(LOGDIR)

list:
	$(PY) scripts/dia_cli.py list

clean-runs:
	rm -rf runs
