# Makefile (Conda‑based) – Choreo2Groove pipeline
# ===============================================================
# Requires:
#   • Conda env named $(CONDA_ENV) (create once: conda create -y -n choreo2groove python=3.10)
#   • GNU make on PATH (Windows: choco install make)
# All commands are executed via `conda run -n`, so no activate hacks.
# ===============================================================

# ---------- User‑tunable vars ----------
CONDA_ENV   ?= choreo2groove
CONDA_RUN   := conda run -n $(CONDA_ENV)
PY          := $(CONDA_RUN) python
PIP         := $(CONDA_RUN) pip
RAW_ROOT    ?= aist_raw
DATA_ROOT   ?= dataset_root
EPOCHS      ?= 30
BS          ?= 16
SEQ_LEN     ?= 512
LR          ?= 1e-4
MODEL_SCRIPT:= choreo2groove.py
SCRIPTS_DIR := scripts

DOWNLOAD_SCRIPT   := $(SCRIPTS_DIR)/download_aistpp.sh
EXTRACT_SCRIPT    := $(SCRIPTS_DIR)/extract_pose.py
TRANSCRIBE_SCRIPT := $(SCRIPTS_DIR)/transcribe_drums.py
SYNC_SCRIPT       := $(SCRIPTS_DIR)/sync_pose_beats.py

.PHONY: all deps download extract transcribe sync dataset train sample clean help

all: deps dataset train      # default full pipeline

# ---------------------------------------------------------------
# 1 · Dependency install   (Conda‑forge for compiled wheels, pip for rest)
# ---------------------------------------------------------------
deps:
	@conda env list | grep -q "$(CONDA_ENV)" || ( \
	  echo "✗ Conda env '$(CONDA_ENV)' not found. Create it first:" && \
	  echo "    conda create -y -n $(CONDA_ENV) python=3.11" && exit 1 )
	$(PIP) install --quiet --upgrade \
	    git+https://github.com/google/aistplusplus_api.git \
	    pretty_midi==0.2.10 \
	    miditoolkit==0.1.17 \
	    pytorch-lightning>=2.2 \
	    librosa tqdm \
	    boto3 pandas pillow joblib absl-py==1.4.0 requests

# ---------------------------------------------------------------
# 2 · Data preparation
# ---------------------------------------------------------------
download: $(DOWNLOAD_SCRIPT)
	bash $(DOWNLOAD_SCRIPT) $(RAW_ROOT)

extract: $(EXTRACT_SCRIPT) download
	$(PY) $(EXTRACT_SCRIPT) --npz_root $(RAW_ROOT)/annotations --out_root $(DATA_ROOT)

transcribe: $(TRANSCRIBE_SCRIPT) download
	$(PY) $(TRANSCRIBE_SCRIPT) --audio_root $(RAW_ROOT)/audio --out_root $(DATA_ROOT)

sync: $(SYNC_SCRIPT) extract transcribe
	$(PY) $(SYNC_SCRIPT) --dataset_root $(DATA_ROOT)

dataset: sync

# ---------------------------------------------------------------
# 3 · Training / sampling
# ---------------------------------------------------------------
train: dataset
	$(PY) $(MODEL_SCRIPT) --data_root $(DATA_ROOT) \
	    --epochs $(EPOCHS) --batch_size $(BS) \
	    --seq_len $(SEQ_LEN) --lr $(LR)

sample:
	$(PY) utils/sample.py --model_ckpt lightning_logs/latest.ckpt \
	    --pose_np pose.npy --out_midi generated.mid

# ---------------------------------------------------------------
clean:
	rm -rf $(DATA_ROOT) $(RAW_ROOT) lightning_logs 2>/dev/null || true

help:
	@echo "Targets:" && \
	echo "  all          deps + dataset + train (default)" && \
	echo "  deps         install Python deps in Conda env" && \
	echo "  download     fetch AIST++" && \
	echo "  extract      NPZ → pose.npy" && \
	echo "  transcribe   audio → drums.mid" && \
	echo "  sync         align pose & MIDI" && \
	echo "  train        train Lightning model" && \
	echo "  sample       quick generation demo" && \
	echo "  clean        remove data & logs" && \
	echo "Variables: RAW_ROOT DATA_ROOT EPOCHS BS SEQ_LEN LR CONDA_ENV"
