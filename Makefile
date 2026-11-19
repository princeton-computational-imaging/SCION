# Reproducing the paper (docs/reproducing_results.md).
#
#   make dry-run                     list the jobs and check every config composes
#   make reproduce                   train and evaluate every job on local GPUs, then render
#   make reproduce BACKEND=slurm     the same jobs as a Slurm array, then a render job
#   make render                      tables and figure panels from existing runs, no training
#   make tables                      tables only, on CPU, from the records in evals/runs
#   make figure1 RUN_DIR=<run dir>   Figure 1 panels from one trained run
#
# Variables: TARGETS (default all; names from experiments/paper/targets.yaml),
# GPUS (e.g. 0,1; default all visible), JOBS_PER_GPU (default 1),
# PAPER_DIR (default paper_artifacts), ARGS (other flags, e.g. --allow-dirty).
# DATA_ROOT, TMC13_BINARY and SLURM_* are read from reproduce.env when it
# exists; copy reproduce.env.example.

TARGETS      ?= all
BACKEND      ?= local
GPUS         ?=
JOBS_PER_GPU ?= 1
PAPER_DIR    ?= paper_artifacts
ARGS         ?=

REPRODUCE = set -a; [ ! -f reproduce.env ] || . ./reproduce.env; set +a; \
	python -m experiments.paper.reproduce $(TARGETS) --paper-dir $(PAPER_DIR) $(ARGS)

.PHONY: help reproduce dry-run render tables figure1

# First target, so a bare `make` prints the usage instead of starting a run.
help:
	@sed -n '/^#/!q; s/^# \{0,1\}//p' $(firstword $(MAKEFILE_LIST))

reproduce:
	$(REPRODUCE) --backend $(BACKEND) --jobs-per-gpu $(JOBS_PER_GPU) $(if $(GPUS),--gpus $(GPUS))

dry-run:
	$(REPRODUCE) --dry-run

render:
	$(REPRODUCE) --render-only

tables:
	python -m experiments.paper.render all --runs-root evals/runs --out $(PAPER_DIR)

EVAL_INDEX ?= 0
OUTPUT     ?= $(PAPER_DIR)/figures/figure1_panels
figure1:
	@if [ -z "$(RUN_DIR)" ]; then \
	    echo "error: RUN_DIR=<evals/runs/.../seed<N>> is required" >&2; exit 2; \
	fi
	python scripts/build_figure1_panels.py \
	    --run-dir     $(RUN_DIR) \
	    --eval-index  $(EVAL_INDEX) \
	    --output      $(OUTPUT) \
	    $(if $(MAX_CANDS),--max-candidates $(MAX_CANDS))
