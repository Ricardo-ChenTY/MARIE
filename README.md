# ProveTok

Minimal reproducibility repository for the ProveTok 3D CT report generation pipeline.

## Repository Layout

- `ProveTok_Main_experiment/`: core pipeline modules for Stage 0 to Stage 5
- `Scripts/`: runnable shell and Python entrypoints
- `run_mini_experiment.py`: main experiment launcher
- `train_wprojection.py`: projection training entrypoint
- `validate_stage0_4_outputs.py`: output validator
- `requirements.txt` and `environment.yaml`: environment setup
- `manifests/`: split files and dataset manifests

## Setup

Use either Conda:

```bash
conda env create -f environment.yaml
conda activate provetok_env_py312
```

or pip:

```bash
pip install -r requirements.txt
```

## Main Commands

Projection training:

```bash
python train_wprojection.py
```

Main experiment:

```bash
python run_mini_experiment.py --help
```

5k pipeline:

```bash
bash Scripts/run_stage0_5_5k.sh
```

Full ablation chain:

```bash
bash Scripts/run_5k_full_pipeline.sh
```

Validation:

```bash
python validate_stage0_4_outputs.py --help
```

## Notes

- Dataset manifests are expected under `manifests/`.
- Some Stage 5 scripts require a local Hugging Face model or API backend.
- Model checkpoints and weights are not bundled in this repository.
