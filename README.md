# RainPro - ICLR 2026
Official implementation of "[RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours](https://arxiv.org/abs/2505.10271)"


## Getting Started

Install the [`uv`](https://docs.astral.sh/uv/) package manager.

Setup dependencies:

```bash
uv sync
source .venv/bin/activate
```

## RainPro-2R (SEVIR Benchmark)
This module evaluates the RainPro architecture and Ordinal Consistent Loss on the [SEVIR benchmark](https://registry.opendata.aws/sevir/), using radar-only inputs for precipitation forecasting up to 2 hours ahead.

To train RainPro-2R, run the following command:

```
python main.py fit <run_name> --config rainpro2r.yml --data.data_root <sevir data>
```

To evaluate the model, run the following command:

```
python main.py test --config runs/<run_name>/config.yml  <run_name> --ckpt_path runs/<run_name>/checkpoints/best.ckpt [--thresholds]
```

This loads the model from the specified checkpoint and evaluates it on the test set, applying optimized thresholds (calculating them if they are not already available).

Checkpoints and precomputed thresholds are available in [Google Drive](https://drive.google.com/drive/folders/1Vfv7arDCNUL3oKVS_n0wsoFy7ELHTlGp?usp=sharing). Download and place them in: `runs/<run_name>/checkpoints/`

## RainPro-8 (Architecture)
Coming soon!

## Citation
If you find our work useful for your research, please cite our [paper](https://arxiv.org/abs/2505.10271):
```
@InProceedings{sarabia2026rainpro,
  title     = {RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours},
  author    = {Rafael Pablos Sarabia and Joachim Nyborg and Morten Birk and Jeppe Liborius Sj{\o}rup and Anders Lillevang Vesterholt and Ira Assent},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
}
```

## Acknowledgements
The SEVIR benchmark framework is based on [DeminYu98/DiffCast](https://github.com/DeminYu98/DiffCast), specifically dataset preprocessing, sample generation, training configuration, and evaluation metrics.