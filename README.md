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
The RainPro-8 architecture, which leverages multi-source data, is implemented in `rainpro/network/rainpro8.py`. A demonstration notebook showcasing its usage is also provided (`rainpro8.ipynb`).

## RainPro-8 (Taiwan)

This module trains/evaluates the RainPro-8 model on Taiwan data as an obs-only baseline: QPESUMS (radar) + STA_H8 (satellite). GFS is available as an optional additional input (`--data.include_gfs true`), to compare the obs-only arm against obs+GFS. See `docs/rainpro_dataset.md` for the source mapping.

Fill in the zarr store paths in `rainpro8.yml` (`data.data_root`), then:

```
python main_rainpro8.py fit <run_name> --config rainpro8.yml
```

To evaluate:

```
python main_rainpro8.py test --config runs/<run_name>/config.yml <run_name> --ckpt_path runs/<run_name>/checkpoints/best.ckpt [--thresholds]
```

Implementation notes / what still needs project-specific calibration before real training:

- **Store schema**: `rainpro/data/rainpro8_dataset.py` assumes each zarr store exposes a `time` dimension (where applicable) and `lat`/`lon` (or `XLAT`/`XLONG`) coordinates; use `data.variable_aliases`/`data.latlon_names` in the config if the real stores name things differently (e.g. STA_H8's per-pixel lookup table).
- **GFS normalization**: only sources with well-known physical ranges (radar dBZ, IR brightness temperature) have default min-max bounds (`rainpro/data/normalize.py`). Any GFS variables (`--data.include_gfs true`) are left un-normalized until per-variable training-set statistics are computed and passed via `data.norm_bounds`.
- **Spatial regridding** (`rainpro/data/regrid.py`) is nearest-neighbor on a local equirectangular approximation, not a full geographic reprojection.
- **Canvas geometry**: target/context sizes intentionally reuse the original European RainPro-8 geometry (`rainpro/data/rainpro8_sources.py`) since the network hard-assumes exact 2x relationships between resolution tiers; Taiwan's native (non-square, smaller) source coverage is embedded in these canvases with out-of-coverage pixels treated as missing.
- **No 16km tier without GFS**: with `include_gfs=False` there is no 16km-tier source at all; `rainpro.network.rainpro8.RainPro` detects the 0-channel tier (`in_dims`) and skips that branch of the encoder rather than embedding an empty input.

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