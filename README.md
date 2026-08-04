# Post-Processing of Atmospheric River Persistence

Source code for the study *Post-Processing of Atmospheric River Persistence by Machine Learning: A Pre-trained Tabular Transformer across Mid-latitude West Coasts*.

An atmospheric river (AR) is judged to persist when the observed integrated vapor transport (IVT) stays above a regional threshold at every six hour step within a target window. Comparing a raw D−2 forecast directly against that threshold recovers only about 30% of the 24 hour persistence events. This repository post-processes the forecast instead: a regression model predicts the **minimum observed IVT over the target window** from eight raw forecast values, and that single predicted value is compared with the same regional threshold.

Two regions are treated as fully independent pipelines: California (near San Francisco) and Chile (near Valparaíso).

## Data

| Source | Description |
|---|---|
| GEFS v12 reforecast | Single control member (c00) plus four perturbed members, issued 00Z on D−2, IVT at eight lead times from 48 to 90 hours. Retrieved from `https://noaa-gefs-retrospective.s3.amazonaws.com/` |
| ERA5 reanalysis | Ground truth IVT, computed from the vertical integrals of eastward and northward water vapour flux (`viwve`, `viwvn`), 6 hourly. Retrieved from the `reanalysis-era5-single-levels` dataset via the CDS API |

Period: 2000 to 2019 for the forecasts, 1980 to 2023 for the observations.

## Pipeline

```
gefs_ivt.py                          GEFS v12 reforecast -> gefs_ivt_{region}_d2*.npz
scripts/download_era5_ivt_chile.py   ERA5 -> data/raw/ivt_{region}_1980_2023.npy
scripts/build_circ_chile.py          circulation indices -> data/raw/circ_indices*.npz
scripts/download_openmeteo_weather.py  surface variables -> data/raw/openmeteo_*.npz
build_efeat_npz.py                   environment features -> data/raw/efeat_{region}.npz
build_op_denom_full.py               target days, labels, targets -> opdenom_full_{region}{,_18h,_30h}.npz
build_ens_fcv.py                     five member ensemble -> ens_fcv_{region}{,_18h,_30h}.npz
```

`build_op_denom_full.py` defines the operational denominator (all days whose observed daily maximum IVT is at least 0.5 x threshold), the binary persistence label, the regression target (observed minimum IVT over the window), and the raw forecast baseline.

## Reproducing the results

| Script | Produces |
|---|---|
| `colab_final_2reg.py` | Tables 5, 6, 8, 9. Seven models x three persistence criteria x two regions, with bootstrap confidence intervals |
| `colab_fill_gaps.py` | Table 7. Feature ablation, and the 18 hour block for five of the seven models |
| `colab_18h_tabicl_tabnet.py` | The 18 hour column of Table 9 for TabICL and TabNet |
| `eval_ens_baseline.py` | Table 10. Ensemble decisions used as a baseline against the single control forecast |
| `colab_ens_input.py` | Table 11. Ensemble information added to the model input |

All experiments share the same evaluation design: walk-forward cross validation with five test blocks and a 64 day buffer, predictions pooled across folds, F1 on the persistence class, and a paired case-resampling bootstrap with 2,000 resamples.

`colab_*.py` scripts were run on Google Colab with a GPU. `eval_ens_baseline.py` runs locally on CPU.

```
pip install -r requirements.txt
python build_op_denom_full.py
python build_ens_fcv.py
python eval_ens_baseline.py
```

## Models

TabPFN (proposed), TabICL, TabNet, LightGBM, LSTM, 1d-CNN, and linear regression. All share the same input (eight raw forecast IVT values), the same regression target, and the same decision rule. The pre-trained models are used with default settings; only LightGBM is tuned, using chronological cross validation restricted to the first time block, which is never used for testing.

## License

MIT. See `LICENSE`.