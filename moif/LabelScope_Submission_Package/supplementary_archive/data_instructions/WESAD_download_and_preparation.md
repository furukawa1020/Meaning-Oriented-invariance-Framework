# WESAD Dataset Download and Preparation

## Source

WESAD (Wearable Stress and Affect Detection) is publicly available from:

**URL**: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx

**Citation**: P. Schmidt et al., "Introducing WESAD, a multimodal dataset for wearable stress and affect detection," in *Proc. ACM Int. Conf. Multimodal Interaction (ICMI)*, 2018, pp. 400--407.

## License

Please refer to the dataset's original license terms before use.

## Download Instructions

1. Visit the URL above and download the WESAD archive.
2. Extract to a local directory, e.g., `data/WESAD/`.

## Expected Directory Structure

```
data/WESAD/
├── S2/
│   ├── S2.pkl
│   └── S2_quest.csv
├── S3/
│   ├── S3.pkl
│   └── S3_quest.csv
...
└── S17/
    ├── S17.pkl
    └── S17_quest.csv
```

## Signal Extraction

Signals used in this study:
- **Chest ECG**: sampled at 700 Hz, resampled to 256 Hz for HRV feature extraction
- **Chest EDA**: sampled at 700 Hz
- **Wrist Accelerometer (ACC)**: sampled at 32 Hz, used for SMA behavioral covariate

Signal magnitude area (SMA) of wrist ACC (with gravity removed):
```
SMA = mean(|ax - g_x| + |ay - g_y| + |az - g_z|)
```
where gravity components are estimated as the mean over the session.

## Preprocessing Parameters

See `configs/preprocessing_config.json` for all parameters including:
- Window size: 60 s
- Window overlap: 50%
- Feature set: 14 ECG + EDA features (see manuscript Section IV-A)
- Label: Stress (1) vs. Non-Stress (0) [Baseline + Amusement]
- Subjects: S2-S17 (N=15, S12 excluded in original dataset)
