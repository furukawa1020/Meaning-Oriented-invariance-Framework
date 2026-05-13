# SWELL-KW Dataset Download and Preparation

## Source

SWELL-KW is publicly available from:

**URL**: https://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html

**Citation**: S. Koldijk et al., "The SWELL-KW dataset for research on stress and user modeling," in *Proc. ACM Int. Conf. Multimodal Interaction (ICMI)*, 2014, pp. 43--50.

## License

Please refer to the dataset's original license terms before use.

## Relevant Files

From the SWELL-KW archive, the following files are used:

- Physiological signals (EDA, BVP, skin temperature) per participant per condition
- Computer interaction log files: `keyboard_mouse_log.csv` per participant

## Expected Structure

```
data/SWELL_KW/
├── pp01/
│   ├── physiological/
│   └── computer_interaction/
│       └── keyboard_mouse_log.csv
...
└── pp25/
```

## Block Aggregation

Procedure for creating 15-min block-level observations:

1. Extract 1-min windows with 50% overlap
2. Aggregate to 15-min blocks (15 windows per block) using `scripts/swell_aggregation.py`
3. Behavioral covariate = sum of keyboard + mouse interaction count per block

## Conditions

- **Neutral**: baseline condition (label = 0)
- **Time Pressure (TP)**: stressor condition (label = 1)
- **Interruption (Intr.)**: stressor condition (label = 1)

For binary PA, TP and Interruption blocks are averaged within each subject
to form one stressor-composite block (see `scripts/swell_aggregation.py`).

## Preprocessing Parameters

See `configs/preprocessing_config.json` for details.
