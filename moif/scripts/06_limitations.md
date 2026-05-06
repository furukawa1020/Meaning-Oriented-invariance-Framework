# 06 Limitations

## Context within THMS Paper
Acknowledging boundaries explicitly is crucial for a Transactions paper to prevent reviewers from attacking out-of-scope aspects.

## Key Limitations to Acknowledge

1. **Binary Task Formulation**: We analyzed binary state discrimination (Baseline vs. Active). Real-world HMS applications often involve continuous, multidimensional subjective states (e.g., arousal-valence circumplex).
2. **Public Dataset Constraints**: WESAD and CASE are highly controlled laboratory datasets. While they provide excellent standardized benchmarks, they lack the true unpredictable hardware drift, multi-day sensor wear, and environmental noise of free-living (ambulatory) deployments.
3. **Fixed Classifier Model**: To isolate causal effects, we intentionally restricted our main analysis to fixed-parameter Logistic Regression. While robust, state-of-the-art HMS may employ deep learning architectures (e.g., LSTMs, Transformers) whose complex feature representations might exhibit different sensitivities to normalization choices.
4. **Subjective Label Ambiguity**: The labels "Stress" and "Arousal" rely on the assumption that the experimental stimuli perfectly induced the intended subjective states across all participants uniformly, which is inherently flawed in affective computing.
