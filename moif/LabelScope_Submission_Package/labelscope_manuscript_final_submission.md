# LabelScope: Auditing State-Label Claim Support in Physiology-Based Human-Machine Systems

**Abstract**
Physiology-based human-machine systems (HMS) often train predictive models on operational labels such as questionnaire scores, task conditions, behavioral errors, or affective annotations. However, high predictive accuracy does not by itself establish that the target label can support the HMS claim made from the model. A label may be temporally sparse, behaviorally confounded, structurally unsupported in physiological feature space, or sensitive to preprocessing choices. In such cases, model performance may reflect the learnability of an operational proxy rather than evidence for a physiology-grounded user-state claim.

This paper introduces LabelScope, a conservative pre-modeling audit framework for claim-relative state-label validity in physiology-based HMS. LabelScope audits whether an operational label can support a specified HMS claim under a given modeling resolution, proxy interpretation, feature representation, and deployment context. The framework consists of Resolution, Proxy, Structure, and Claim audits, and assigns a capped claim level using a conservative claim-capping rule.

We illustrate LabelScope through implementation sanity checks and audit demonstrations using public physiological datasets, including SWELL-KW, CASE, and WESAD. These demonstrations illustrate how LabelScope caps unsupported claims arising from resolution mismatch, behavioral proxy confounding, random-like physiological neighborhood structure, and pipeline-dependent effects. LabelScope does not determine whether a psychological state exists or whether a construct is invalid; rather, it specifies the strongest defensible HMS claim supported by the tested operational label and pipeline. The framework provides a reporting discipline for interpreting predictive accuracy together with audited claim levels in physiology-based HMS research.

---

## I. Introduction
Physiology-based human-machine systems (HMS) aim to improve interaction by monitoring user states through physiological signals such as heart rate, skin conductance, and brain activity. By building computational models that sense and respond to internal states—such as mental workload, stress, or affect—researchers seek to create adaptive systems that enhance user performance and well-being. This objective has driven extensive research into physiological feature extraction and predictive modeling, supported by increasingly sophisticated machine learning algorithms and a growing number of public physiological datasets.

In current HMS research, predictive accuracy is frequently treated as the primary metric for evaluating a sensing pipeline. A model that achieves high accuracy in predicting an operational label—such as a questionnaire score or a task condition—is often interpreted as evidence of the underlying psychological state. However, we argue that this focus on predictive accuracy creates a reporting trap. High accuracy measures the learnability of an operational target within a specific dataset; it does not, by itself, justify the scope of the HMS claim made from that model. The central problem addressed in this paper is not whether physiological signals can be modeled, but whether the operational labels used as modeling targets can support the HMS claims made from those models.

This problem arises because operational labels in physiology-based HMS may fail to support the claims made from physiological models. A high-accuracy model may be targeting a label that is temporally sparse (leading to pseudo-replication artifacts), behaviorally confounded (reflecting user activity rather than internal state), or structurally unsupported (indistinguishable from random noise in physiological feature space). When these constraints are ignored, HMS research risks making over-interpreted claims that lead to misdirected interventions and difficult-to-reproduce findings. To address this, we propose that state-label validity should be reported as a pre-modeling audit discipline before predictive performance is interpreted as HMS evidence.

In this paper, we define **state-label validity** as a claim-relative property of operational labels in physiology-based HMS. Under this definition, a label is not valid or invalid in isolation; rather, its defensible claim level depends on the specific HMS inference intended by the researcher. We propose **LabelScope**, a pre-modeling audit framework designed to assign conservative claim levels based on four distinct audit modules: Resolution, Proxy, Structure, and Claim audits. LabelScope does not determine whether a psychological state exists; instead, it specifies the strongest defensible HMS claim that can be made from a given operational label, feature representation, and modeling resolution.

This paper makes three contributions:
1. We define state-label validity as a **claim-relative property** of operational labels in physiology-based HMS, shifting the focus from construct truth to claim supportability.
2. We propose **LabelScope**, a formal audit framework that assigns defensible claim levels based on systematic triggers for pseudo-replication, behavioral confounding, and physiological neighborhood structure.
3. We **illustrate LabelScope through audit demonstrations** across synthetic controls and several public physiological datasets (SWELL-KW, CASE, and WESAD). We illustrate how unsupported HMS claims are conservatively capped before predictive accuracy is interpreted.

---

## II. Related Work

### II-A. Physiology-Based HMS and Affective Computing
Physiology-based HMS are closely related to affective computing, which established the broader goal of building computational systems that relate to, arise from, or influence human affect. Picard’s foundational work positioned affective computing as a field concerned not only with recognizing emotion, but also with designing systems that respond to affective information in human-computer interaction. In physiology-based HMS, this agenda is often operationalized through pipelines that extract physiological features and train models to predict affective, stress-related, workload-related, or performance-related labels.

The field has evolved through several decades of modeling surveys and meta-analyses. Picard et al. (2001) highlighted the early potential for machine emotional intelligence, while Calvo and D’Mello (2010) provided a comprehensive review of affect detection across diverse modalities. More recently, Martinez et al. (2017) and Bota et al. (2019) have surveyed the state of the art in physiological-based mental status monitoring and emotion recognition. Despite significant advances in predictive accuracy, this literature leaves open a reporting gap: how to specify what kind of HMS claim is actually supported by the operational label being predicted. LabelScope addresses this gap by shifting the focus to the claim supportability of the operational label within a given physiological modeling pipeline.

### II-B. Public Physiological Datasets and Operational Labels
The development of physiology-based HMS has been accelerated by the availability of public datasets. SWELL-KW was collected specifically for stress and user modeling in knowledge work settings, incorporating stressors such as time pressure and email interruptions alongside block-level subjective assessments. CASE (Continuous Affective States Evaluation) provides continuous valence and arousal annotations with synchronized physiological recordings, while WESAD (Wearable Stress and Affect Detection) offers a benchmark for stress and affect detection using wearable sensors.

These datasets are essential not because their labels are intrinsically valid for all HMS applications, but because they expose different forms of label-resolution, behavioral proxy, and pipeline-dependency problems. For instance, the block-level labels in SWELL-KW are frequently used for minute-level prediction, creating a risk of pseudo-replication. Continuous annotations in CASE raise questions about temporal alignment, while WESAD and CASE both expose how normalization choices and dataset-specific baselines can shape downstream claims. We use these datasets in this paper to illustrate how such failure modes can be diagnosed and used to cap HMS claims, ensuring that modeling convenience is not mistaken for support for a physiology-based HMS claim.

### II-C. Operational Labels, Construct Validity, and Subjective Annotation
The problem addressed by LabelScope is related to, but narrower than, construct validity. Cronbach and Meehl’s (1955) formulation of construct validity traditionally concerns the evidential basis for interpreting a measure as reflecting a theoretical attribute. This tradition is critical for HMS because operational labels are often treated as if they directly and uniquely indexed latent user states.

LabelScope does not attempt to adjudicate these broader psychological questions. Instead, it addresses a specific pipeline-level question: whether an operational label can support a specified physiology-based HMS claim at a given modeling resolution, proxy interpretation, feature representation, and deployment context. This distinction is essential because the same label may support one claim while failing to support another. By treating labels as claim-relative targets rather than intrinsically valid or invalid constructs, LabelScope provides a conservative auditing layer for engineering-focused HMS research.

### II-D. Documentation, Auditing, and Claim Boundaries in ML/HMS
LabelScope is also related to recent work on documentation and accountability in machine learning. Model Cards (Mitchell et al., 2019) propose standardized documentation for trained models, and Datasheets for Datasets (Gebru et al., 2021) propose structured documentation of dataset properties. These frameworks aim to improve transparency and prevent misinterpretation of ML artifacts.

LabelScope adds a physiology-specific claim-capping layer for HMS. Whereas model cards document properties of trained models and datasheets document properties of datasets, LabelScope audits the relation among physiological features, operational labels, behavioral covariates, temporal resolution, and intended HMS claims. Its output is a capped claim level indicating the strongest defensible HMS interpretation under the tested pipeline. In this sense, LabelScope contributes a reporting discipline for physiology-based HMS: model performance should be reported together with the audited claim level, the capping module, and the recommended modeling action.

Taken together, prior work provides physiological sensing pipelines, operational labels, benchmark datasets, and general documentation practices. However, it does not provide a physiology-specific mechanism for specifying how far an HMS claim can be supported by a given operational label under a particular modeling resolution, proxy interpretation, feature representation, and preprocessing pipeline. LabelScope addresses this gap by auditing candidate operational labels before predictive accuracy is interpreted as evidence of user-state modeling.

---

## III. State-Label Validity in Physiology-Based HMS

### III-A. The Interpretability Premise
Physiology-based HMS frequently rely on operational labels as proxies for unobservable psychological states. A common assumption in HMS research is that high predictive accuracy demonstrates the validity of these labels. We argue that this assumption is incomplete because it does not examine whether the operational label can support the intended HMS claim under the tested physiological pipeline. Accuracy is a measure of model fit, not of the defensible scope of the intended HMS claim.

We therefore propose the **Interpretability Premise**: the interpretability of a physiological sensing model depends on the audited relationship between the operational label and the physiological feature space. Within this framework, validity is not an intrinsic property of a label; it is a **claim-relative property**. A label is evaluated only with respect to the specific HMS claim it intends to support at a given resolution, representation, and behavioral context.

### III-B. Establishing Defensible Claim Levels
The establishment of **defensible claim levels** should be reported as a pre-modeling audit discipline before predictive performance is interpreted as HMS evidence. Before a sensing model is deployed for HMS intervention, the operational labels must be audited to ensure the intended HMS claim does not generalize beyond the scope supported by the physiological evidence. LabelScope provides the formal framework for this auditing process.

---

## IV. The LabelScope Framework

### IV-A. System Formalization
LabelScope is implemented as a pre-modeling audit pipeline. Given a physiological feature table $X$, a candidate operational label $y$, behavioral covariates $C_B$, and an intended HMS claim $S$, LabelScope returns audit metrics, module-wise claim levels, and a final **capped claim level** $L_{Final}$. LabelScope is a conservative reporting discipline; each audit module assigns the maximum defensible claim level supported by the tested pipeline.

### IV-B. Audit Modules

#### 1) Resolution Audit (RA)
RA evaluates pseudo-replication risk. We define $R_P = N_{model}/N_{ind}$. A high $R_P$ combined with low $N_{ind}$ **caps the claim at Level 0** at the intended resolution because the modeling resolution is not supported by a sufficient number of independent label observations.

#### 2) Proxy Audit (PA)
PA evaluates material association with behavioral covariates using a confounding index:
$$\rho_{LC} = \max_{c \in C_B}\lvert\mathrm{Assoc}(y,c)\rvert$$
High $\rho_{LC}$ **caps the claim at Level 2**, indicating that the label should be interpreted as behavior- or proxy-specific rather than as a pure internal state label.

#### 3) Structure Audit (SA)
SA-Structure (Structure Test): Tests whether local label organization is stronger than a shuffled-label control using \(B\) shuffled-label permutations:
\[
\hat{p}_{shuf}
=
\frac{
1 + \sum_{b=1}^{B} \mathbf{1}\left(T_{shuf}^{(b)} \ge T_{obs}\right)
}{
B + 1
}.
\]
If \(\hat{p}_{shuf} \ge \alpha\), the claim is capped at Level 1.

SA-Ambiguity (Ambiguity Test): Evaluated only when SA-Structure does not cap the claim below Level 3. A Level 3 candidate assignment is given only when $A_{local}$ falls within a pre-specified range $A_{low} \le A_{local} \le A_{high}$.

### IV-C. Claim Capping Rule
The final claim level $L_{Final}$ is the minimum of the module-wise assignments:
$$L_{Final} = \min(L_{RA}, L_{PA}, L_{SA}, L_{CA})$$

### IV-D. Claim Ladder (Defensible Scope)
- **Level 0**: No Modeling Claim at the Intended Resolution
- **Level 1**: Dataset- or Pipeline-Specific Observation
- **Level 2**: Behavior/Proxy-Specific Observation
- **Level 3**: Audited Structural Ambiguity
- **Level 4**: Candidate Valid State Label

---

## V. Audit Demonstrations

### V-A. Demonstration Design
We apply LabelScope to several candidate operational labels to **illustrate the claim-capping behavior of the framework**. The purpose is not to establish universal validity but to **illustrate how unsupported physiology-based claims are conservatively restricted** under different failure modes.

### V-B. Synthetic Sanity Checks
We used controlled synthetic cases as implementation sanity checks to **check** whether the audit modules produce the expected claim caps under controlled patterns of pseudo-replication and proxy confounding. These checks **examine** whether the audit pipeline reproduces the intended decision rules and are not treated as empirical validation of the framework.

### V-C. SWELL-KW: Resolution and Proxy Failure Modes
SWELL-KW is used to illustrate how LabelScope caps physiology-based HMS claims when candidate labels are temporally sparse, behaviorally confounded, or unsupported by detectable neighborhood structure.

For block-level subjective labels such as MentalEffort and Performance(recoded), the intended claim was minute-level monitoring. However, the independent label observations were collected at the block level (75 independent blocks vs. over 3,000 minute-level samples). LabelScope therefore identified a resolution mismatch and capped the claim at Level 0 at the intended resolution. This indicates that block-level subjective labels do not support minute-level physiology-based HMS claims without risking pseudo-replication.

For error_rate, the label has higher temporal density, but the Proxy Audit indicated material association with typing activity ($\rho_{LC} > 0.45$). Therefore, the defensible claim is behavior- or proxy-specific (Level 2) rather than a pure internal-state claim. This result illustrates how higher temporal resolution does not necessarily support a stronger HMS claim if behavioral confounding is present.

For residual objective error, defined as the component of error behavior not explained by behavioral volume, the Structure Audit did not identify local physiological neighborhood structure distinguishable from shuffled controls. The claim was therefore capped at Level 1, indicating that the result should be treated as a dataset-specific observation rather than evidence of a physiology-grounded work-error tendency.

### V-D. CASE and WESAD: Auxiliary Claim-Restriction Cases
CASE and WESAD illustrate how LabelScope restricts claims when apparent findings remain sensitive to contrast definitions, temporal controls, or preprocessing choices.

For CASE valence and arousal annotations, local label mixing was observed in physiological neighborhoods. However, the observed contrast was insufficient to support a stronger **valence-specific physiological underdetermination claim**. The claim is therefore capped at Level 1, as local mixing alone is not sufficient for a Level 3 claim without pre-specified non-degenerate ambiguity and non-random neighborhood structure.

For temporal lag analyses, apparent alignment between physiological signals and state labels was sensitive to slow temporal structure. Under the tested controls, the lagged alignment could not be separated from low-frequency drift or autocorrelation artifacts. LabelScope therefore restricts the claim to Level 1 rather than allowing a general claim about optimal physiological response lag.

For normalization-dependent effects, improvements induced by baseline normalization were found to be sensitive to dataset-specific offsets and baseline choices. LabelScope treats such findings as pipeline-sensitive (Level 1) unless the effect is decomposed from subject-level offsets. The capped claim is therefore that normalization affects the tested pipeline, not that it reveals a general user-state structure.

### V-E. Claim Revision Examples
The practical role of LabelScope as a claim-revision mechanism is **illustrated** through the following three case studies, showing how audit findings translate into restricted HMS interpretations.

First, in the case of SWELL-KW MentalEffort, the original claim of minute-level physiological monitoring was revised to a **block-level subjective claim**. The audit revealed a severe resolution mismatch (Level 0 cap), as the 75 independent questionnaire observations could not support the over 3,000 minute-level modeling targets. This revision forces the researcher to either collect higher-frequency labels or restrict the HMS claim to coarse, block-level summaries where independent evidence is available.

Second, for the SWELL-KW error_rate label, the original claim of detecting internal impairment was revised to a **behavior/proxy-specific claim**. Although the temporal resolution was high, the Proxy Audit identified a material association with physical typing volume (Level 2 cap). The revised claim acknowledges that the detected markers may reflect behavioral activity rather than purely internal state changes, requiring the HMS to be reported as a behavioral proxy model under the tested pipeline.

Third, in the CASE dataset, the original claim of valence-specific physiological underdetermination was revised to a **dataset- or pipeline-specific affective observation**. While local label mixing was observed, the lack of pre-registered structural targets and insufficient contrast for **a valence-specific physiological underdetermination claim** led to a Level 1 cap. This revision ensures that the finding is not over-claimed as a general property of affective physiology, but is correctly reported as an observation sensitive to the CASE dataset and the specific feature representation used.

---

## VI. Discussion

### VI-A. Why State-Label Auditing Must Precede Accuracy Interpretation
Predictive accuracy cannot be interpreted independently of the operational label being predicted. A model may achieve high performance while still targeting a label that is temporally sparse or behaviorally confounded. By requiring that accuracy be reported alongside an audited claim level, LabelScope ensures that modeling convenience is not mistaken for support for a physiology-based HMS claim.

### VI-B. Implications for Physiology-Based HMS Design
The necessity of state-label auditing has direct implications for HMS deployment. If the operational label used during training is behaviorally confounded, the resulting system may trigger interventions based on an unsupported interpretation of the operational label. LabelScope guides researchers toward more robust HMS designs and should be treated as an engineering safeguard.

### VI-C. Claim-Capping as a Reporting Discipline
LabelScope serves as a reporting discipline that restricts what may be claimed from a given model. This prevents over-interpreted and difficult-to-reproduce claims in physiological computing. It defines the boundaries of defensible inference rather than rejecting datasets or constructs outright.

### VI-D. Limitations
LabelScope is not a "validity oracle." First, audit thresholds are conservative triggers, not universal boundaries. Second, the Structure Audit is sensitive to feature representation and neighborhood definitions. Third, the demonstrations are illustrative and not prospective validation. Fourth, LabelScope does not replace domain-specific validation. Fifth, the present demonstrations are retrospective. Finally, Level 4 remains a future validation category.

### VI-E. Future Work
Future work should evaluate LabelScope in prospective HMS studies where thresholds are pre-specified. Studies should collect higher-frequency independent labels when high-frequency claims are intended. Structure Audit could be extended to alternative representations. Finally, LabelScope could be integrated with Model Cards and dataset documentation.
