# Architecture Revision Plan Notebooks

Generated from `ARCHITECTURE_REVISION_PLAN.md`. Each priority has a separated pretrain and finetune notebook.

Outputs are isolated under `outputs/revise_plan/<priority_id>/pretrain` and `outputs/revise_plan/<priority_id>/finetune`.

## Priority 1 - Temporal Context Expansion

A1 beat windowing with 2-second context, A2 multi-beat sequence encoding, and A3 RR-interval feature hooks.

- `notebook_revise_plan/priority_01_temporal_context/pretrain_priority_01_temporal_context.ipynb`
- `notebook_revise_plan/priority_01_temporal_context/finetune_priority_01_temporal_context.ipynb`

## Priority 2 - Learnable Anatomical Priors

B1 per-class prior strength, B2 learnable prior matrix with regularization, and B3 prior evolution visualization hooks.

- `notebook_revise_plan/priority_02_learnable_priors/pretrain_priority_02_learnable_priors.ipynb`
- `notebook_revise_plan/priority_02_learnable_priors/finetune_priority_02_learnable_priors.ipynb`

## Priority 3 - MI-Specific ECG Feature Extraction

C1 ST-segment and T-wave feature extraction modules fused with CNN lead tokens.

- `notebook_revise_plan/priority_03_mi_specific_features/pretrain_priority_03_mi_specific_features.ipynb`
- `notebook_revise_plan/priority_03_mi_specific_features/finetune_priority_03_mi_specific_features.ipynb`

## Priority 4 - Spatial-Anatomical Encoding

D1 lead positional encoding and D2 lead neighborhood attention based on anatomical lead distances.

- `notebook_revise_plan/priority_04_spatial_anatomical_encoding/pretrain_priority_04_spatial_anatomical_encoding.ipynb`
- `notebook_revise_plan/priority_04_spatial_anatomical_encoding/finetune_priority_04_spatial_anatomical_encoding.ipynb`

## Priority 5 - Class Imbalance Handling

E1 focal loss and E2 per-class prior scaling for rare MI localization classes.

- `notebook_revise_plan/priority_05_class_imbalance/pretrain_priority_05_class_imbalance.ipynb`
- `notebook_revise_plan/priority_05_class_imbalance/finetune_priority_05_class_imbalance.ipynb`

## Priority 6 - Interpretability and Validation

F1 Grad-CAM, F2 prior-attention agreement, and F4 class activation maps for MI types.

- `notebook_revise_plan/priority_06_interpretability_validation/pretrain_priority_06_interpretability_validation.ipynb`
- `notebook_revise_plan/priority_06_interpretability_validation/finetune_priority_06_interpretability_validation.ipynb`

