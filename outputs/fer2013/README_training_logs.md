# FER2013 training logs extracted from notebook

- Source notebook: notebooks/emotionclip_reid_jupyterhub_fer2013.ipynb
- Extracted at: 2026-07-03T16:36:31
- Training raw chars: 859656
- Training clean lines: 1850
- Step rows: 1760
- Epoch rows: 50
- Validation rows: 30
- Best validation accuracy line: epoch 8, accuracy 0.7115, balanced_acc 0.6741, macro_f1 0.6833, ece 0.4405
- Best metrics accuracy: 0.705907
- Best metrics balanced_accuracy: 0.683225
- Best metrics macro_f1: 0.686370
- Best metrics ece: 0.374565

Files:
- training_notebook_raw.log (859656 bytes)
- training_notebook_clean.log (859656 bytes)
- training_notebook_context.log (9009 bytes)
- training_events.log (160221 bytes)
- training_step_losses.csv (74754 bytes)
- training_epoch_losses.csv (2234 bytes)
- validation_metrics.csv (1767 bytes)
- best_metrics_summary.json (1087 bytes)
- validation_accuracy.csv: accuracy/balanced_acc/macro_f1 per validation epoch.
- validation_uce.csv: requested uce column; source metric is ece because notebook/repo does not log a separate UCE.
- accuracy_uce_summary.json: best accuracy and lowest ece/uce summary.
- validation_uncertainty.csv: best-checkpoint uncertainty_risk_auc from notebook output.
- uncertainty_summary.json: uncertainty_risk_auc plus sample inference uncertainty.
- inference_sample_uncertainty.json: full single-image inference output including uncertainty.
