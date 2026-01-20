# A Transformer-based Multi-Branch Framework for Translation Efficiency in *S. Cerevisiae*

This repository is for a submitted paper: A Transformer-based Multi-Branch Framework for Translation Efficiency in *S. Cerevisiae*



# How to start

1. Download TRIM and TRIM_5UTR for the source codes, preprocessors and best model checkpoints can be found with link: https://zenodo.org/records/18288488?preview=1 (temporary link)

2. Customize your own input in the required format for TRIM to make predictions.(Test input is also given under TRIM/data/test_input.txt.)
3. Download necessary dependencies(Detailed in Dependencies) and change the corresponding file path in TRIM/predict_with_rand_seq.py
4. Run `predict_with_rand_seq.py`
5. View the prediction results under TRIM/outputs/.



# Supplementary File for TRIM

* Datasets:
  * `filtered_dataset.txt`  is the overall dataset for TIRM
  * `filtered_dataset.test.csv`,`filtered_dataset.train.csv`,`filtered_dataset.val.csv` refer to the corresponding dataset in different stages of training, test, and validation.
* Preprocessors:
  * `env_preproc.joblib`, `Gene_utr_struct.jsonl`, `target_TE_value_zscaler.pkl` are TRIM's preprocessors.
* Best model checkpoints:
  * `r2_reg=0.795.ckpt` is the best saved checkpoint for TRIM
  * `r2_reg=0.785.ckpt` is the best saved checkpoint for TRIM_5UTR

# Dependencies

 **Further details to be updated... Last update: 2026.1.18**