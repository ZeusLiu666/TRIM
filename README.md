# A Transformer-based Multi-Branch Framework for Translation Efficiency in *S. Cerevisiae*

This repository is for a submitted paper: A Transformer-based Multi-Branch Framework for Translation Efficiency in *S. Cerevisiae*

# How to clone

1. Clone the repository:

  ```bash
  git clone https://github.com/ZeusLiu666/TRIM.git
  cd TRIM
  ```

2. Build with mamba

  ```bash
  conda env create -f environment.yml
  conda activate TRIM
  ```


# Essential Files Needed

1. You'll need to download the following files **FOR ALL POSSIBLE PURPOSES**:

  * `target_TE_value_zscaler.pkl` for Z-score normalization and denormalization;
  
  * `Gene_utr_struct.jsonl` for secondary structure information;

  * `env_preproc.joblib` for env input encoding.

2. Put the files above under a specified path.

  * You can choose wherever you like. If so, don't forget to accordingly change the path in the codes.

  * The path is set by default at:
  ```TRIM/data/preprocessor/```

These files can be downloaded from zenodo: https://zenodo.org/records/18288488?preview=1 (temporary link)

# Predict with the best model weights

1. The data preprocessors and best checkpoints can be found on zenodo: https://zenodo.org/records/18288488?preview=1 (temporary link)

  * For just loading the best model, you need only `r2_reg=0.795.ckpt` for TRIM and `r2_reg=0.785.ckpt` for TRIM_5UTR. 

2. Put the best checkpoint under a specified path.

  * You can choose wherever you like. If so, don't forget to accordingly change the path in the codes.

  * The path of TRIM is set by default at:
  ```
  TRIM/outputs/logs/version_0/checkpoints/best-epoch=022-val/r2_reg=0.795.ckpt
  ```

  * The path of TRIM_5UTR is set by default at:
  ```
  TRIM_5UTR/outputs/logs/version_0/checkpoints/local200-epoch=160-val/r2_reg=0.785.ckpt
  ```

## For TRIM to make prediction:
1. Customize your own input in the required format for TRIM to make predictions.

* The example input `example_input.csv` is updated on zenodo.

2. Specify the file you want to predict. Run `predict_with_rand_seq.py`.

```bash
python predict_with_rand_seq.py
```

3. View the prediction result under the same input path. The output file replace the input file's `.csv` with `_with_pred.csv`. 

## For TRIM_5UTR to make prediction:

1. Customize your own input in the required format for TRIM_5UTR to make predictions.

* The example input `test_input.csv` is updated on zenodo.

2. Specify the file you want to predict. Run `predict_5U.py`.

```bash
python predict_5U.py
```

3. View the prediction result under the output path: `outputs/predict/`

# Supplementary File for TRIM

* Datasets:

  * `filtered_dataset.txt`  is the overall dataset for TIRM

  * `filtered_dataset.test.csv`,`filtered_dataset.train.csv`,`filtered_dataset.val.csv` refer to the corresponding dataset in different stages of training, test, and validation.

* Preprocessors:

  * `env_preproc.joblib`, `Gene_utr_struct.jsonl`, `target_TE_value_zscaler.pkl` are TRIM's preprocessors.

* Best model checkpoints:

  * `r2_reg=0.795.ckpt` is the best checkpoint for **TRIM**

  * `r2_reg=0.785.ckpt` is the best checkpoint for **TRIM_5UTR**

# Dependencies
```bash
  - python==3.10
  - pip
  - numpy==1.22.4
  - pandas==1.4.2
  - scipy==1.8.1
  - statsmodels==0.14.6
  - biopython==1.86
  - openpyxl==3.1.5
  - tqdm==4.67.1
  - tensorboardx==2.6.2.2
  - mlflow==3.7.0
  - ipykernel
  - torch==2.5.1+cu124
  - torchvision==0.20.1+cu124
  - pytorch-lightning==2.6.0
  - torchmetrics==1.8.2
  - logomaker==0.8.7
  - python-docx==1.2.0
  - python-dotenv==1.2.1
  - pure_eval==0.2.3
  - scikit-learn==1.0.2
  - matplotlib==3.5.2
  - matplotlib-inline==0.1.6
  - seaborn>=0.13.2
  - viennarna=2.7.2
```