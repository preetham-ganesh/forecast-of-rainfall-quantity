# Forecast of Rainfall Quantity and its Variation using Environmental Features

[Preetham Ganesh](https://www.linkedin.com/in/preethamganesh/), [Harsha Vardhini Vasu](https://www.linkedin.com/in/harshavardhini1/), [Dayanand Vinod*](https://in.linkedin.com/in/dayanand-vinod).

[[IEEE]](https://ieeexplore.ieee.org/document/8960026), [[PDF]](https://preetham-ganesh.github.io/website/documents/forecast_rainfall.pdf)

## Contents

- [Setup](https://github.com/preetham-ganesh/forecast-of-rainfall-quantity#setup)
- [Data Preprocessing](https://github.com/preetham-ganesh/forecast-of-rainfall-quantity#data-preprocessing)

## Setup

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements

Requires: Python 3.6.

```bash
# Clone this repository
git clone https://github.com/preetham-ganesh/forecast-of-rainfall-quantity.git
cd forecast-of-rainfall-quantity

# Create a Conda environment with dependencies
conda env create -f environment.yml
conda activate forq_env
pip install -r requirements.txt
```

## Data Preprocessing

```bash
python3 data_preprocessing.py
python3 elbow_method.py
python3 clustering_and_data_splitting.py
```

## Model Training and Testing

```bash
python3 models.py
```

## Variation Analysis

```bash
python3 variation_analysis.py
```

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Citation

If you use this code, please cite the following:

```
@INPROCEEDINGS{8960026,
  author={Ganesh, Preetham and Vasu, Harsha Vardhini and Vinod, Dayanand},
  booktitle={2019 Innovations in Power and Advanced Computing Technologies (i-PACT)}, 
  title={Forecast of Rainfall Quantity and its Variation using Environmental Features}, 
  year={2019},
  volume={1},
  number={},
  pages={1-8},
  doi={10.1109/i-PACT44901.2019.8960026}}
```
