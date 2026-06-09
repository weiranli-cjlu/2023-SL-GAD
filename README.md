# SL-GAD

## Setup
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn optuna pandas --torch-backend=cu128
```

## Usage
To train and evaluate on BlogCatalog:
```
python run.py --device cuda:0 --expid 1 --dataset BlogCatalog --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
To train and evaluate on Flickr:
```
python run.py --device cuda:0 --expid 2 --dataset Flickr --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
To train and evaluate on Cora:
```
python run.py --device cuda:0 --expid 3 --dataset cora --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
To train and evaluate on CiteSeer:
```
python run.py --device cuda:0 --expid 4 --dataset citeseer --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.4
```
To train and evaluate on PubMed:
```
python run.py --device cuda:0 --expid 5 --dataset pubmed --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.4
```
To train and evaluate on ACM:
```
python run.py --device cuda:0 --expid 6 --dataset ACM --trials 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.2
```