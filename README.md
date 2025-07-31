<div align="center">
    <h1>GCLUSE: A novel deep graph clustering approach with similarity
aware embeddings</h1>
</div>


# Requirements
> [!NOTE]
> Higher versions should be also compatible.

* torch
* torchvision
* torchaudio
* torch-scatter
* torch-sparse
* torch-cluster
* munkres
* kmeans-pytorch
* Scipy
* Scikit-learn

```bash
pip install -r requirements.txt
```


# Reproduction

> The same code can be used for Citeseer, Amazon-Photo and Amazon-Computers by changing the dataset name.

* Cora
  ```
  !python train.py --runs 1 --dataset 'Computers' --hidden '512' --1_1 100 --l_2 --tau 0.5 --ns 0.5 --lr 0.0005 --epochs_sim 150 --epochs_cluster 150 --wd 1e-3 --alpha 0.9
  ```
