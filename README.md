# DisenIDP

This repo provides a reference implementation of DisenIDP as described in the paper:

[Enhancing Information Diffusion Prediction with Self-Supervised Disentangled User and Cascade Representations](https://scholar.google.com.hk/citations?view_op=view_citation&hl=zh-CN&user=CU28LO0AAAAJ&citation_for_view=CU28LO0AAAAJ:IWHjjKOFINEC)  
Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023  
[dl.acm.org/doi/abs/10.1145/3583780.3615230](https://dl.acm.org/doi/abs/10.1145/3583780.3615230)

## Dependencies
Install the dependencies via [Anaconda](https://www.anaconda.com/):
+ Python (>=3.9)
+ PyTorch (>=2.0.1)
+ NumPy (>=1.26.1)
+ Scipy (>=1.7.3)
+ torch-geometric(>=2.3.1)
+ tqdm(>=4.66.1)

```python
# create virtual environment
conda create --name DisenIDP python=3.9

# activate environment
conda activate DisenIDP

#install pytorh from pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

# install other dependencies
pip install -r requirements.txt
```

## Dataset

We provide the twitter dataset in our repository, if you want get other datasets, you can find them in our paper, or you can send email to us, we are pleased to offer you other datasets.

## Usage

Here we provide the implementation of DisenIDP along with twitter dataset.

+ To train and evaluate on Twitter:
```python
python run.py -data_name=twitter
```
More running options are described in the codes, e.g., `-data_name= twitter`

## Folder Structure

DisenIDP
```
└── data: # The file includes datasets
    ├── twitter
       ├── cascades.txt       # original data
       ├── cascadetrain.txt   # training set
       ├── cascadevalid.txt   # validation set
       ├── cascadetest.txt    # testing data
       ├── edges.txt          # social network
       ├── idx2u.pickle       # idx to user_id
       ├── u2idx.pickle       # user_id to idx
       
└── models: # The file includes each part of the modules in MetaCas.
    ├── ConvBlock.py.py # The core source code of Convolution.
    ├── model.py # The core source code of DisenIPD.
    ├── TransformerBlock.py # The core source code of time-aware attention.

└── utils: # The file includes each part of basic modules (e.g., metrics, earlystopping).
    ├── EarlyStopping.py  # The core code of the early stopping operation.
    ├── Metrics.py        # The core source code of metrics.
    ├── graphConstruct.py # The core source code of building hypergraph.
    ├── parsers.py        # The core source code of parameter settings. 
└── Constants.py:    
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
└── Optim.py:          # Optimization.

```

## Cite

If you find our paper & code are useful for your research, please consider citing us 😘:

```bibtex
@inproceedings{cheng2023enhancing,
  title={Enhancing Information Diffusion Prediction with Self-Supervised Disentangled User and Cascade Representations},
  author={Cheng, Zhangtao and Ye, Wenxue and Liu, Leyuan and Tai, Wenxin and Zhou, Fan},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3808--3812},
  year={2023}
}
```

## Contact

For any questions please open an issue or drop an email to: `zhangtao.cheng at outlook.com`