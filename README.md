# Traveling Words

This is the official repository for the paper "Traveling Words: A Geometric Interpretation of Transformers".

arXiv: https://arxiv.org/pdf/2309.07315.pdf

<p align="center">
<img width="512" alt="image" src="https://github.com/santiag0m/traveling-words/assets/32421689/17e923c9-ff97-4d40-a6eb-fc7a7112b338">
</p>

## Installation

To clone the repositorty and its dependencies run:

```bash
git clone --recurse-submodules https://github.com/santiag0m/traveling-words.git
```

To install requirements (Python 3.10):

```bash
pip install -r requirements.txt
```

# Experiments

## Measuring the impact of layer normalization parameters on word embeddings

```bash
python word_embeddings.py
```

This scripts replicates the paper results on the top-k (`k=5`) tokens from the word embedding matrix according to different scoring schemes:

* Norm: The original embedding vector norm
* Scaled Norm: The vector norm after multiplying by the layer norm scaling parameter `gamma`
* Bias: Original norm + bias score
* Scaled and Bias: Scaled norm + bias score


## Probing query-key and key-value interactions from attention heads


```bash
python common_nouns.py
```

This script loads the 100 most common nouns in the english language and probes the attention heads of transformer blocks 0, 5 and 11 of the 124M parameter version of GPT-2, according to the geometric interpretation of attention as acting upon a high-dimensional ellipsoid containing keys and queries.

To test this with a different model (consistent with the nanoGPT implementation) or on a different block:

```python
from utils.model_utils import load_model
from common_nouns import attention_ellipse

model, encode, decode = load_model(init_from="gpt2")
attention_ellipse(
  model=model, encode=encode, decode=decode, block_idx=0, print_top_k_nearest=5
)
```

Results will be saved in two separate log files: `common_noun_logs_qk.txt` and `common_noun_logs_vo.txt`

## Probing left and right singular vectors from the virtual matrices `W_qk` and `W_vo`

```bash
python svd_analysis.py
````

This script computes the Singular Value Decomposition (SVD) of the `W_qk` and `W_vo` matrices for attention heads at blocks 0, 5 and 11 of GPT-2, and probes their singular vectors according the geometric interpretation discussed on the paper.

To test with different models or on a different block:

```python
from utils.model_utils import load_model
from svd_analysis import attention_svd

model, encode, decode = load_model(init_from="gpt2")
attention_svd(model=model, block_idx=0, top_k_nearest=5, plot=False)
```

Results will be saved in two separate log files `svd_qk_logs.txt` and `svd_vo_logs.txt`

## Visualize the trajectory of latent states throughout the transformer

```bash
python trajectory.py
```

This script plots the 3D UMAP vectors of the latent states of a given phrase (e.g. "To kill two birds with one stone") as they change throughout transformer blocks.


<p align="center">
<img width="512" alt="image" src="https://github.com/santiag0m/traveling-words/assets/32421689/0e355484-2b9d-4bfb-ad39-53640e157c87">
</p>
