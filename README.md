This repository contains a PyTorch implementation for our paper [***Improved Generation of Adversarial Examples Against Safety-aligned LLMs***](https://arxiv.org/abs/2405.20778). 

## Environments
* Python 3.8.8
* PyTorch 2.2.0
* transformers 4.35.2
* tokenizers 0.15.0

## Usage
To generate adversarial suffixes, run:
```
method=${method} model=${model} seed=${seed} bash scripts/exp.sh
```
where ```method=gcg / gcg_lsgm_0.5 (gamma=0.5) / gcg_lila_16 (lila_layer=16) / gcg_combine_0.5_16_10 (gamma=0.5, lila_layer=16, num_train_queries=10) / gcgens (universal suffix) / gcgens_combine_0.5_16_10``` and ```model=llama2 / llama2-13b / mistral```.

To evaluate the adversarial suffixes, run:
```
logdir=${logdir} bash scripts/eval.sh
```

## Citation
Please cite our work in your publications if it helps your research:

```
@article{li2024improved,
  title={Improved Generation of Adversarial Examples Against Safety-aligned LLMs},
  author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
  journal={arXiv preprint arXiv:2405.20778},
  year={2024}
}
```
