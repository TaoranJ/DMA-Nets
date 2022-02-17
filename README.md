# DMA-Nets

The official `Pytorch` implementation of `DMA-Nets` proposed in paper "Dynamic Multi-Context Attention Networks for Citation Forecasting of Scientific Publications", AAAI 2021.

## Installation and Usage

Install `torch`, `numpy`, `python-dateutil`, and `tqdm` first. Run `bash dma-nets.sh` for a quickstart. See the first couple of lines of `dma_nets.sh` for examples.

```bash
bash dma_nets.sh [gpu to use] [dataset to use]
```

## Paper Abstract

**Title**: Dynamic Multi-Context Attention Networks for Citation Forecasting of Scientific Publications

**Abstract**: Forecasting citations of scientific patents and publications is a crucial task for understanding the evolution and development of technological domains and for foresight into emerging technologies. By construing citations as a time series, the task can be cast into the domain of temporal point processes. Most existing work on forecasting with temporal point processes, both conventional and neural network-based, only performs single-step forecasting. In citation forecasting, however, the more salient goal is n-step forecasting: predicting the arrival time and the technology class of the next n citations. In this paper, we propose Dynamic Multi-Context Attention Networks (DMA-Nets), a novel deep learning sequence-to-sequence (Seq2Seq) model with a novel hierarchical dynamic attention mechanism for long-term citation forecasting. Extensive experiments on two real-world datasets demonstrate that the proposed model learns better representations of conditional dependencies over historical sequences compared to state-of-the-art counterparts and thus achieves significant performance for citation predictions. The dataset and code have been made available online.

## Citing

If you used data or codes in this repo in your research, please cite "[Dynamic Multi-Context Attention Networks for Citation Forecasting of Scientific Publications](https://ojs.aaai.org/index.php/AAAI/article/view/16970)".

```bib
@article{Ji_Self_Fu_Chen_Ramakrishnan_Lu_2021, 
  title={Dynamic Multi-Context Attention Networks for Citation Forecasting of Scientific Publications}, 
  volume={35}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/16970}, 
  number={9}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Ji, Taoran and Self, Nathan and Fu, Kaiqun and Chen, Zhiqian and Ramakrishnan, Naren and Lu, Chang-Tien}, 
  year={2021}, 
  month={May}, 
  pages={7953-7960} 
}
```

## Copyright and License

Codes in this repo released under MIT license.

## Collaboration

Feel free to contact me (see my profile for email address) if you're interested in related research topics or want to use patent-related data.
