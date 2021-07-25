# ManifoldMatching
A Pytorch implementation of "Manifold Matching via Deep Metric Learning for Generative Modeling" (in ICCV 2021).

Link to original paper: https://arxiv.org/abs/2106.10777

If you find the paper or code useful, please cite as
```
@misc{dai2021adversarial,
      title={Adversarial Manifold Matching via Deep Metric Learning for Generative Modeling}, 
      author={Mengyu Dai and Haibin Hang},
      year={2021},
      eprint={2106.10777},
      archivePrefix={arXiv}
}
```

# Dependencies
- Pytorch 1.0.1

# Dataset
Download data to the data path. The sample code uses [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

# Training
To train a model for unconditonal generation, run:

```
python train.py
```
