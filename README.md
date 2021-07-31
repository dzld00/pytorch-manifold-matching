# Manifold Matching
A Pytorch implementation of "Manifold Matching via Deep Metric Learning for Generative Modeling" (in ICCV 2021).
![](/images/pipeline.png)

Link to original paper: https://arxiv.org/abs/2106.10777

# Dependencies
- Pytorch 1.0.1

# Dataset
Download data to the data path. The sample code uses [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

# Training
To train a model for unconditonal generation, run:

```
python train.py
```

# Citation
```
@misc{daiandhang2021manifold,
      title={Manifold Matching via Deep Metric Learning for Generative Modeling}, 
      author={Mengyu Dai and Haibin Hang},
      year={2021},
      eprint={2106.10777},
      archivePrefix={arXiv}
}
```
