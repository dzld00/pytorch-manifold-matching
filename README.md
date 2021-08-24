# Manifold Matching via Deep Metric Learning for Generative Modeling
A Pytorch implementation of "Manifold Matching via Deep Metric Learning for Generative Modeling" (ICCV 2021). 
<p align="center">
<img src="/images/noise_sphere.gif" align="middle" width="500">
</p>
Paper: https://arxiv.org/abs/2106.10777

# Objective functions
Objective for metric learning:
```
triplet_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle)
```
Objective for manifold matching with learned metric:
```
g_loss = p_dist + c_dist 
```
where 
```
ml_real_out = netML(real_img) # real data
ml_fake_out = netML(fake_img) # generated data 

# shuffle in batch
r1=torch.randperm(batch_size)
r2=torch.randperm(batch_size)
ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])

# pairwise distances 
pd_r = pairwise_distances(ml_real_out, ml_real_out) 
pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
 
# matching terms 
p_dist =  torch.dist(pd_r,pd_f,2) # matching 2-diameters             
c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2) # matching centroids  
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
We also tried our objective on generating higher resolution images using a [StyleGAN2](https://arxiv.org/abs/1912.04958) data generator and a simple metric generator. Implemenation details can be found [here](implementation-stylegan2). Below are randomly generated 512x512 samples on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset at ~150K iterations:
<p align="center">
<img src="/images/144300.png" align="middle" width="800">
</p>

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
