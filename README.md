# RDGAN (ICME 2019)
ICME 2019 paper "RDGAN: Retinex Decomposition based Adversarial Learning for Low-Light Enhancement" [1]

We will release our code soon
### Abstract
Pictures taken under the low-light condition often suffer from low contrast and loss of image details, thus an approach that can effectively improve low-light images is demanded. Traditional Retinex-based methods assume that the reflectance components of low-light images keep unchanged, which neglect the color distortion and lost details. In this paper, we propose an end-to-end learning-based framework that first decomposes the low-light image and then learns to fuse the decomposed results to obtain the high quality enhanced result. Our framework can be divided into a RDNet (Retinex Decomposition Network) for decomposition and a FENet (Fusion Enhancement Network) for fusion. Specific multi-term losses are respectively designed for the two networks. We also present a new RDGAN (Retinex Decomposition based Generative Adversarial Network) loss, which is computed on the decomposed reflectance components of the enhanced and the reference images. Experiments demonstrate that our approach is good at color and detail restoration, which outperforms other state-of-the-art methods.
### Citation
```
[1]  @inproceedings{RDGAN,
         author = {Junyi Wang and Weimin Tan and Xuejing Niu and Bo Yan},
         title = {RDGAN: Retinex Decomposition based Adversarial Learning for Low-Light Enhancement},
         booktitle = {ICME},
         year = {2019}
     }
```
