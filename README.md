# Spatio-Temporal Adversarial Video Super Resolution (VSR)

#### Developers - Ankit Chadha (ankitrc@stanford.edu), Mohamed Masoud (masoudm@uw.edu)
--------
This repository is based off RBPN's [PyTorch RBPN implementation](https://github.com/alterzero/RBPN-PyTorch)

This was done as part of CS231n: Convolutional Neural Networks for Visual Recognition - Stanford / Spring 2019 class project.

<img src="https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution/blob/master/images/vsr1.png?raw=true" width="800" height="600">
--------


<img src="https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution/blob/master/images/spatio_postr.png">
<img src="https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution/blob/master/images/time.png">
Spatial Time series output showcasing the effectiveness of the Discriminator

--------

### Abstract
--------

Video super-resolution (VSR) aims to infer a high- resolution (HR) video sequence from multiple low- resolution (LR) frames. VSR helps with many important applications such as new scene generation, anomaly detec- tion and recovery of information lost during video compres- sion. For single-image super-resolution (SISR), adversarial training has been highly successful, yielding realistic and highly detailed results. The Recurrent Back-Projection Net- work (RBPN) uses a recurrent framework that treats each frame as a separate spatial source of information [1]. Ad- ditionally, RBPN combines the frames’ spatial information in a recurrent iterative refinement framework inspired by the idea of back-projection to produce temporally coherent multiple-image super-resolution (MISR). The recurrent gen- erative framework integrates spatial and temporal contexts from continuous video frames using a recurrent encoder- decoder module that fuses multi-frame information with the SR version of the target frame [1]. In this project, we pro- pose a novel architecture for a spatio-temporal adversarial recurrent discriminator to achieve photorealistic and tem- porally coherent super resolved frames with a more sophis- ticated objective function to fine-tune spatial and tempo- ral features of RBPN. Our quantitative and visual analyses highlight the enhanced capability of the RBPN generator to be able to learn high frequency details both spatially and temporally.

### Neural Architecture
--------
Here is an overview of our network architecture 

<img src="https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution/blob/master/images/overall.png" width="200" height="200">

<img src="https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution/blob/master/images/disc.png" width="200" height="200">


### Dataset (TOFLOW)
--------
The dataset used in this project is vimeo90k by TOFlow [13]. We are using the septuplet dataset which has 91,701 7- frame sequences which cover a variety of motion and detail in the videos. We augmented the data by flipping, rotat- ing, and randomly cropping the input frames to extend the distribution and variance of the dataset and reduce overfit- ting. Each augmentation was applied to the whole septuplet to avoid discrepancies between frames. The choice of this dataset was inspired by RBPN [1], which we choose as the model generator. The proposed model uses as an input 7 LR frames including the target LR frame for SISR. The net- work is trained on patches rather than full frames. These patches are 64x64 random crops from the original video frames. The pre-trained network does not perform any ex- plicit normalization to the data. The data was arranged into 70-15-15 split for train, validation and test sets. Addition- ally, following most of the video super resolution literature comparisons, we conducted further test experiments over the Vid4 dataset. That allows to further test the general- izability of our model on other datasets and to have a fair comparison with SoTA methods. You can download the dataset [here](http://toflow.csail.mit.edu/).


### Command Lines
--------
This repository has command line bash files with the optimal hyperparameters our network was tuned for. 
```
1. Sanity Check 
#Launch a debug run on 1 example out of the SQuAD 2.0 training set - Beyonce paragraph 
examples/rundbg.sh

2. Train on SQuAD 2.Q
#Fine tunes BERT layers on SQuAD 2.Q and trains additional directed co-attention layers.
run_bertqa_expt.sh

3. Train on SQuAD 2.0
#Fine tunes BERT embedding layers on SQuAD 2.0 and trains additional directed co-attention layers.
examples/run_bertqa.sh
```

### BibTeX
--------
```
@misc{Stanford-CS231n,
  author = {Chadha,Ankit;Masoud,M},
  title = {Spatio-Temporal Adversarial Video Super Resolution},
  year = {2019},
  publisher = {Stanford-CS231n},
  howpublished = {\url{https://github.com/masoudML/Spatio_Temporal_Adversarial_Video_Super_Resolution}}
}
```

### References
--------
Refer to the paper for more details on our hyperparameters chosen.

[1] M. Haris, G. Shakhnarovich, N. Ukita. Recurrent Back-Projection Network for Video Super-Resolution, Accepted: Conference on Computer Vision and Pat- tern Recognition, (2019).

[2] M. Haris, G. Shakhnarovich, N. Ukita. Deep Back- Projection Networks For Super-Resolution , Confer- ence on Computer Vision and Pattern Recognition, (2018).

[3] M. Chu, Y. Xie, L. L. Taix, N. Thuerey, tem- poGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow, textitACM Transac- tion on Graphics, (2018).

[4] Y. Xie, E. Franz, M. Chu, N. Thuerey, tempoGAN: A Temporally Coherent, Volumetric GAN for Super- resolution Fluid Flow, textitACM Transaction on Graphics, (2018).
