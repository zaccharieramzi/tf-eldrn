# TensorFlow implementation of the ELDRN
[![Build Status](https://travis-ci.com/zaccharieramzi/tf-eldrn.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tf-eldrn)

The MWCNN is a network introduced by Aditi Panda et al. in "Exponential linear unit dilated residual network for digital image denoising", Journal of Electronic Imaging 2018. If you use this network, please cite their work appropriately.

I could not find the official implementation for this work.

The goal of this implementation in TensorFlow is to be easy to read and to adapt:

- all the code is in one file
- defaults are those from the paper (for gray image denoising)
- there is no other imports than from TensorFlow
