# Structure of repository

This repository is split into 3 parts, according to the 3 parts of my FYP
project:

1. Black-box hard-label attack on multi-class image classification models via
   Bayesian Optimization -> [bo_attack](bo_attack) folder.
2. JPEG Compression Attack on ManTra-Net -> [mantra_net](mantra_net) folder.
3. Defense against JPEG attacks via estimating previous quantization matrices ->
   [estimate_jpeg_qtables](estimate_jpeg_qtables) folder.

## Datasets used
- [RAISE](http://loki.disi.unitn.it/RAISE/download.html) - dataset of
  high-resolution, raw, unprocessed, and uncompressed images.
- [CASIA2.0](https://github.com/namtpham/casia2groundtruth) &
  [CASIA1.0](https://github.com/namtpham/casia1groundtruth) - forgery detection
  datasets.
