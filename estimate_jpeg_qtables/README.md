# Defense against JPEG attacks via estimating previous quantization matrices

This repository implements a forgery detection method against JPEG compression
attacks. It also contains the unofficial implementation of the previous
quantization matrix estimation (PQE) algorithm in
[Estimating Previous Quantization Factors on Multiple JPEG Compressed
Images](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-021-00120-7).

## Guide to notebooks:

- [results_analysis_CASIA1.0.ipynb](results_analysis_CASIA1.0.ipynb) - notebook
  analyzing results of all experiments done for this part of the project.

### [pqe](pqe) folder
This folder contains all experiments related to the previous quantization
estimation algorithm:
- [estimate_qtables.ipynb](pqe/estimate_qtables.ipynb) - unofficial
  implementation of PQE algorithm described by [Estimating Previous Quantization
  Factors on Multiple JPEG Compressed
  Images](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-021-00120-7).
- [estimate_qtables_via_torchjpeg.ipynb](pqe/estimate_qtables_via_torchjpeg.ipynb)
  - a faster alternative implementation utilizing the `torchjpeg` package.
- [estimate_qtables_raise1k.ipynb](pqe/estimate_qtables_raise1k.ipynb) -
  experiment on effectiveness of implementation on the
  [RAISE](http://loki.disi.unitn.it/RAISE/download.html) dataset.
- [estimate_qtables_raise1k_vary_block_size.ipynb](pqe/estimate_qtables_raise1k_vary_block_size.ipynb)
  - same experiment as above but with varying block sizes.

### [forgery_dectection](forgery_detection) folder
This folder contains notebooks on forgery detection using the PQE algorithm.
- [detect_forgery.ipynb](forgery_detection/detect_forgery.ipynb) -
  implementation of forgery detection algorithm.
- [detect_forgery_CASIA1.0.ipynb](forgery_detection/detect_forgery_CASIA1.0.ipynb)
  - forgery detection experiment on CASIA1.0 dataset.
