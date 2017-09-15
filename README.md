# tacotron_pytorch

[![Build Status](https://travis-ci.org/r9y9/tacotron_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/tacotron_pytorch)

PyTorch implementation of [Tacotron](https://arxiv.org/abs/1703.10135) speech synthesis model.

Inspired from [keithito/tacotron](https://github.com/keithito/tacotron). Currently not as much good speech quality as [keithito/tacotron](https://github.com/keithito/tacotron) can generate, but it seems to be basically working. You can find some generated speech examples trained on [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) at [here](http://nbviewer.jupyter.org/github/r9y9/tacotron_pytorch/blob/master/notebooks/Test%20Tacotron.ipynb).

If you are comfortable working with TensorFlow, I'd recommend you to try
https://github.com/keithito/tacotron instead. The reason to rewrite it in PyTorch is that it's easier to debug and extend (multi-speaker architecture, etc) at least to me.

## Requirements

- PyTorch
- TensorFlow (if you want to run the training script. This definitely can be optional, but for now required.)

## Installation

```
git clone --recursive https://github.com/r9y9/tacotron_pytorch
pip install -e . # or python setup.py develop
```

If you want to run the training script, then you need to install additional dependencies.

```
pip install -e ".[train]"
```

## Training

The package relis on [keithito/tacotron](https://github.com/keithito/tacotron) for text processing, audio preprocessing and audio reconstruction (added as a submodule). Please follows the quick start section at https://github.com/keithito/tacotron and prepare your dataset accordingly.

If you have your data prepared, assuming your data is in `"~/tacotron/training"` (which is the default), then you can train your model by:

```
python train.py
```

Alignment, predicted spectrogram, target spectrogram, predicted waveform and checkpoint (model and optimizer states) are saved per 1000 global step in `checkpoints` directory. Training progress can be monitored by:

```
tensorboard --logdir=log
```

## Testing model

Open the notebook in `notebooks` directory and change `checkpoint_path` to your model.
