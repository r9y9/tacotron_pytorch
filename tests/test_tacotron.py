# coding: utf-8
import sys
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "..", "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols
import torch
from torch.autograd import Variable
from tacotron_pytorch import Tacotron
import numpy as np


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def test_taco():
    B, T_out, D_out = 2, 400, 80
    r = 5
    T_encoder = T_out // r

    texts = ["Thank you very much.", "Hello"]
    seqs = [np.array(text_to_sequence(
        t, ["english_cleaners"]), dtype=np.int) for t in texts]
    input_lengths = np.array([len(s) for s in seqs])
    max_len = np.max(input_lengths)
    seqs = np.array([_pad(s, max_len) for s in seqs])

    x = torch.LongTensor(seqs)
    y = torch.rand(B, T_out, D_out)
    x = Variable(x)
    y = Variable(y)

    model = Tacotron(n_vocab=len(symbols), r=r)

    print("Encoder input shape: ", x.size())
    print("Decoder input shape: ", y.size())
    a, b, c = model(x, y, input_lengths=input_lengths)
    print("Mel shape:", a.size())
    print("Linear shape:", b.size())
    print("Attention shape:", c.size())

    assert c.size() == (B, T_encoder, max_len)

    # Test greddy decoding
    a, b, c = model(x, input_lengths=input_lengths)
