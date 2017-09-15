import torch
from torch.autograd import Variable
from torch import nn

from tacotron_pytorch.attention import BahdanauAttention, AttentionWrapper
from tacotron_pytorch.attention import get_mask_from_lengths


def test_attention_wrapper():
    B = 2

    encoder_outputs = Variable(torch.rand(B, 100, 256))
    memory_lengths = [100, 50]

    mask = get_mask_from_lengths(encoder_outputs, memory_lengths)
    print("Mask size:", mask.size())

    memory_layer = nn.Linear(256, 256)
    query = Variable(torch.rand(B, 128))

    attention_mechanism = BahdanauAttention(256)

    # Attention context + input
    rnn = nn.GRUCell(256 + 128, 256)

    attention_rnn = AttentionWrapper(rnn, attention_mechanism)
    initial_attention = Variable(torch.zeros(B, 256))
    cell_state = Variable(torch.zeros(B, 256))

    processed_memory = memory_layer(encoder_outputs)

    cell_output, attention, alignment = attention_rnn(
        query, initial_attention, cell_state, encoder_outputs,
        processed_memory=processed_memory,
        mask=None, memory_lengths=memory_lengths)

    print("Cell output size:", cell_output.size())
    print("Attention output size:", attention.size())
    print("Alignment size:", alignment.size())

    assert (alignment.sum(-1) == 1).data.all()


test_attention_wrapper()
