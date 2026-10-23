"""FrameTeacher from several threads at once: each call must see its own last layer's output."""
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")
from stutterclear import teacher  # noqa: E402


class LastLayer(nn.Module):
    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier

    def forward(self, x):
        self.barrier.wait(timeout=10)  # both calls are inside the last layer at the same time
        return (x * 2,)


class FakeEncoder(nn.Module):
    """Stands in for WavLM: hidden states (input, normed last), and the raw last layer via the hook."""

    def __init__(self, barrier):
        super().__init__()
        self.encoder = SimpleNamespace(layers=nn.ModuleList([LastLayer(barrier)]))
        self.layers = self.encoder.layers  # registered as a submodule

    def forward(self, wave, output_hidden_states=True):
        h0 = wave[..., None].repeat(1, 1, 4)
        out = self.encoder.layers[-1](h0)[0]
        return SimpleNamespace(hidden_states=(h0, out * 0 - 1))  # a post-norm stand-in, never used


def tiny_teacher(barrier):
    m = teacher.FrameTeacher.__new__(teacher.FrameTeacher)
    nn.Module.__init__(m)
    m.normalize = False
    m.ssl = FakeEncoder(barrier)
    m.layer_weights = nn.Parameter(torch.tensor([-30.0, 30.0]))  # all weight on the last layer
    m.proj = nn.Identity()
    m.rnn = lambda x: (x, None)
    m.out = nn.Identity()
    m.ssl.encoder.layers[-1].register_forward_hook(teacher._keep_last)
    return m


def test_concurrent_calls_each_see_their_own_last_layer():
    barrier = threading.Barrier(2)
    m = tiny_teacher(barrier)
    waves = {"long": torch.ones(1, 210), "short": torch.full((1, 49), 3.0)}
    out, errors = {}, []

    def run(name):
        try:
            with torch.no_grad():
                out[name] = m(waves[name])
        except Exception as e:  # a shape mismatch here was the bug
            errors.append(e)

    threads = [threading.Thread(target=run, args=(k,)) for k in waves]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors, errors
    # gelu of the raw last layer (2 x the input), not the other call's, not the post-norm stand-in
    assert out["long"].shape == (1, 210, 4) and out["short"].shape == (1, 49, 4)
    assert torch.allclose(out["long"], torch.nn.functional.gelu(torch.tensor(2.0)).expand(1, 210, 4), atol=1e-4)
    assert torch.allclose(out["short"], torch.nn.functional.gelu(torch.tensor(6.0)).expand(1, 49, 4), atol=1e-4)
