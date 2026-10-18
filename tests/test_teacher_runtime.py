"""Live and recordings can use different teachers: each checkpoint must load (and stay) separately."""
import torch

from stutterclear import teacher
from stutterclear import teacher_runtime as tr


class Tiny(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone_name = backbone
        self.w = torch.nn.Linear(1, 1)

    def load_state_dict(self, state, strict=True):  # nothing to load in the stand-in
        return None


def test_each_checkpoint_gets_its_own_model(monkeypatch, tmp_path):
    monkeypatch.setattr(teacher, "FrameTeacher", Tiny)
    monkeypatch.setattr(torch, "load", lambda path, **k: {"backbone": str(path), "state_dict": {}})
    monkeypatch.setattr(tr, "_model", {})
    v1, v2 = tmp_path / "v1.pt", tmp_path / "v2.pt"
    a, _ = tr.get_teacher(v1, "cpu")
    b, _ = tr.get_teacher(v2, "cpu")
    assert a is not b and a.backbone_name.endswith("v1.pt") and b.backbone_name.endswith("v2.pt")
    assert tr.get_teacher(v1, "cpu")[0] is a  # loaded once, then reused


def test_recordings_teacher_device(monkeypatch):
    monkeypatch.setenv("STUTTERCLEAR_RECORDINGS_DEVICE", "cpu")
    assert tr.recordings_device() == "cpu"
    monkeypatch.delenv("STUTTERCLEAR_RECORDINGS_DEVICE")
    monkeypatch.setattr(tr, "RECORDINGS_CKPT", tr.TEACHER_CKPT)
    assert tr.recordings_device() is None  # same checkpoint as live: share its model
