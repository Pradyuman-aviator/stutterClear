"""Demo render cache: samples stay rendered; a failed Nemotron check is retried later."""
from stutterclear.app import demo


def test_samples_stay_cached_and_a_failed_guard_is_retried(monkeypatch):
    calls = []

    def fake(sid, engine):
        calls.append(sid)
        return {"timeline": {"guard": {"error": "timeout"} if sid == "down" else None}}

    monkeypatch.setattr(demo, "_render", fake)
    monkeypatch.setattr(demo, "_samples", {})
    demo.render("ok", "natural")
    demo.render("ok", "natural")
    assert calls == ["ok"]  # rendered once, then kept

    clock = [1000.0]
    monkeypatch.setattr(demo.time, "time", lambda: clock[0])
    demo.render("down", "natural")
    demo.render("down", "natural")
    assert calls.count("down") == 1  # not hammering Nemotron on every click
    clock[0] += demo.GUARD_RETRY_S + 1
    demo.render("down", "natural")
    assert calls.count("down") == 2  # asked again later
