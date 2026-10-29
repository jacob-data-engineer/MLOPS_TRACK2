"""APP_BACKGROUND=off: a container that only shows pages (2026-10-28).

The refactor slot on 8092 runs code still being built against production's
tables, bucket and queue. Its startup recovery would fail production's
in-flight inference/eval rows, and its poll loop would start queued runs
and chain retrains. With the switch off it does neither.
"""

import asyncio
import pathlib

from ppe_pipeline_api import main


def test_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv("APP_BACKGROUND", raising=False)
    assert main.background_enabled() is True
    monkeypatch.setenv("APP_BACKGROUND", "on")
    assert main.background_enabled() is True
    monkeypatch.setenv("APP_BACKGROUND", " OFF ")
    assert main.background_enabled() is False


def test_off_skips_recovery_and_the_poll_loop(monkeypatch):
    calls = []
    monkeypatch.setenv("APP_BACKGROUND", "off")
    monkeypatch.setattr(main.track2_shared, "recover_interrupted_jobs",
                        lambda: calls.append("recover"))

    async def loop():
        calls.append("poll")

    async def idle():
        return None

    monkeypatch.setattr(main.ticker, "queue_poll_loop", loop)
    monkeypatch.setattr(main.track2_shared, "warm_registry_cache", lambda: None)
    monkeypatch.setattr(main.track2_shared, "registry_warm_loop", idle)

    async def run():
        async with main.lifespan(main.app):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert calls == []


def test_on_still_recovers_and_polls(monkeypatch):
    calls = []
    monkeypatch.delenv("APP_BACKGROUND", raising=False)
    monkeypatch.setattr(main.track2_shared, "recover_interrupted_jobs",
                        lambda: calls.append("recover"))

    async def loop():
        calls.append("poll")

    async def idle():
        return None

    monkeypatch.setattr(main.ticker, "queue_poll_loop", loop)
    monkeypatch.setattr(main.track2_shared, "warm_registry_cache", lambda: None)
    monkeypatch.setattr(main.track2_shared, "registry_warm_loop", idle)

    async def run():
        async with main.lifespan(main.app):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert calls == ["recover", "poll"]


def test_the_deploy_passes_it_to_the_container():
    script = (pathlib.Path(__file__).resolve().parents[1] / "deploy.sh").read_text(encoding="utf-8")
    assert 'APP_BACKGROUND="${APP_BACKGROUND:-on}"' in script
    assert "-e APP_BACKGROUND='$APP_BACKGROUND'" in script


def test_a_look_only_deploy_never_ships_the_shared_lambdas():
    """The router and queue runner serve production; the refactor slot's
    unpromoted code must not reach them."""
    script = (pathlib.Path(__file__).resolve().parents[1] / "deploy.sh").read_text(encoding="utf-8")
    off = script.index('if [ "$APP_BACKGROUND" = "off" ]; then')
    apply = script.index("terraform apply -auto-approve")
    skip = script.index('elif [ "${SKIP_LAMBDA:-}" = "1" ]; then')
    assert off < skip < apply
