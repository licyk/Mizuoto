"""Browser end-to-end tests (``-m e2e``; ``scripts/dev.py test-e2e``): the built web UI (``mizuoto/webui/dist``) served by
a real server on a new data folder, driven in headless Chromium by Playwright. The dummy engine needs no download, so
a whole speak runs. Skipped without Playwright, its Chromium or a built UI."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

DIST = Path(__file__).resolve().parents[2] / "mizuoto" / "webui" / "dist" / "index.html"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    if not DIST.is_file():
        pytest.skip("the web UI is not built (python scripts/dev.py web)")
    port = _free_port()
    data = tmp_path_factory.mktemp("data")
    log = open(data.parent / "server.log", "w")  # noqa: SIM115
    env = {**os.environ, "MIZUOTO_DATA_DIR": str(data), "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "mizuoto", "webui", "--port", str(port), "--no-open"], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"the server stopped: {(data.parent / 'server.log').read_text()[-2000:]}")
            try:
                urllib.request.urlopen(f"{url}/api/v1/app/version", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("the server did not start in 60 s")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


@pytest.fixture(scope="session")
def browser() -> Generator[Any, None, None]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # the package without its browser
            pytest.skip(f"Playwright's Chromium is missing (python -m playwright install chromium): {str(e).splitlines()[0]}")
        yield b
        b.close()


class Page:
    """A page of the UI with the console's errors and the page's uncaught exceptions collected."""

    def __init__(self, browser: Any, url: str, width: int = 1280, height: int = 900, locale: str = "en-US") -> None:
        self.url = url
        self.context = browser.new_context(viewport={"width": width, "height": height}, locale=locale)
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.page.on("console", lambda m: self.errors.append(f"console: {m.text}") if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"page: {e}"))

    def open(self, route: str) -> Any:
        self.page.goto(f"{self.url}/#{route}")
        self.page.wait_for_load_state("networkidle")
        return self.page

    def close(self) -> None:
        self.context.close()


@pytest.fixture
def ui(browser: Any, server: str) -> Generator[Any, None, None]:
    pages: list[Page] = []

    def make(**kw: Any) -> Page:
        page = Page(browser, server, **kw)
        pages.append(page)
        return page

    yield make
    for page in pages:
        page.close()
