"""The web UI in a real browser: every screen loads without errors, a speak runs from the text to the result, and a
phone-sized window neither scrolls sideways nor loses a screen."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

SCREENS = ["/speak", "/projects", "/tools", "/voices", "/train", "/models", "/history", "/settings"]


def test_every_screen_loads_without_errors(ui):
    page = ui()
    for route in SCREENS:
        p = page.open(route)
        assert p.locator("main").inner_text().strip(), route
    assert page.errors == []


def test_a_speak_runs_from_the_text_to_the_result(ui):
    page = ui()
    p = page.open("/speak")
    p.get_by_label("Text").first.fill("A browser test speaks this line.")
    p.get_by_role("button", name="Speak", exact=True).click()
    row = p.locator("section.results li.row").first
    row.wait_for(timeout=60_000)
    assert row.locator(".name").inner_text() == "A browser test speaks this line."
    p = page.open("/history")
    p.get_by_text("A browser test speaks this line.").first.wait_for(timeout=10_000)
    assert page.errors == []


@pytest.mark.parametrize("width", [320, 390])
def test_a_phone_window_fits_and_reaches_every_screen(ui, width):
    page = ui(width=width, height=800)
    p = page.open("/speak")
    for route in SCREENS:
        p = page.open(route)
        assert p.evaluate("document.documentElement.scrollWidth") <= width, route
    p = page.open("/speak")
    labels = p.locator("nav.bar .label").all_inner_texts()
    assert labels == ["Speak", "Projects", "Tools", "Voices", "More"]
    p.get_by_role("button", name="More").click()
    p.get_by_role("menuitem", name="History").click()
    p.wait_for_url("**/#/history")
    assert "active" in (p.locator("nav.bar button.more").get_attribute("class") or "")
    assert page.errors == []


def test_the_chinese_ui(ui):
    page = ui(locale="zh-CN")
    p = page.open("/speak")
    assert p.locator("nav.rail").inner_text().count("合成") == 1
    assert page.errors == []
