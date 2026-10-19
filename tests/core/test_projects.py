from pathlib import Path

import pytest

from mizuoto.core.audio.io import probe
from mizuoto.core.errors import NotFoundError, ValidationError
from mizuoto.core.outputs.models import OutputQuery
from mizuoto.core.projects.models import ExportRequest, ProjectInput, RenderRequest, ScriptImport, Speaker
from mizuoto.core.tts.models import CommonParams

SCRIPT = """# The station

The rain had stopped.

Alice: Is this the last train?
Bob: [Station master] It left an hour ago.
Alice: Then we walk.

# Morning
The sun came up over the hills. Nobody spoke.
"""


def make(services, **kw):
    kw.setdefault("name", "The Night Train")
    kw.setdefault("model", "dummy/default")
    kw.setdefault("speakers", [Speaker(id="narrator", name="Narrator", preset="alto")])
    return services.projects.create(ProjectInput(**kw))


def test_import_a_script_with_dialogue(services):
    p = make(services)
    p = services.projects.import_script(p.id, ScriptImport(text=SCRIPT))
    assert [s.name for s in p.speakers] == ["Narrator", "Alice", "Bob"]
    assert [c.title for c in p.chapters] == ["The station", "Morning"]
    lines = p.chapters[0].lines
    assert [(line.speaker, line.text) for line in lines] == [
        (None, "The rain had stopped."),
        ("alice", "Is this the last train?"),
        ("bob", "[Station master] It left an hour ago."),
        ("alice", "Then we walk."),
    ]
    assert lines[0].ends_paragraph and not lines[1].ends_paragraph and lines[3].ends_paragraph
    assert p.stats.lines == 5 and p.stats.missing == 5
    # Sentences as lines; a one-off "Note:" is not a speaker.
    q = services.projects.import_script(p.id, ScriptImport(text="Note: two sentences here. And here!", split="sentence", replace=True))
    assert [line.text for line in q.chapters[0].lines] == ["Note: two sentences here.", "And here!"]
    assert all(line.speaker is None for line in q.chapters[0].lines)
    assert services.projects.import_script(p.id, ScriptImport(text=SCRIPT, replace=True)).stats.lines == 5
    again = services.projects.script(p.id)
    assert again.startswith("# The station\n\nThe rain had stopped.\n\nAlice: Is this the last train?\nBob: [Station master]")
    assert "Then we walk.\n\n# Morning\n\nThe sun came up" in again


def test_render_mark_stale_and_export(services):
    p = make(services)
    p = services.projects.import_script(p.id, ScriptImport(text=SCRIPT))
    cast = services.projects.editable(p)
    for s in cast.speakers[1:]:
        s.preset = {"alice": "soprano", "bob": "bass"}[s.id]
    cast.speakers[2].params = CommonParams(speed=1.2)
    p = services.projects.update(p.id, cast)
    job = services.projects.render(p.id, wait=True, origin="test")
    assert job.state == "completed", job.error
    assert job.result["lines"] == 5 and job.result["failed"] == 0
    p = services.projects.get(p.id)
    assert p.stats.rendered == 5 and p.stats.seconds > 1
    take = services.outputs.get(p.chapters[0].lines[2].take)
    assert take.kind == "line" and take.params["speed"] == 1.2 and take.meta["project"] == p.id and take.format == "flac"
    # Takes stay out of History.
    assert not [o for o in services.outputs.list_outputs().items if o.kind == "line"]
    assert len(services.outputs.list_outputs(OutputQuery(kind="line")).items) == 5
    with pytest.raises(ValidationError):
        services.projects.render(p.id)

    # Editing a line's text, or its speaker's settings, makes it stale; the rest stays rendered.
    edit = services.projects.editable(p)
    edit.chapters[0].lines[0].text = "The rain had finally stopped."
    edit.speakers[1].params = CommonParams(speed=0.9)
    p = services.projects.update(p.id, edit)
    assert [line.state for line in p.chapters[0].lines] == ["stale", "stale", "rendered", "stale"]
    with pytest.raises(ValidationError):
        services.projects.export(p.id, wait=True)
    job = services.projects.render(p.id, RenderRequest(chapters=[p.chapters[0].id]), wait=True)
    assert job.result["lines"] == 3
    p = services.projects.get(p.id)
    assert p.stats.rendered == 5 and len(p.chapters[0].lines[0].takes) == 2

    # Going back to an earlier take.
    first = p.chapters[0].lines[0].takes[0].output_id
    p = services.projects.use_take(p.id, p.chapters[0].lines[0].id, first)
    assert p.chapters[0].lines[0].state == "stale"

    job = services.projects.export(p.id, ExportRequest(format="mp3", allow_missing=True), wait=True)
    assert job.state == "completed", job.error
    out = services.outputs.get(job.result["output_id"])
    assert out.kind == "project" and out.name == "The Night Train" and out.format == "mp3"
    cues = out.meta["cues"]
    assert [c["speaker"] for c in cues] == ["Narrator", "Alice", "Bob", "Alice", "Narrator"]
    assert [c["title"] for c in out.meta["chapters"]] == ["The station", "Morning"]
    takes = sum(services.outputs.get(line.take).duration for c in p.chapters for line in c.lines)
    pauses = (p.paragraph_pause_ms + 2 * p.line_pause_ms + p.chapter_pause_ms) / 1000
    assert abs(out.duration - (takes + pauses)) < 0.2
    assert abs(probe(Path(out.path)).duration - out.duration) < 0.1
    srt = services.projects.subtitles(out.id)
    assert srt.startswith("1\n00:00:00,000 --> ") and "Is this the last train?" in srt
    assert services.projects.subtitles(out.id, "vtt").startswith("WEBVTT\n\n00:00:00.000 --> ")

    # One chapter alone.
    job = services.projects.export(p.id, ExportRequest(chapters=[p.chapters[1].id]), wait=True)
    assert services.outputs.get(job.result["output_id"]).name == "The Night Train — Morning"


def test_failures_are_kept_per_line(services):
    p = make(services, speakers=[Speaker(id="narrator", name="Narrator", preset="alto"), Speaker(id="ghost", name="Ghost", preset="nobody")])
    p = services.projects.import_script(p.id, ScriptImport(text="Hello.\nGhost: Boo.\nGhost: Boo again."))
    job = services.projects.render(p.id, wait=True)
    assert job.state == "completed" and job.result["failed"] == 2
    p = services.projects.get(p.id)
    assert [line.state for line in p.chapters[0].lines] == ["rendered", "failed", "failed"]
    assert "nobody" in (p.chapters[0].lines[1].error or "")


def test_a_speakers_settings_outlive_a_change_of_model(services):
    # A speaker set up for one model keeps its settings under another; each line's request carries
    # only what its model takes, so a change of model does not fail every line.
    speaker = Speaker(id="narrator", name="Narrator", instruct="Slowly", params=CommonParams(speed=1.2, temperature=0.5), engine_options={"delay_ms": 5})
    p = services.projects.import_script(make(services, speakers=[speaker]).id, ScriptImport(text="Hello."))
    line = p.chapters[0].lines[0].id
    req = services.projects.request_for(p.id, line)
    assert req.style.instruct == "Slowly" and req.params.speed == 1.2 and req.engine_options == {"delay_ms": 5}

    cast = services.projects.editable(p)
    cast.model = "qwen3tts/1.7b-base"
    p = services.projects.update(p.id, cast)
    assert p.speakers[0].instruct == "Slowly" and p.speakers[0].engine_options == {"delay_ms": 5} and p.instruct_unused == ["narrator"]
    req = services.projects.request_for(p.id, line)
    assert req.style is None and req.engine_options == {} and req.params.model_dump(exclude_none=True) == {"temperature": 0.5}

    # An instruction this model follows only with a preset speaker: not with its own voice.
    cast.model = "cosyvoice/cosyvoice-300m-instruct"
    assert services.projects.request_for(services.projects.update(p.id, cast).id, line).style is None
    cast.speakers[0].preset = "中文女"
    assert services.projects.request_for(services.projects.update(p.id, cast).id, line).style.instruct == "Slowly"

    # Back to the first model: everything again. A model Mizuoto does not know gets the settings as saved.
    cast.model, cast.speakers[0].preset = "dummy/default", None
    p = services.projects.update(p.id, cast)
    req = services.projects.request_for(p.id, line)
    assert req.style.instruct == "Slowly" and req.engine_options == {"delay_ms": 5} and req.params.speed == 1.2 and p.instruct_unused == []
    assert "instruct_unused" not in (services.settings.projects_dir / p.slug / "project.json").read_text()
    cast.model = "nope/nothing"
    assert services.projects.request_for(services.projects.update(p.id, cast).id, line).engine_options == {"delay_ms": 5}


def test_checks_and_delete(services):
    p = make(services)
    edit = services.projects.editable(services.projects.import_script(p.id, ScriptImport(text="One.\nTwo.")))
    edit.chapters[0].lines[1].speaker = "nobody"
    with pytest.raises(ValidationError):
        services.projects.update(p.id, edit)
    edit.chapters[0].lines[1].speaker = None
    edit.speakers.append(Speaker(id="narrator", name="Again"))
    with pytest.raises(ValidationError):
        services.projects.update(p.id, edit)
    no_model = services.projects.create(ProjectInput(name="No model"))
    no_model = services.projects.import_script(no_model.id, ScriptImport(text="Hello."))
    assert services.projects.render(no_model.id, wait=True).result["failed"] == 1
    assert "no model" in (services.projects.get(no_model.id).chapters[0].lines[0].error or "")

    services.projects.render(p.id, wait=True)
    takes = [line.take for line in services.projects.get(p.id).chapters[0].lines]
    # A removed line's takes go with it.
    edit = services.projects.editable(services.projects.get(p.id))
    edit.chapters[0].lines.pop()
    services.projects.update(p.id, edit)
    with pytest.raises(NotFoundError):
        services.outputs.get(takes[1])
    services.projects.delete(p.slug)
    with pytest.raises(NotFoundError):
        services.outputs.get(takes[0])
    assert [s.name for s in services.projects.list_projects()] == ["No model"]


def test_render_on_the_server_batches_lines(make_services):
    services = make_services(start_background=True)
    p = make(services)
    p = services.projects.import_script(p.id, ScriptImport(text="\n".join(f"Line number {i}." for i in range(6))))
    job = services.projects.render(p.id)
    done = services.jobs.wait(job.id, 120)
    assert done.state == "completed", done.error
    p = services.projects.get(p.id)
    assert p.stats.rendered == 6
    speak_jobs = [services.jobs.get(services.outputs.get(line.take).job_id) for line in p.chapters[0].lines]
    assert all(j.origin == "project" for j in speak_jobs)
    assert any("batch" in (j.result or {}) for j in speak_jobs)
