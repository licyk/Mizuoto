"""Recorded file paths follow the data folder: moved, renamed, copied, reached through a link, or written by a version
that kept absolute paths."""

import shutil
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

import pytest

from mizuoto.core import codes
from mizuoto.core.context import build_services
from mizuoto.core.errors import NotFoundError
from mizuoto.core.files import from_stored, is_absolute_text, relocations, to_stored
from mizuoto.core.storage import PathCheckRequest
from mizuoto.core.tts.models import CommonParams, SpeakRequest


def test_path_text_helpers(tmp_path):
    base = tmp_path / "data"
    (base / "outputs").mkdir(parents=True)
    assert to_stored(base / "outputs" / "a.wav", base) == "outputs/a.wav"
    assert to_stored(tmp_path / "elsewhere.wav", base) == str(tmp_path / "elsewhere.wav")
    link = tmp_path / "link"
    link.symlink_to(base, target_is_directory=True)
    assert to_stored(link / "outputs" / "a.wav", base) == "outputs/a.wav"
    assert from_stored("outputs/a.wav", base) == base / "outputs" / "a.wav"
    assert from_stored(str(tmp_path / "x.wav"), base) == tmp_path / "x.wav"
    for absolute in ("/x/y.wav", "C:\\Users\\me\\mizuoto\\outputs\\a.wav", "D:/m/a.wav", "\\\\server\\share\\a.wav"):
        assert is_absolute_text(absolute)
    assert not is_absolute_text("outputs/2026-10/a.wav")
    # A Windows record opened on Linux still yields candidates, longest suffix first, never with ".."
    found = relocations("C:\\Users\\me\\mizuoto\\outputs\\2026-10\\a.wav", [base])
    assert found[0] == base / "Users" / "me" / "mizuoto" / "outputs" / "2026-10" / "a.wav"
    assert base / "outputs" / "2026-10" / "a.wav" in found and found[-1] == base / "2026-10" / "a.wav"
    assert all(".." not in PurePosixPath(p.as_posix()).parts for p in found)


def _populate(data: Path, wav_file: Path):
    s = build_services(data_dir=data, environ={})
    job = s.tts.speak(SpeakRequest(model="dummy/default", text="Hello there.", params=CommonParams(seed=1)), wait=True)
    assert job.state == "completed" and job.result, job.error
    output = s.outputs.get(job.result["output_id"])
    upload = s.audio.upload("ref.wav", [wav_file.read_bytes()])
    rows = {r["id"]: r["path"] for r in s.db.fetchall("SELECT id, path FROM outputs UNION ALL SELECT id, path FROM audio_files")}
    s.close()
    return output, upload, rows


def _open(data: Path):
    return build_services(data_dir=data, environ={})


@pytest.mark.parametrize("how", ["moved", "renamed", "linked"])
def test_outputs_and_uploads_follow_the_data_folder(how, tmp_path, wav_file):
    first = tmp_path / "a" / "mizuoto"
    output, upload, rows = _populate(first, wav_file)
    assert all(not is_absolute_text(p) for p in rows.values()), rows  # recorded relative
    if how == "moved":
        data = tmp_path / "b" / "mizuoto"
        data.parent.mkdir()
        shutil.move(str(first), str(data))
    elif how == "renamed":
        data = first.with_name("mizuoto-renamed")
        first.rename(data)
    else:
        data = tmp_path / "link"
        data.symlink_to(first, target_is_directory=True)
    s = _open(data)
    try:
        path = s.outputs.path_of(output.id)
        assert path.is_file() and path.is_relative_to(s.settings.data_dir)
        assert s.outputs.get(output.id).missing is False
        assert s.audio.path_of(upload.id).is_file()
        assert s.audio.peaks(path).points > 0
        with zipfile.ZipFile(BytesIO(b"".join(s.outputs.zip([output.id])))) as z:
            assert len(z.namelist()) == 1
        again = s.audio.register_path(s.audio.path_of(upload.id))
        assert again.id == upload.id  # the same file is not recorded twice
        assert s.outputs.delete([output.id]) == [output.id]
        assert not path.exists()
    finally:
        s.close()


def test_a_copied_data_folder_reads_and_deletes_only_its_own_files(tmp_path, wav_file):
    original = tmp_path / "original"
    output, _, _ = _populate(original, wav_file)
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    s = _open(copy)
    try:
        path = s.outputs.path_of(output.id)
        assert path.is_relative_to(copy)
        s.outputs.delete([output.id])
    finally:
        s.close()
    assert not path.exists()
    assert list((original / "outputs").rglob("*.wav")), "the original's file must stay"


def _make_legacy(data: Path) -> None:
    """Rewrite the records the way versions before relative paths kept them: absolute, and no ``meta``."""
    s = _open(data)
    for table in ("outputs", "audio_files"):
        for row in s.db.fetchall(f"SELECT id, path FROM {table}"):
            s.db.execute(f"UPDATE {table} SET path = ? WHERE id = ?", (str(s.storage.resolve(row["path"])), row["id"]))
    s.db.execute("DELETE FROM meta")
    s.close()


def test_a_legacy_database_is_rewritten_when_opened_after_a_move(tmp_path, wav_file):
    first = tmp_path / "a" / "mizuoto"
    output, upload, _ = _populate(first, wav_file)
    _make_legacy(first)
    elsewhere = tmp_path / "server.wav"
    shutil.copyfile(wav_file, elsewhere)
    s = _open(first)
    external = s.audio.register_path(elsewhere)
    s.close()
    data = tmp_path / "b" / "mizuoto"
    data.parent.mkdir()
    shutil.move(str(first), str(data))

    s = _open(data)  # the start-up reconcile rewrites them
    try:
        recorded = {r["id"]: r["path"] for r in s.db.fetchall("SELECT id, path FROM outputs UNION ALL SELECT id, path FROM audio_files")}
        assert not is_absolute_text(recorded[output.id]) and not is_absolute_text(recorded[upload.id])
        assert recorded[external.id] == str(elsewhere.resolve())  # a file outside the data folder stays absolute
        assert s.outputs.path_of(output.id).is_file() and s.audio.path_of(upload.id).is_file()
        assert s.db.get_meta("data_dir") == str(data.resolve())
    finally:
        s.close()


def test_a_legacy_record_is_found_again_when_read(tmp_path, wav_file):
    """A record still absolute and stale when it is read (written by an older version after the folder moved)."""
    data = tmp_path / "mizuoto"
    output, _, _ = _populate(data, wav_file)
    s = _open(data)
    try:
        stale = "/gone/old/mizuoto/" + s.db.fetchone("SELECT path FROM outputs WHERE id = ?", (output.id,))["path"]
        s.db.execute("UPDATE outputs SET path = ? WHERE id = ?", (stale, output.id))
        assert s.outputs.path_of(output.id).is_file()
        assert not is_absolute_text(s.db.fetchone("SELECT path FROM outputs WHERE id = ?", (output.id,))["path"])
    finally:
        s.close()


def test_a_look_alike_is_not_taken_for_the_recorded_file(tmp_path, wav_file):
    data = tmp_path / "mizuoto"
    output, upload, _ = _populate(data, wav_file)
    s = _open(data)
    try:
        out_path, up_path = s.outputs.path_of(output.id), s.audio.path_of(upload.id)
        out_path.write_bytes(out_path.read_bytes() + b"\0")  # another size
        content = bytearray(up_path.read_bytes())
        content[-1] ^= 1  # same size, other content (audio files keep their hash)
        up_path.write_bytes(bytes(content))
        for table, row_id, path in (("outputs", output.id, out_path), ("audio_files", upload.id, up_path)):
            s.db.execute(f"UPDATE {table} SET path = ? WHERE id = ?", ("/gone/mizuoto/" + s.storage.store(path), row_id))
        with pytest.raises(NotFoundError) as out_error:
            s.outputs.path_of(output.id)
        with pytest.raises(NotFoundError):
            s.audio.path_of(upload.id)
    finally:
        s.close()
    detail = out_error.value.detail
    assert detail["reason"] == codes.FILE_MISSING and detail["kind"] == "output" and detail["recorded"].startswith("/gone/")


def test_missing_files_are_reported_never_skipped(tmp_path, wav_file):
    data = tmp_path / "mizuoto"
    output, upload, _ = _populate(data, wav_file)
    s = _open(data)
    try:
        s.outputs.path_of(output.id).unlink()
        assert s.outputs.get(output.id).missing is True
        assert [o.missing for o in s.outputs.list_outputs().items] == [True]
        with pytest.raises(NotFoundError) as e:
            b"".join(s.outputs.zip([output.id]))
        assert e.value.detail["ids"] == [output.id]
        with zipfile.ZipFile(BytesIO(b"".join(s.outputs.zip([output.id], skip_missing=True)))) as z:
            assert z.namelist() == ["MISSING.txt"] and output.id in z.read("MISSING.txt").decode()

        report = s.storage.check()
        assert [m.id for m in report.missing] == [output.id] and report.relative == 1 and not report.fixed
        pruned = s.storage.check(PathCheckRequest(prune_missing=True))
        assert pruned.pruned == 1
        with pytest.raises(NotFoundError):
            s.outputs.get(output.id)
        assert s.storage.check().missing == []
        assert s.outputs.delete([upload.id]) == []  # not an output
    finally:
        s.close()


def test_rebase_points_an_external_folder_elsewhere(tmp_path, wav_file):
    data = tmp_path / "mizuoto"
    old, new = tmp_path / "disk-old", tmp_path / "disk-new"
    old.mkdir()
    shutil.copyfile(wav_file, old / "clip.wav")
    s = _open(data)
    try:
        audio = s.audio.register_path(old / "clip.wav")
        shutil.move(str(old), str(new))
        assert s.storage.check().missing[0].id == audio.id
        report = s.storage.check(PathCheckRequest(rebase_from=str(old.resolve()), rebase_to=str(new)))
        assert report.rebased == 1 and report.missing == []
        assert s.audio.path_of(audio.id) == new / "clip.wav"
    finally:
        s.close()


def test_relative_folder_settings_are_inside_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = build_services(data_dir=tmp_path / "data", environ={"MIZUOTO_PATHS__OUTPUTS_DIR": "results"})
    try:
        assert s.settings.outputs_dir == (tmp_path / "data" / "results").resolve()
    finally:
        s.close()


def _open_and_report(path: str) -> str | None:
    from mizuoto.core.context import build_services

    try:
        build_services(data_dir=Path(path), environ={}).close()
        return None
    except Exception as e:  # reported to the test
        return f"{type(e).__name__}: {e}"


def test_processes_opening_a_moved_folder_at_once_all_succeed(tmp_path, wav_file):
    import multiprocessing as mp

    first = tmp_path / "a"
    output, _, _ = _populate(first, wav_file)
    _make_legacy(first)
    data = tmp_path / "b"
    shutil.move(str(first), str(data))
    with mp.get_context("spawn").Pool(4) as pool:
        assert pool.map(_open_and_report, [str(data)] * 4) == [None] * 4
    s = _open(data)
    try:
        assert s.outputs.path_of(output.id).is_file()
    finally:
        s.close()
