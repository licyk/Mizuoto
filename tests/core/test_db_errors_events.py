from mizuoto.core import codes
from mizuoto.core.db.database import MIGRATIONS, Database
from mizuoto.core.errors import ERROR_CLASSES, AssetMissingError, MizuotoError, ModelMissingError, ParameterError, error_from_code
from mizuoto.core.events import LocalEventBus
from mizuoto.core.events.models import EventBase, SettingsChangedEvent


def test_migrations_and_client_state(tmp_path):
    db = Database(tmp_path / "k.db")
    assert db.schema_version == len(MIGRATIONS)
    tables = {r["name"] for r in db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "voices", "audio_files", "transcripts", "outputs", "presets", "client_state"} <= tables
    db.set_client_state("prefs", {"theme": "dark"})
    assert db.get_client_state("prefs") == {"theme": "dark"}
    db.close()
    again = Database(tmp_path / "k.db")
    assert again.created is False and again.get_client_state("prefs") == {"theme": "dark"}
    again.close()


def test_error_codes_are_unique_and_round_trip():
    for code, cls in ERROR_CLASSES.items():
        err = error_from_code(code, "msg", {"x": 1})
        assert isinstance(err, cls)
        assert err.code == code
        assert err.detail["x"] == 1
    assert isinstance(error_from_code(codes.ASSET_MISSING, "m", {"assets": ["a"]}), AssetMissingError)
    p = error_from_code(codes.PARAMETER_OUT_OF_RANGE, "m", {"fields": [{"name": "speed"}]})
    assert isinstance(p, ParameterError) and p.code == codes.PARAMETER_OUT_OF_RANGE and p.http_status == 422
    unknown = error_from_code("something_new", "m")
    assert type(unknown) is MizuotoError and unknown.code == "something_new"
    assert ModelMissingError.exit_code == 7


def test_event_bus_and_registry():
    bus = LocalEventBus()
    got: list[EventBase] = []
    unsubscribe = bus.subscribe(got.append)

    def failing(event: EventBase) -> None:
        raise RuntimeError("a failing handler does not stop the others")

    bus.subscribe(failing)
    bus.publish(SettingsChangedEvent(keys=["a"]))
    unsubscribe()
    bus.publish(SettingsChangedEvent(keys=["b"]))
    assert [e.keys for e in got] == [["a"]]  # ty: ignore[unresolved-attribute]
    assert SettingsChangedEvent in EventBase.get_events()


def test_services_publish_settings_changes(services):
    got = []
    services.events.subscribe(got.append)
    services.settings.update({"server": {"port": 8123}})
    assert got and got[-1].keys == ["server.port"]
