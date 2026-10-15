"""Print the OpenAPI schema without starting a server."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mizuoto.api.app import create_app
from mizuoto.core.context import build_services


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        services = build_services(data_dir=Path(tmp), environ={})
        try:
            app = create_app(services, serve_ui=False)
            sys.stdout.write(json.dumps(app.openapi(), indent=2, sort_keys=False) + "\n")
        finally:
            services.close()


if __name__ == "__main__":
    main()
