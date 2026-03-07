from __future__ import annotations

import os

from app.api import create_app
from app.config import get_server_host, get_server_port, parse_bool_env

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=get_server_host(),
        port=get_server_port(),
        reload=parse_bool_env(os.getenv("KOKORO_RELOAD"), default=False),
    )
