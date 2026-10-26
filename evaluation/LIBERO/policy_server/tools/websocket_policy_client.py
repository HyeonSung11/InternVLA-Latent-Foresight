import logging
import os
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Websocket policy client for simulation evaluation."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300.0) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for server at %s", self._uri)
        start_time = time.time()

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_attempts = [
                    {
                        "compression": None,
                        "max_size": None,
                        "additional_headers": headers,
                        "open_timeout": 30,
                        "ping_interval": 20,
                        "ping_timeout": 20,
                    },
                    {
                        "compression": None,
                        "max_size": None,
                        "additional_headers": headers,
                        "open_timeout": 30,
                    },
                    {
                        "compression": None,
                        "max_size": None,
                        "extra_headers": headers,
                        "open_timeout": 30,
                    },
                ]

                conn = None
                last_type_error: Optional[TypeError] = None
                for kwargs in connect_attempts:
                    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    try:
                        conn = websockets.sync.client.connect(self._uri, **call_kwargs)
                        break
                    except TypeError as exc:
                        last_type_error = exc
                        continue

                if conn is None:
                    raise RuntimeError(f"No compatible websockets.connect signature found: {last_type_error}")
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                logging.info("Still waiting for server %s ...", self._uri)
                time.sleep(1)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def predict_action(self, query_info: Dict) -> Dict:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
