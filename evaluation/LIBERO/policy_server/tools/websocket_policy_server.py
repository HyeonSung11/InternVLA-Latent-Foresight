import asyncio
import logging
import time
import traceback
from typing import Any

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serve a policy backend with a websocket protocol."""

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._protocol_version = str(self._metadata.get("protocol_version", "2.0"))
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info("Idle timeout (%ss) reached, shutting down server.", self._idle_timeout)
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _route_message(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        msg_type = msg.get("type", "infer")

        if msg_type == "ping":
            return {
                "status": "ok",
                "ok": True,
                "type": "ping",
                "request_id": req_id,
                "protocol_version": self._protocol_version,
            }

        if msg_type in ("infer", "predict_action"):
            payload = msg.get("payload", msg)
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "protocol_version": self._protocol_version,
                    "error": {
                        "message": "Payload must be a dict",
                        "payload_type": str(type(payload)),
                    },
                }

            try:
                if hasattr(self._policy, "predict_action"):
                    output = self._policy.predict_action(payload)
                else:
                    output = self._policy.infer(payload)
            except Exception as exc:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "protocol_version": self._protocol_version,
                    "error": {"message": str(exc)},
                }

            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "protocol_version": self._protocol_version,
                "data": output,
            }

        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "protocol_version": self._protocol_version,
            "error": {"message": f"Unsupported message type '{msg_type}'"},
        }
