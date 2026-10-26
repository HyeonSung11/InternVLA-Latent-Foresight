from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.LIBERO.policy_server.backends.backend_factory import build_backend
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy server for LIBERO evaluation.")
    parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint path for PI05 or InternVLA-A1.5.")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--stats_key", type=str, default="", help="Stats key in stats.json; auto-pick when empty.")
    parser.add_argument("--robot_type", type=str, default="", help="Robot type override for stats composition.")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close.")

    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="",
        help="Override VLM model path used by the chat processor (InternVLA-A1.5). "
        "If empty, falls back to vlm_model_name_or_path saved in the checkpoint config.",
    )

    parser.add_argument(
        "--wan_model_path",
        type=str,
        default="",
        help="Override the WAN checkpoint dir (InternVLA-A1.5 only). Ignored when "
        "--action_loss_only is enabled (the default).",
    )

    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default="",
        help="Override the WAN VAE weights path (InternVLA-A1.5 only). Defaults to "
        "<wan_model_path>/Wan2.2_VAE.pth when --wan_model_path is set.",
    )

    parser.add_argument(
        "--action_loss_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip WAN weight loading during inference (InternVLA-A1.5 only). "
        "Default: on. Use --no-action_loss_only when a full WAN checkpoint is available.",
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="standard",
        choices=("standard", "optimized"),
        help="Inference backend for InternVLA-A1.5. 'optimized' requires --action_loss_only.",
    )

    parser.add_argument("--mock_policy", type=str, default="", choices=["", "random"], help="Use mock backend instead of model checkpoint.")
    parser.add_argument("--mock_action_dim", type=int, default=7)
    parser.add_argument("--mock_chunk_size", type=int, default=16)

    parser.add_argument("--debug", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    stats_key = args.stats_key or None
    robot_type = args.robot_type or None
    mock_policy = args.mock_policy or None

    if not mock_policy and not args.ckpt_path:
        raise ValueError("--ckpt_path is required unless --mock_policy is set")

    backend = build_backend(
        ckpt_path=args.ckpt_path,
        device=args.device,
        stats_key=stats_key,
        robot_type=robot_type,
        resize_size=args.resize_size,
        mock_policy=mock_policy,
        mock_action_dim=args.mock_action_dim,
        mock_chunk_size=args.mock_chunk_size,
        vlm_model_path=args.vlm_model_path or None,
        wan_model_path=args.wan_model_path or None,
        wan_vae_path=args.wan_vae_path or None,
        action_loss_only=args.action_loss_only,
        inference_backend=args.inference_backend,
    )

    metadata = backend.metadata()
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Backend metadata: %s", metadata)

    server = WebsocketPolicyServer(
        policy=backend,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.mock_policy:
        raise RuntimeError("CUDA requested but not available")
    main(args)
