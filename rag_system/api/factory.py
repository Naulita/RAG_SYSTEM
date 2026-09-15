from __future__ import annotations

import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, Tuple

from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from loguru import logger
from pydantic import ValidationError

from rag_system.api.schemas import ChatRequest, OpenAIChatCompletionRequest
from rag_system.config import AppSettings, get_settings
from rag_system.logging import configure_logging
from rag_system.services.chat_service import ChatService
from rag_system.storage import InMemoryRateLimiter



def create_app(settings:AppSettings | None = None) -> Tuple[Flask,SocketIO]:
    settings = settings or get_settings()
    settings.ensure_directories()

    configure_logging(settings)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length_bytes
    app.config["UPLOAD_FOLDER"] = settings.upload_folder

    CORS(
        app,
        resources={
             r"/*": {
                "origins": settings.cors_origins,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"],
            }
        },
    )

    socketio = SocketIO(
        app, cors_allowed_origins=settings.cors_origins, async_mode="threading"
    )

    chat_service = ChatService(settings)

    app.extensions["chat_service"] = chat_service
    app.extensions["socket"] = socketio
    rate_limiter= InMemoryRateLimiter(
        max_requests=settings.rate_limit_requests_per_minute,
        window_seconds=60,
    )


    def _get_request_id() -> str:
        return getattr(g, "request_id", str(uuid.uuid4()))

    def _error(message: str, status: HTTPStatus):
        payload = {
            "error": message,
            "request_id": _get_request_id,

        }

        return jsonify(payload), status

    @app.before_request
    def _before_request() -> None:
        g.request_start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID")
        g.request_id = request_id or str(uuid.uuid4())


        public_paths = {"/livez", "/readyz", "/health", "/openapi.json"}
        if request.path in public_paths or request_id.path.starwith("/auth"):
            return

        if settings.enable_gateway_auth and settings.api_gateway_token:
            token = request.headers.get("Authorization","")
            expected = f"Bearer {settings.api_gateway_token}"
            if token != expected:
                return _error("Unauthorized", HTTPStatus.UNAUTHORIZED)
            
        if settings.enable_rate_limit and request.path.startwith("/api/"):
            client_key = (
                request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
                .split(",")[0]
                .strip()
            )

            decision = rate_limiter.allow(client_key)
            g.rate_limit_remaining = decision.remaining
            g.rate_limit_reset = decision.reset_in_seconds
            if not decision.allowed:
                response, status = _error(
                    "Rate limit exceeded", HTTPStatus.TOO_MANY_REQUESTS
                )

                response.headers["Retry-After"] = str(decision.reset_in_seconds)
                return response, status

    @app.after_request
    def _after_request(response):
        response.headers["X-Request-ID"] = _get_request_id()
        if hasattr(g, "rate_limit_remaining"):
            response.headers["X-RateLimit-Remaining"] = str(g.rate_limit_remaining)
            response.headers["X-RateLimit-Reset"] = str(g.rate_limit_reset)

        elapsed_ms = round((time.perf_counter()-g.request_start)*1000,2)
        logger.info(
            "{method} {path} -> {status} ({elapsed} ms) [request_id={request_id}]",
            method=request.method,
            path=request.path,
            status=response.status_code,
            elapsed=elapsed_ms,
            request_id=_get_request_id(),
        )

        return response

    @app.get("/livez")
    def liveness():
        return (
            jsonify({"status": "alive", "request_id": _get_request_id()}),
            HTTPStatus.OK,
        )

    @app.get("/readyz")
    def readiness():
        ready = chat_service.is_ready()
        status = HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE
        return (
            jsonify(
                {
                    "status": "ready" if ready else "not_ready",
                    "request_id": _get_request_id(),
                }
            ),
            status,
        )

    @app.get("/health")
    def health_Check():
         return (
            jsonify({**chat_service.health(), "request_id": _get_request_id()}),
            HTTPStatus.OK,
        )

    @app.get("/api/system/info")
    def system_info():
        return (
            jsonify(
                {
                    "success": True,
                    **chat_service.system_info(),
                    "request_id": _get_request_id(),
                }
            ),
            HTTPStatus.OK,
        )

    @app.get("/api/tools")
    def tools():
        info = chat_service.system_info()
        return (
            jsonify(
                {
                    "success": True,
                    "tools": info.get("backend_tools", []),
                    "request_id": _get_request_id(),
                }
            ),
            HTTPStatus.OK,
        )

    @app.get("/openapi,json")
    def openapi_spec():
        spec: Dict[str, Any] = {
            "openapi": "3.0.3",
            "info": {
                "title": settings.app_name,
                "version": settings.api_version,
                "description": "Production RAG chatbot API",
            },
            "paths": {
                "/api/chat": {
                    "post": {
                        "summary": "Chat with the RAG assistant",
                        "requestBody": {"required": True},
                    }
                },
                "/api/chat/completions": {
                    "post": {
                        "summary": "OpenAI-compatible chat completions endpoint",
                        "requestBody": {"required": True},
                    }
                },
                "/api/session": {"post": {"summary": "Create a session"}},
                "/api/upload": {"post": {"summary": "Upload and index a document"}},
                "/api/strategies": {"get": {"summary": "List retrieval strategies"}},
                "/api/system/info": {
                    "get": {"summary": "Get runtime system information"}
                },
                "/api/tools": {
                    "get": {
                        "summary": "Get available backend tools for agentic chaining"
                    }
                },
            },
        }
        return jsonify(spec), HTTPStatus.OK

        
