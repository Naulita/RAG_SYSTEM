from __future__ import annotations

from typing import Any, Dict, List

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


from rag_system.config import AppSettings

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)


class BackendApiClient:
    """Client for backend routes used by the agentic RAG orchestrator."""

    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._session = requests.Session()
        self._session.headers.update({"Authorization":f"Bearer {settings.api_token}"})


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,)
    def _get(
        self, endpoint: str, params: Dict[str, Any] | None = None
    ) -> Dict[str, Any] | List[Any]:
        url = f"{self._settings.api_base_url.rstrip('/')}{endpoint}"
        response = self._session.get(
            url, params=params or {}, timeout=self._settings.api_timeout_seconds
        )
        response.raise_for_status()
        return response.json()

    def _safe_get(
            self, endopoint: str, params: Dict[str,Any] | None = None
    ) -> Dict[str, Any] | List[Any] | None:
        try:
            return self._get(endopoint, params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknow"
            if status == 404:
                return None
            raise

    def ping(self) -> Dict[str, Any]:
        payload= self._get("/ping")
        return payload if isinstance(payload, dict) else {"data": payload}

    def is_available(self) -> bool:
        try: 
            self.ping()
            return True
        except Exception as exc:
            logger.warning("Backend API ping failded: {}", exc)
            return False
    