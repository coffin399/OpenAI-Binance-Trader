from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

try:  # Optional dependency for Gemini SDK
    import google.genai as google_genai
except ImportError:  # pragma: no cover - optional
    google_genai = None


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    temperature: float = 0.0
    api_keys: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_attempts: Optional[int] = None


class ProviderError(Exception):
    pass


class AllAPIKeysExhausted(ProviderError):
    pass


class AIProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._index = 0
        self._session = requests.Session()

    def _next_key(self) -> str:
        if not self.config.api_keys:
            raise AllAPIKeysExhausted(f"No API keys configured for {self.config.name}")
        with self._lock:
            key = self.config.api_keys[self._index]
            self._index = (self._index + 1) % len(self.config.api_keys)
        return key

    def generate(self, prompt: str, max_attempts: Optional[int] = None) -> str:
        attempts = 0
        configured_attempts = self.config.max_attempts or len(self.config.api_keys)
        max_attempts = max_attempts or configured_attempts

        while attempts < max_attempts:
            api_key = self._next_key()
            attempts += 1
            try:
                return self._call_api(prompt, api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Provider %s failed with key %s (attempt %s/%s): %s",
                    self.config.name,
                    api_key[:6] + "..." if api_key else "<empty>",
                    attempts,
                    max_attempts,
                    exc,
                )
                time.sleep(0.5)
        raise AllAPIKeysExhausted(
            f"All API keys exhausted for provider {self.config.name}"
        )

    def _call_api(self, prompt: str, api_key: str) -> str:
        if self.config.name.lower() == "gemini" or "generativelanguage.googleapis.com" in self.config.base_url:
            return self._call_gemini(prompt, api_key)

        system_prompt = self.config.system_prompt or "You are a trading strategy assistant. Reply with BUY, SELL, or HOLD."  # noqa: E501
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        response = self._session.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        if response.status_code == 401:
            raise ProviderError("Unauthorized - likely invalid API key")
        if response.status_code >= 400:
            raise ProviderError(
                f"Provider {self.config.name} error {response.status_code}: {response.text}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Provider returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise ProviderError("Provider returned empty content")
        return content

    def _call_gemini(self, prompt: str, api_key: str) -> str:
        if google_genai is not None:
            try:
                return self._call_gemini_sdk(prompt, api_key)
            except Exception as sdk_error:  # noqa: BLE001 - fallback to REST
                logger.warning(
                    "Gemini SDK call failed (%s). Falling back to HTTP request.", sdk_error
                )
        return self._call_gemini_http(prompt, api_key)

    def _call_gemini_sdk(self, prompt: str, api_key: str) -> str:
        if google_genai is None:
            raise ProviderError("google.genai library not installed")

        client = google_genai.Client(api_key=api_key)
        messages: List[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderError("Gemini SDK returned no choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        text: str = ""

        if isinstance(message, dict):
            text = (message.get("content") or message.get("text") or "").strip()
        elif message is not None:
            text = (
                getattr(message, "content", None)
                or getattr(message, "text", "")
            )
            if isinstance(text, list):
                text = " ".join(str(part) for part in text)
            text = (text or "").strip()

        if not text and hasattr(first_choice, "content"):
            choice_content = getattr(first_choice, "content")
            if isinstance(choice_content, str):
                text = choice_content.strip()
            elif isinstance(choice_content, list):
                text = " ".join(str(part) for part in choice_content).strip()

        if not text:
            raise ProviderError("Gemini SDK returned empty content")

        return text

    def _call_gemini_http(self, prompt: str, api_key: str) -> str:
        system_prompt = self.config.system_prompt or "You are a trading strategy assistant. Reply with BUY, SELL, or HOLD."  # noqa: E501
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "model": self.config.model,
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                    ],
                }
            ],
            "safetySettings": [],
            "generationConfig": {
                "temperature": self.config.temperature,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [
                    {"text": system_prompt},
                ]
            }

        endpoint = f"{self.config.base_url.rstrip('/')}/models/{self.config.model}:generateContent"
        response = self._session.post(endpoint, json=payload, headers=headers, timeout=30)
        if response.status_code == 401:
            raise ProviderError("Unauthorized - likely invalid API key")
        if response.status_code >= 400:
            raise ProviderError(
                f"Provider {self.config.name} error {response.status_code}: {response.text}"
            )

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("Provider returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = " ".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise ProviderError("Provider returned empty content")
        return text


class AIProviderManager:
    def __init__(self, ai_config_section) -> None:
        self.providers: Dict[str, AIProvider] = {}
        if not ai_config_section:
            return
        providers_config = getattr(ai_config_section, "providers", [])
        for entry in providers_config:
            provider = self._build_provider(entry)
            if provider:
                self.providers[provider.config.name] = provider

    def _build_provider(self, entry) -> Optional[AIProvider]:
        try:
            max_attempts = entry.get("max_attempts")
            if max_attempts is not None:
                try:
                    max_attempts = int(max_attempts)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid max_attempts for provider %s. Falling back to default.",
                        entry.get("name"),
                    )
                    max_attempts = None

            config = ProviderConfig(
                name=entry.get("name"),
                base_url=entry.get("base_url"),
                model=entry.get("model"),
                temperature=float(entry.get("temperature", 0.0)),
                api_keys=list(entry.get("api_keys", [])),
                system_prompt=entry.get("system_prompt"),
                max_attempts=max_attempts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Invalid AI provider configuration: %s", exc)
            return None

        if not config.name or not config.base_url or not config.model:
            logger.error("Incomplete AI provider configuration: %s", entry)
            return None

        return AIProvider(config)

    def has_provider(self, name: str) -> bool:
        return name in self.providers

    def generate(self, provider_name: str, prompt: str) -> str:
        provider = self.providers.get(provider_name)
        if not provider:
            raise ProviderError(f"Provider '{provider_name}' not configured")
        return provider.generate(prompt)

    def available_strategies(self) -> Iterable[str]:
        return self.providers.keys()
