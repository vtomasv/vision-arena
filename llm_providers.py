"""
LLM Providers Module - Integración con diferentes proveedores de LLM con visión
"""

import base64
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import httpx


@dataclass
class LLMResponse:
    """Respuesta estructurada de un LLM"""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True


@dataclass
class LLMConfig:
    """Configuración de un proveedor LLM"""
    name: str
    provider: str
    model: str
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key[:8] + "..." if self.api_key else "",
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_params": self.extra_params
        }


class BaseLLMProvider(ABC):
    """Clase base abstracta para proveedores de LLM"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        """Analiza una imagen con el prompt dado"""
        pass
    
    def _encode_image(self, image_path: str) -> str:
        """Codifica una imagen en base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Obtiene el tipo MIME de la imagen"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(ext, "image/jpeg")


class OpenAIProvider(BaseLLMProvider):
    """Proveedor para OpenAI y APIs compatibles (incluyendo modelos locales via Ollama)"""
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            
            headers = {
                "Content-Type": "application/json",
            }
            
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                **self.config.extra_params
            }
            
            api_base = self.config.api_base or "https://api.openai.com/v1"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.config.model,
                provider=self.config.provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=latency,
                cost_estimate=self._estimate_cost(input_tokens, output_tokens),
                raw_response=data,
                success=True
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error=str(e),
                success=False
            )
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en el modelo"""
        # Precios aproximados por 1M tokens (pueden variar)
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        }
        
        model_pricing = pricing.get(self.config.model, {"input": 0, "output": 0})
        cost = (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1_000_000
        return round(cost, 6)


class AnthropicProvider(BaseLLMProvider):
    """Proveedor para Anthropic Claude"""
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            api_base = self.config.api_base or "https://api.anthropic.com/v1"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_base}/messages",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=latency,
                cost_estimate=self._estimate_cost(input_tokens, output_tokens),
                raw_response=data,
                success=True
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error=str(e),
                success=False
            )
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en el modelo Claude"""
        pricing = {
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        }
        
        model_pricing = pricing.get(self.config.model, {"input": 0, "output": 0})
        cost = (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1_000_000
        return round(cost, 6)


class OllamaProvider(BaseLLMProvider):
    """Proveedor para modelos locales via Ollama"""
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            
            api_base = self.config.api_base or "http://localhost:11434"
            
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{api_base}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            # Ollama proporciona métricas diferentes
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            
            return LLMResponse(
                content=data.get("response", ""),
                model=self.config.model,
                provider="ollama",
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
                latency_ms=latency,
                cost_estimate=0.0,  # Modelos locales sin costo
                raw_response=data,
                success=True
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider="ollama",
                latency_ms=latency,
                error=str(e),
                success=False
            )


class GoogleProvider(BaseLLMProvider):
    """Proveedor para Google Gemini"""
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            
            # Usar el endpoint de OpenAI compatible si está configurado
            if self.config.api_base:
                # Usar formato OpenAI compatible
                headers = {
                    "Content-Type": "application/json",
                }
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                payload = {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.config.api_base}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                
                latency = (time.time() - start_time) * 1000
                usage = data.get("usage", {})
                
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=self.config.model,
                    provider="google",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    latency_ms=latency,
                    cost_estimate=0.0,
                    raw_response=data,
                    success=True
                )
            else:
                # API nativa de Google
                api_key = self.config.api_key
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={api_key}"
                
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": media_type,
                                        "data": image_data
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens
                    }
                }
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                
                latency = (time.time() - start_time) * 1000
                
                content = ""
                if "candidates" in data and data["candidates"]:
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    content = "".join(p.get("text", "") for p in parts)
                
                usage = data.get("usageMetadata", {})
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="google",
                    input_tokens=usage.get("promptTokenCount", 0),
                    output_tokens=usage.get("candidatesTokenCount", 0),
                    total_tokens=usage.get("totalTokenCount", 0),
                    latency_ms=latency,
                    cost_estimate=0.0,
                    raw_response=data,
                    success=True
                )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider="google",
                latency_ms=latency,
                error=str(e),
                success=False
            )


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory para crear el proveedor correcto basado en la configuración"""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "google": GoogleProvider,
        "openai-compatible": OpenAIProvider,
    }
    
    provider_class = providers.get(config.provider.lower(), OpenAIProvider)
    return provider_class(config)
