"""
LLM Providers Module - Integración con diferentes proveedores de LLM con visión
Soporta modelos locales via Ollama con host.docker.internal para uso en Docker
"""

import base64
import time
import json
import re
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
    
    def to_dict_full(self) -> Dict[str, Any]:
        """Versión completa con API key para almacenamiento"""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_params": self.extra_params
        }


def substitute_variables(text: str, context: Dict[str, Any]) -> str:
    """
    Sustituye variables en el texto usando el contexto JSON.
    Las variables se indican con $variable o $data.campo para acceder a campos anidados.
    """
    if not context:
        return text
    
    def get_nested_value(obj: Any, path: str) -> Any:
        """Obtiene un valor anidado de un objeto usando notación de punto"""
        parts = path.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current
    
    # Buscar todas las variables $variable o $path.to.value
    pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    
    def replace_var(match):
        var_path = match.group(1)
        value = get_nested_value(context, var_path)
        if value is not None:
            return str(value)
        return match.group(0)  # Mantener original si no se encuentra
    
    return re.sub(pattern, replace_var, text)


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
            
            # Construir payload base - solo con campos requeridos
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
                ]
            }
            
            # Modelos que usan max_completion_tokens en lugar de max_tokens
            # (modelos nuevos como gpt-4.5, gpt-5, o1, etc.)
            new_models = ["gpt-5", "gpt-4.5", "o1", "o3", "o1-mini", "o1-preview"]
            uses_new_api = any(self.config.model.startswith(m) for m in new_models)
            
            # Agregar max_tokens solo si está configurado (> 0)
            if self.config.max_tokens and self.config.max_tokens > 0:
                if uses_new_api:
                    payload["max_completion_tokens"] = self.config.max_tokens
                else:
                    payload["max_tokens"] = self.config.max_tokens
            
            # Agregar temperature solo si está configurado y el modelo lo soporta
            # (los modelos o1/o3 no soportan temperature)
            reasoning_models = ["o1", "o3"]
            is_reasoning_model = any(self.config.model.startswith(m) for m in reasoning_models)
            
            if self.config.temperature is not None and self.config.temperature >= 0 and not is_reasoning_model:
                payload["temperature"] = self.config.temperature
            
            # Agregar parámetros extra solo si están configurados
            extra = self.config.extra_params or {}
            if extra.get("top_p") is not None:
                payload["top_p"] = extra["top_p"]
            if extra.get("frequency_penalty") is not None:
                payload["frequency_penalty"] = extra["frequency_penalty"]
            if extra.get("presence_penalty") is not None:
                payload["presence_penalty"] = extra["presence_penalty"]
            if extra.get("seed") is not None:
                payload["seed"] = extra["seed"]
            
            api_base = self.config.api_base or "https://api.openai.com/v1"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # Capturar el cuerpo de la respuesta para errores
                response_text = response.text
                
                # Manejar errores HTTP con mensajes claros
                if response.status_code != 200:
                    error_detail = f"Error HTTP {response.status_code}"
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_obj = error_json["error"]
                            error_message = error_obj.get("message", "")
                            error_type = error_obj.get("type", "")
                            error_code = error_obj.get("code", "")
                            error_detail = f"[{error_type}] {error_message}"
                            if error_code:
                                error_detail += f" (code: {error_code})"
                        else:
                            error_detail = str(error_json)
                    except:
                        error_detail = f"Error HTTP {response.status_code}: {response_text[:500]}"
                    
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider=self.config.provider,
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_detail,
                        success=False,
                        raw_response={"status_code": response.status_code, "body": response_text[:1000]}
                    )
                
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
            
        except httpx.TimeoutException:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error="Timeout: La solicitud tardó demasiado. Intente con una imagen más pequeña o aumente el timeout.",
                success=False
            )
        except httpx.ConnectError as e:
            latency = (time.time() - start_time) * 1000
            api_base = self.config.api_base or "https://api.openai.com/v1"
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error=f"Error de conexión a {api_base}: {str(e)}",
                success=False
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error=f"Error inesperado: {type(e).__name__}: {str(e)}",
                success=False
            )
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en el modelo"""
        pricing = {
            # OpenAI GPT-4o series
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            # OpenAI GPT-4.1 series
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            # OpenAI GPT-5 series (estimado)
            "gpt-5": {"input": 5.00, "output": 15.00},
            "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
            # OpenAI o1/o3 reasoning models
            "o1": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o1-preview": {"input": 15.00, "output": 60.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            # Google Gemini
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
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
            
            # Agregar parámetros extra si existen
            extra = self.config.extra_params or {}
            if "top_p" in extra:
                payload["top_p"] = extra["top_p"]
            if "top_k" in extra:
                payload["top_k"] = extra["top_k"]
            if self.config.temperature is not None:
                payload["temperature"] = self.config.temperature
            
            api_base = self.config.api_base or "https://api.anthropic.com/v1"
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_base}/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = f"Error HTTP {response.status_code}"
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_obj = error_json["error"]
                            error_message = error_obj.get("message", "")
                            error_type = error_obj.get("type", "")
                            error_detail = f"[{error_type}] {error_message}"
                        else:
                            error_detail = str(error_json)
                    except:
                        error_detail = f"Error HTTP {response.status_code}: {response.text[:500]}"
                    
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider="anthropic",
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_detail,
                        success=False
                    )
                
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
                provider="anthropic",
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
                provider="anthropic",
                latency_ms=latency,
                error=f"Error: {type(e).__name__}: {str(e)}",
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
    """
    Proveedor para modelos locales via Ollama.
    Soporta host.docker.internal para acceder a Ollama desde Docker.
    """
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            
            # Usar api_base configurado o localhost por defecto
            # Para Docker, usar host.docker.internal:11434
            api_base = self.config.api_base or "http://localhost:11434"
            
            options = {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
            
            # Agregar parámetros extra si existen
            extra = self.config.extra_params or {}
            if "top_p" in extra:
                options["top_p"] = extra["top_p"]
            if "top_k" in extra:
                options["top_k"] = extra["top_k"]
            if "repeat_penalty" in extra:
                options["repeat_penalty"] = extra["repeat_penalty"]
            if "seed" in extra:
                options["seed"] = extra["seed"]
            
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": options
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{api_base}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = f"Error HTTP {response.status_code}: {response.text[:500]}"
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider="ollama",
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_detail,
                        success=False
                    )
                
                data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
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
            
        except httpx.ConnectError as e:
            latency = (time.time() - start_time) * 1000
            api_base = self.config.api_base or "http://localhost:11434"
            return LLMResponse(
                content="",
                model=self.config.model,
                provider="ollama",
                latency_ms=latency,
                error=f"No se puede conectar a Ollama en {api_base}. Verifique que Ollama está corriendo.",
                success=False
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.config.model,
                provider="ollama",
                latency_ms=latency,
                error=f"Error: {type(e).__name__}: {str(e)}",
                success=False
            )


class GoogleProvider(BaseLLMProvider):
    """Proveedor para Google Gemini"""
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        try:
            image_data = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            
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
                
                # Agregar parámetros extra
                extra = self.config.extra_params or {}
                if "top_p" in extra:
                    payload["top_p"] = extra["top_p"]
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.config.api_base}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code != 200:
                        error_detail = f"Error HTTP {response.status_code}: {response.text[:500]}"
                        return LLMResponse(
                            content="",
                            model=self.config.model,
                            provider="google",
                            latency_ms=(time.time() - start_time) * 1000,
                            error=error_detail,
                            success=False
                        )
                    
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
                
                generation_config = {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens
                }
                
                # Agregar parámetros extra
                extra = self.config.extra_params or {}
                if "top_p" in extra:
                    generation_config["topP"] = extra["top_p"]
                if "top_k" in extra:
                    generation_config["topK"] = extra["top_k"]
                
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
                    "generationConfig": generation_config
                }
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload)
                    
                    if response.status_code != 200:
                        error_detail = f"Error HTTP {response.status_code}"
                        try:
                            error_json = response.json()
                            if "error" in error_json:
                                error_detail = error_json["error"].get("message", error_detail)
                        except:
                            error_detail = f"Error HTTP {response.status_code}: {response.text[:500]}"
                        
                        return LLMResponse(
                            content="",
                            model=self.config.model,
                            provider="google",
                            latency_ms=(time.time() - start_time) * 1000,
                            error=error_detail,
                            success=False
                        )
                    
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
                error=f"Error: {type(e).__name__}: {str(e)}",
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
