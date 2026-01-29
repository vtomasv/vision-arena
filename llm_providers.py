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
    
    # Modelos de OpenAI que soportan visión
    VISION_MODELS = {
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        # Modelos compatibles via proxy (Manus)
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gemini-2.5-flash",
    }
    
    # Modelos que NO soportan visión
    NON_VISION_MODELS = {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
    }
    
    def _validate_model(self) -> tuple:
        """Valida si el modelo soporta visión. Retorna (is_valid, error_message)"""
        model = self.config.model.lower()
        
        # Si hay api_base personalizado (no OpenAI directo), asumir que el modelo es válido
        if self.config.api_base and "api.openai.com" not in self.config.api_base:
            return (True, "")
        
        # Verificar modelos conocidos que NO soportan visión
        for non_vision in self.NON_VISION_MODELS:
            if non_vision in model:
                return (False, f"El modelo '{self.config.model}' no soporta análisis de imágenes. Use gpt-4o, gpt-4o-mini o gpt-4-turbo para visión.")
        
        # Verificar si es un modelo de visión conocido
        for vision_model in self.VISION_MODELS:
            if vision_model in model:
                return (True, "")
        
        # Modelo desconocido - advertir pero intentar
        return (True, "")
    
    async def analyze_image(self, image_path: str, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        # Validar modelo antes de hacer la llamada
        is_valid, error_msg = self._validate_model()
        if not is_valid:
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=0,
                error=error_msg,
                success=False
            )
        
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
                
                # Manejar errores HTTP con mensajes más claros
                if response.status_code == 404:
                    error_detail = f"Modelo '{self.config.model}' no encontrado. "
                    if "api.openai.com" in api_base:
                        error_detail += "Verifique que el modelo existe y soporta visión. Modelos válidos: gpt-4o, gpt-4o-mini, gpt-4-turbo"
                    else:
                        error_detail += f"Verifique que el modelo está disponible en {api_base}"
                    
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider=self.config.provider,
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_detail,
                        success=False
                    )
                
                if response.status_code == 401:
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider=self.config.provider,
                        latency_ms=(time.time() - start_time) * 1000,
                        error="API Key inválida o sin permisos. Verifique su configuración.",
                        success=False
                    )
                
                if response.status_code == 429:
                    return LLMResponse(
                        content="",
                        model=self.config.model,
                        provider=self.config.provider,
                        latency_ms=(time.time() - start_time) * 1000,
                        error="Límite de rate excedido. Espere un momento e intente nuevamente.",
                        success=False
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
            
        except httpx.HTTPStatusError as e:
            latency = (time.time() - start_time) * 1000
            error_msg = f"Error HTTP {e.response.status_code}: {str(e)}"
            
            # Intentar extraer mensaje de error del body
            try:
                error_body = e.response.json()
                if "error" in error_body:
                    error_msg = error_body["error"].get("message", error_msg)
            except:
                pass
            
            return LLMResponse(
                content="",
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                error=error_msg,
                success=False
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
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
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
