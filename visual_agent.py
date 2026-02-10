"""
Visual Agent Module - Agente interactivo especializado en tratamiento de imágenes.
Puede ejecutar código Python sobre imágenes, ampliarlas, cortarlas, analizar regiones,
y generar documentos firmados con los resultados.

El agente planifica artefactos antes de ejecutar, y devuelve mensajes estructurados
con partes (plan, texto markdown, código, imágenes, resultados) para renderizado rico.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from llm_providers import LLMConfig, create_provider


# Librerías de procesamiento de imágenes que deben estar disponibles
REQUIRED_PACKAGES = [
    "Pillow", "numpy", "opencv-python-headless", "matplotlib"
]

SYSTEM_PROMPT = """You are an advanced Visual Forensic Agent specialized in image analysis and processing.
You have access to a Python execution environment with the following libraries pre-installed:
- PIL/Pillow (image manipulation)
- numpy (numerical operations)  
- cv2/OpenCV (computer vision)
- matplotlib (plotting and visualization)

WORKFLOW - You MUST follow this structure in every response:

## 1. PLAN (Required)
First, present a brief plan of what you will do. Use a markdown section like:

### Plan
1. [First artifact/step]
2. [Second artifact/step]
3. [Expected output]

## 2. EXECUTION
Write Python code blocks to perform the operations. Each code block will be automatically executed.

## 3. RESULTS
After code execution, present findings using rich Markdown formatting:
- Use **bold** for emphasis
- Use tables for structured data
- Use headers (##, ###) to organize sections
- Reference generated images using: ![description](filename.png)

IMPORTANT RULES:
- The input image path is available as IMAGE_PATH
- Save output images to OUTPUT_DIR (both are injected automatically)
- Use descriptive filenames for output images (e.g., "cropped_plate.png", "zoomed_face.png")
- Always use `plt.savefig()` instead of `plt.show()` for matplotlib plots
- Print relevant findings to stdout - these will be shown to the user
- Handle errors gracefully with try/except
- When analyzing specific regions, provide bounding box coordinates
- For forensic analysis, maintain image integrity and document all transformations
- ALWAYS reference output images in your markdown using ![description](filename.png) syntax
- Use Markdown formatting throughout your response for rich display

RESPONSE FORMAT EXAMPLE:
```
### Plan
1. Crop the vehicle from the main image
2. Enhance the license plate region
3. Analyze and report findings

Let me start by isolating the vehicle...

```python
import cv2
# ... code here
```

### Results

The analysis reveals:

| Feature | Value |
|---------|-------|
| Vehicle Type | Sedan |
| Color | Silver |
| Plate | ABC-123 |

![Cropped vehicle](cropped_vehicle.png)

The license plate after enhancement:

![Enhanced plate](enhanced_plate.png)
```

You can perform these operations:
- CROP: Extract specific regions of interest
- ZOOM: Magnify areas for detailed inspection
- ENHANCE: Improve contrast, brightness, sharpness
- ANNOTATE: Add bounding boxes, labels, arrows
- MEASURE: Calculate dimensions, distances, angles
- COMPARE: Side-by-side comparison of regions
- FILTER: Apply various image filters
- DETECT: Edge detection, contour finding
- COLOR ANALYSIS: Extract dominant colors, histograms
- OCR PREPARATION: Pre-process for text recognition
"""


@dataclass
class MessagePart:
    """Una parte estructurada de un mensaje del agente"""
    type: str  # "plan", "text", "code", "code_output", "image", "error"
    content: str = ""
    language: str = "python"  # Para bloques de código
    output: str = ""  # Salida del código
    success: bool = True  # Si el código se ejecutó correctamente
    filename: str = ""  # Para imágenes
    caption: str = ""  # Para imágenes
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type, "content": self.content}
        if self.type == "code":
            d["language"] = self.language
            d["output"] = self.output
            d["success"] = self.success
        elif self.type == "image":
            d["filename"] = self.filename
            d["caption"] = self.caption
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessagePart":
        return cls(
            type=data.get("type", "text"),
            content=data.get("content", ""),
            language=data.get("language", "python"),
            output=data.get("output", ""),
            success=data.get("success", True),
            filename=data.get("filename", ""),
            caption=data.get("caption", "")
        )


@dataclass
class AgentMessage:
    """Mensaje en la conversación del agente"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""  # Contenido raw (markdown) para compatibilidad
    parts: List[MessagePart] = field(default_factory=list)  # Partes estructuradas
    images: List[str] = field(default_factory=list)  # Filenames de imágenes generadas
    code: str = ""  # Código Python ejecutado (legacy)
    code_output: str = ""  # Salida del código (legacy)
    code_success: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "parts": [p.to_dict() for p in self.parts],
            "images": self.images,
            "code": self.code,
            "code_output": self.code_output,
            "code_success": self.code_success,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        msg = cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            images=data.get("images", []),
            code=data.get("code", ""),
            code_output=data.get("code_output", ""),
            code_success=data.get("code_success", True),
            timestamp=data.get("timestamp", "")
        )
        msg.parts = [MessagePart.from_dict(p) for p in data.get("parts", [])]
        return msg


@dataclass
class AgentSession:
    """Sesión de conversación con el agente"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_path: str = ""
    image_name: str = ""
    llm_config_name: str = ""
    messages: List[AgentMessage] = field(default_factory=list)
    output_dir: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    title: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "image_name": self.image_name,
            "llm_config_name": self.llm_config_name,
            "messages": [m.to_dict() for m in self.messages],
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "title": self.title
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        session = cls(
            id=data.get("id", str(uuid.uuid4())),
            image_path=data.get("image_path", ""),
            image_name=data.get("image_name", ""),
            llm_config_name=data.get("llm_config_name", ""),
            output_dir=data.get("output_dir", ""),
            created_at=data.get("created_at", ""),
            title=data.get("title", "")
        )
        session.messages = [
            AgentMessage.from_dict(m) for m in data.get("messages", [])
        ]
        return session


class CodeExecutor:
    """Ejecutor seguro de código Python para procesamiento de imágenes"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def execute(self, code: str, image_path: str) -> Dict[str, Any]:
        """
        Ejecuta código Python con acceso a la imagen.
        
        Returns:
            Dict con stdout, stderr, success, generated_images (filenames only)
        """
        # Crear script temporal
        script_content = f'''
import sys
import os
os.environ["MPLBACKEND"] = "Agg"

IMAGE_PATH = {repr(image_path)}
OUTPUT_DIR = {repr(self.output_dir)}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Importaciones comunes
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import numpy as np
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- User Code ---
{code}
'''
        
        # Guardar script
        script_path = os.path.join(self.output_dir, f"_agent_script_{uuid.uuid4().hex[:8]}.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Listar archivos antes de la ejecución
        files_before = set(os.listdir(self.output_dir))
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.output_dir
            )
            
            stdout = result.stdout
            stderr = result.stderr
            success = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "Error: Code execution timed out (120s limit)"
            success = False
        except Exception as e:
            stdout = ""
            stderr = f"Error executing code: {str(e)}"
            success = False
        finally:
            # Limpiar script temporal
            try:
                os.remove(script_path)
            except:
                pass
        
        # Detectar imágenes generadas (solo filenames, no paths completos)
        files_after = set(os.listdir(self.output_dir))
        new_files = files_after - files_before
        
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
        generated_images = sorted([
            f for f in new_files 
            if Path(f).suffix.lower() in image_extensions
        ])
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": success,
            "generated_images": generated_images  # Solo filenames
        }


def parse_response_to_parts(raw_content: str, code_results: List[Dict], session_id: str) -> List[MessagePart]:
    """
    Parsea la respuesta raw del LLM en partes estructuradas.
    
    Identifica:
    - Bloques de código Python (```python ... ```)
    - Referencias a imágenes (![...](filename.png))
    - Secciones de plan (### Plan, ## Plan)
    - Texto markdown regular
    """
    parts = []
    
    # Dividir el contenido en segmentos: texto y bloques de código
    # Pattern: captura texto antes de un bloque de código, luego el bloque
    segments = re.split(r'(```(?:python|py)?\s*\n.*?\n```)', raw_content, flags=re.DOTALL)
    
    code_idx = 0  # Índice para mapear con code_results
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Es un bloque de código?
        code_match = re.match(r'```(?:python|py)?\s*\n(.*?)\n```', segment, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            
            # Buscar resultado de ejecución correspondiente
            output = ""
            success = True
            code_images = []
            if code_idx < len(code_results):
                cr = code_results[code_idx]
                output = cr.get("output", "")
                success = cr.get("success", True)
                code_images = cr.get("images", [])
                code_idx += 1
            
            # Agregar parte de código
            parts.append(MessagePart(
                type="code",
                content=code_content,
                language="python",
                output=output,
                success=success
            ))
            
            # Agregar imágenes generadas por este bloque de código
            for img_filename in code_images:
                parts.append(MessagePart(
                    type="image",
                    filename=img_filename,
                    caption=img_filename.replace("_", " ").replace(".png", "").replace(".jpg", "").title()
                ))
        else:
            # Es texto markdown - procesar sub-secciones
            text_parts = _parse_text_segment(segment)
            parts.extend(text_parts)
    
    return parts


def _parse_text_segment(text: str) -> List[MessagePart]:
    """Parsea un segmento de texto markdown en partes."""
    parts = []
    
    # Detectar si contiene una sección de plan
    plan_match = re.search(r'(#{1,3}\s*(?:Plan|Planificación|Planning)\s*\n)(.*?)(?=\n#{1,3}\s|\Z)', 
                           text, re.DOTALL | re.IGNORECASE)
    
    if plan_match:
        # Texto antes del plan
        before = text[:plan_match.start()].strip()
        if before:
            parts.append(MessagePart(type="text", content=before))
        
        # El plan
        plan_content = (plan_match.group(1) + plan_match.group(2)).strip()
        parts.append(MessagePart(type="plan", content=plan_content))
        
        # Texto después del plan
        after = text[plan_match.end():].strip()
        if after:
            parts.append(MessagePart(type="text", content=after))
    else:
        # Sin plan detectado, todo es texto
        if text.strip():
            parts.append(MessagePart(type="text", content=text.strip()))
    
    return parts


class VisualAgent:
    """Agente visual interactivo para procesamiento de imágenes"""
    
    def __init__(self, sessions_dir: str, config_loader=None):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.config_loader = config_loader
        self.active_sessions: Dict[str, AgentSession] = {}
    
    def create_session(self, image_path: str, image_name: str, 
                       llm_config_name: str, title: str = "") -> AgentSession:
        """Crea una nueva sesión de agente"""
        session = AgentSession(
            image_path=image_path,
            image_name=image_name,
            llm_config_name=llm_config_name,
            title=title or f"Session - {image_name}"
        )
        
        # Crear directorio de salida para esta sesión
        session.output_dir = str(self.sessions_dir / session.id / "outputs")
        os.makedirs(session.output_dir, exist_ok=True)
        
        # Mensaje de sistema
        system_msg = AgentMessage(
            role="system",
            content=f"Session started. Analyzing image: {image_name}",
            parts=[MessagePart(type="text", content=f"Session started. Analyzing image: **{image_name}**")]
        )
        session.messages.append(system_msg)
        
        self.active_sessions[session.id] = session
        self._save_session(session)
        
        return session
    
    async def send_message(self, session_id: str, user_message: str,
                           search_image_path: str = None) -> AgentMessage:
        """
        Envía un mensaje al agente y obtiene la respuesta.
        Parsea la respuesta en partes estructuradas para renderizado rico.
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        if search_image_path and os.path.exists(search_image_path):
            session.image_path = search_image_path
            session.image_name = os.path.basename(search_image_path)
            user_message = f"[Image loaded from search: {session.image_name}]\n\n{user_message}"
        
        # Agregar mensaje del usuario
        user_msg = AgentMessage(
            role="user", 
            content=user_message,
            parts=[MessagePart(type="text", content=user_message)]
        )
        session.messages.append(user_msg)
        
        # Obtener configuración LLM
        if not self.config_loader:
            raise ValueError("config_loader is required")
        
        llm_config = self.config_loader(session.llm_config_name)
        if not llm_config:
            raise ValueError(f"LLM config not found: {session.llm_config_name}")
        
        # Construir historial de conversación para el LLM
        conversation_prompt = self._build_conversation_prompt(session, user_message)
        
        # Llamar al LLM
        provider = create_provider(llm_config)
        response = await provider.analyze_image(session.image_path, conversation_prompt)
        
        if not response.success:
            error_msg = AgentMessage(
                role="assistant",
                content=f"Error communicating with LLM: {response.error}",
                parts=[MessagePart(type="error", content=f"Error communicating with LLM: {response.error}")]
            )
            session.messages.append(error_msg)
            self._save_session(session)
            return error_msg
        
        # Procesar la respuesta - buscar bloques de código Python
        assistant_content = response.content
        code_blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)\n```', assistant_content, re.DOTALL)
        
        all_generated_images = []
        all_code_results = []
        
        if code_blocks:
            executor = CodeExecutor(session.output_dir)
            
            for i, code in enumerate(code_blocks):
                result = executor.execute(code, session.image_path)
                
                output_text = ""
                if result["stdout"]:
                    output_text += result["stdout"]
                if result["stderr"] and not result["success"]:
                    output_text += f"\nError:\n{result['stderr']}"
                
                all_code_results.append({
                    "code": code,
                    "output": output_text,
                    "success": result["success"],
                    "images": result["generated_images"]  # Solo filenames
                })
                
                all_generated_images.extend(result["generated_images"])
        
        # Parsear la respuesta en partes estructuradas
        parts = parse_response_to_parts(assistant_content, all_code_results, session.id)
        
        # Agregar imágenes que el LLM referencia con ![...](...) pero que no están en code_results
        # (imágenes referenciadas en el texto que ya fueron generadas)
        image_refs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', assistant_content)
        for caption, filename in image_refs:
            # Verificar si la imagen existe en el output_dir
            clean_filename = os.path.basename(filename)
            img_path = os.path.join(session.output_dir, clean_filename)
            if os.path.exists(img_path) and clean_filename not in all_generated_images:
                all_generated_images.append(clean_filename)
        
        # Crear mensaje de respuesta del asistente
        assistant_msg = AgentMessage(
            role="assistant",
            content=assistant_content,
            parts=parts,
            images=all_generated_images,  # Solo filenames
            code="\n\n".join(code_blocks) if code_blocks else "",
            code_output="\n---\n".join(
                [f"[Block {i+1}] {'OK' if co['success'] else 'ERROR'}\n{co['output']}" 
                 for i, co in enumerate(all_code_results)]
            ) if all_code_results else "",
            code_success=all(co["success"] for co in all_code_results) if all_code_results else True
        )
        
        session.messages.append(assistant_msg)
        self._save_session(session)
        
        return assistant_msg
    
    def _build_conversation_prompt(self, session: AgentSession, current_message: str) -> str:
        """Construye el prompt de conversación incluyendo historial"""
        parts = [SYSTEM_PROMPT]
        parts.append(f"\nCurrent image: {session.image_name}")
        parts.append(f"Image path: {session.image_path}")
        parts.append(f"Output directory: {session.output_dir}\n")
        
        # Incluir historial relevante (últimos N mensajes)
        recent_messages = session.messages[-20:]
        
        for msg in recent_messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                parts.append(f"\nUser: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"\nAssistant: {msg.content}")
                if msg.code_output:
                    parts.append(f"\n[Code execution output: {msg.code_output}]")
                if msg.images:
                    parts.append(f"\n[Generated images: {', '.join(msg.images)}]")
        
        parts.append(f"\nUser: {current_message}")
        parts.append("\nAssistant:")
        
        return "\n".join(parts)
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Obtiene una sesión por ID"""
        return self._get_session(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lista todas las sesiones"""
        sessions = []
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file, "r") as f:
                            data = json.load(f)
                        sessions.append({
                            "id": data["id"],
                            "title": data.get("title", ""),
                            "image_name": data.get("image_name", ""),
                            "llm_config_name": data.get("llm_config_name", ""),
                            "messages_count": len(data.get("messages", [])),
                            "created_at": data.get("created_at", "")
                        })
                    except Exception:
                        continue
        
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión"""
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            return True
        return False
    
    def _get_session(self, session_id: str) -> Optional[AgentSession]:
        """Obtiene una sesión del caché o disco"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        session_file = self.sessions_dir / session_id / "session.json"
        if session_file.exists():
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                session = AgentSession.from_dict(data)
                self.active_sessions[session_id] = session
                return session
            except Exception:
                return None
        return None
    
    def _save_session(self, session: AgentSession):
        """Guarda una sesión a disco"""
        session_dir = self.sessions_dir / session.id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / "session.json"
        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    def generate_session_report(self, session_id: str) -> Optional[str]:
        """
        Genera un reporte PDF firmado de la sesión de agente.
        Retorna el path del PDF generado.
        """
        session = self._get_session(session_id)
        if not session:
            return None
        
        try:
            from pdf_generator import generate_forensic_report
        except ImportError:
            return None
        
        # Construir datos de ejecución simulados para el generador de PDF
        step_results = []
        for i, msg in enumerate(session.messages):
            if msg.role in ("user", "assistant"):
                step_results.append({
                    "step_name": f"{'User Query' if msg.role == 'user' else 'Agent Response'} #{i}",
                    "step_order": i,
                    "prompt_used": msg.content if msg.role == "user" else "",
                    "content": msg.content if msg.role == "assistant" else msg.content,
                    "model_used": session.llm_config_name,
                    "provider_used": "agent",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "success": True,
                    "code": msg.code,
                    "code_output": msg.code_output,
                    "images": msg.images
                })
        
        execution_data = {
            "id": session.id,
            "pipeline_name": f"Agent Session: {session.title}",
            "image_path": session.image_path,
            "image_name": session.image_name,
            "step_results": step_results,
            "context_data": {},
            "started_at": session.created_at,
            "completed_at": datetime.now().isoformat(),
            "total_latency_ms": 0,
            "total_tokens": 0,
            "total_cost": 0
        }
        
        reports_dir = str(self.sessions_dir / session.id)
        output_path = os.path.join(reports_dir, f"agent_report_{session.id[:8]}.pdf")
        
        try:
            result_path = generate_forensic_report(execution_data, reports_dir)
            return result_path
        except Exception as e:
            return None


def ensure_image_packages():
    """Verifica e instala las librerías necesarias para procesamiento de imágenes"""
    missing = []
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python-headless")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    return missing
