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

# Máximo de reintentos automáticos cuando el código falla
MAX_CODE_RETRIES = 2

SYSTEM_PROMPT = """Eres un Agente Visual Forense avanzado especializado en análisis y procesamiento de imágenes.
Tienes acceso a un entorno de ejecución Python con las siguientes librerías pre-instaladas:
- PIL/Pillow (manipulación de imágenes)
- numpy (operaciones numéricas)  
- cv2/OpenCV (visión por computadora)
- matplotlib (gráficos y visualización)

IDIOMA: SIEMPRE debes responder en español. Todos los títulos, descripciones, análisis y comentarios deben estar en español.

FLUJO DE TRABAJO - DEBES seguir esta estructura en cada respuesta:

## 1. PLAN (Obligatorio)
Primero, presenta un plan breve de lo que vas a hacer. Usa una sección markdown como:

### Plan
1. [Primer artefacto/paso]
2. [Segundo artefacto/paso]
3. [Resultado esperado]

## 2. EJECUCIÓN
Escribe bloques de código Python para realizar las operaciones. Cada bloque de código será ejecutado automáticamente.

REGLAS CRÍTICAS PARA EL CÓDIGO:
- La ruta de la imagen de entrada está disponible como IMAGE_PATH (ya inyectada automáticamente)
- Guarda las imágenes de salida en OUTPUT_DIR (ya inyectado automáticamente)
- NO redefinas IMAGE_PATH ni OUTPUT_DIR en tu código, ya están disponibles como variables globales
- Usa nombres descriptivos para las imágenes de salida (ej: "vehiculo_recortado.png", "patente_ampliada.png")
- Siempre usa `plt.savefig()` en lugar de `plt.show()` para gráficos matplotlib
- Imprime hallazgos relevantes a stdout - se mostrarán al usuario
- Maneja errores con try/except
- SIEMPRE usa `os.path.join(OUTPUT_DIR, "nombre_archivo.png")` para guardar imágenes
- Cuando uses cv2.imread(), verifica que el resultado no sea None antes de procesarlo
- Cuando recortes imágenes, verifica las dimensiones antes de hacer el crop

## REGLA FUNDAMENTAL: NUNCA ADIVINES COORDENADAS

**PROHIBIDO** escribir coordenadas hardcodeadas como `img[195:255, 405:525]` para recortar regiones.
Tú NO puedes ver la imagen, por lo tanto NO sabes dónde están los objetos.

**SIEMPRE** debes usar técnicas de computer vision para DETECTAR automáticamente las regiones de interés.

### Técnica OBLIGATORIA para detectar y recortar patentes/matrículas:

USA SIEMPRE este enfoque multi-método con scoring. NUNCA uses un solo método.

```python
import cv2
import numpy as np
import os

img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Error: No se pudo cargar la imagen")
else:
    h, w = img.shape[:2]
    print(f"Imagen cargada: {w}x{h}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    all_candidates = []
    
    # === Método 1: Canny + contornos ===
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0 or cw == 0: continue
        ar = cw / ch
        area = cw * ch
        area_ratio = area / (w * h)
        if 2.0 < ar < 6.0 and 0.0005 < area_ratio < 0.02:
            all_candidates.append((x, y, cw, ch, area, "contornos"))
    
    # === Método 2: Morfología (blackhat + Sobel) ===
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gradX = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gradX = np.absolute(gradX)
    gradX = (255 * ((gradX - gradX.min()) / (gradX.max() - gradX.min() + 1e-6))).astype("uint8")
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0 or cw == 0: continue
        ar = cw / ch
        area = cw * ch
        area_ratio = area / (w * h)
        if 2.0 < ar < 6.0 and 0.0005 < area_ratio < 0.02:
            all_candidates.append((x, y, cw, ch, area, "morfología"))
    
    # === Método 3: Umbralización adaptativa ===
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel3)
    contours3, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours3:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0 or cw == 0: continue
        ar = cw / ch
        area = cw * ch
        area_ratio = area / (w * h)
        if 2.0 < ar < 6.0 and 0.0005 < area_ratio < 0.02:
            all_candidates.append((x, y, cw, ch, area, "adaptativa"))
    
    # === Scoring: puntuar cada candidato ===
    def score_plate(x, y, cw, ch, area, img_w, img_h):
        ar = cw / ch
        area_ratio = area / (img_w * img_h)
        # Aspect ratio ideal de patente ~3.0
        ar_score = max(0, 1.0 - abs(ar - 3.0) / 2.0)
        # Área ideal ~0.5% de la imagen
        area_score = max(0, 1.0 - abs(area_ratio - 0.005) / 0.005)
        # Altura razonable para patente
        height_ratio = ch / img_h
        height_score = 1.0 if 0.02 < height_ratio < 0.08 else 0.5
        # Posición: patentes suelen estar en el tercio medio-inferior
        y_center = (y + ch/2) / img_h
        pos_score = 1.0 if 0.15 < y_center < 0.6 else 0.3
        return ar_score * 0.35 + area_score * 0.30 + height_score * 0.15 + pos_score * 0.20
    
    scored = [(x, y, cw, ch, a, m, score_plate(x, y, cw, ch, a, w, h)) for x, y, cw, ch, a, m in all_candidates]
    scored.sort(key=lambda c: c[6], reverse=True)
    
    # Eliminar duplicados (NMS simple)
    final = []
    for cand in scored:
        is_dup = False
        for existing in final:
            # Calcular IoU
            ix1 = max(cand[0], existing[0])
            iy1 = max(cand[1], existing[1])
            ix2 = min(cand[0]+cand[2], existing[0]+existing[2])
            iy2 = min(cand[1]+cand[3], existing[1]+existing[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = cand[2]*cand[3] + existing[2]*existing[3] - inter
            if union > 0 and inter/union > 0.3:
                is_dup = True
                break
        if not is_dup:
            final.append(cand)
    
    print(f"Candidatos totales: {len(all_candidates)}, Después de NMS: {len(final)}")
    
    # Dibujar debug con todos los candidatos
    debug = img.copy()
    for i, (x, y, cw, ch, a, m, s) in enumerate(final[:5]):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(debug, (x, y), (x+cw, y+ch), color, 2)
        cv2.putText(debug, f"#{i} s={s:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"  #{i}: x={x}, y={y}, w={cw}, h={ch}, ar={cw/ch:.2f}, score={s:.3f}, método={m}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_deteccion.png"), debug)
    
    if final:
        best = final[0]
        x, y, cw, ch = best[:4]
        margin = 15
        patente = img[max(0,y-margin):min(h,y+ch+margin), max(0,x-margin):min(w,x+cw+margin)]
        cv2.imwrite(os.path.join(OUTPUT_DIR, "patente_recortada.png"), patente)
        print(f"Mejor candidato: x={x}, y={y}, w={cw}, h={ch}, score={best[6]:.3f}")
    else:
        print("No se detectó ninguna patente con ningún método.")
```

### Técnica para recortar vehículos u objetos grandes:

```python
import cv2
import numpy as np
import os

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# Usar GrabCut para segmentar el objeto principal
# Inicializar con un rectángulo que cubra la región central
margin_x = int(w * 0.05)
margin_y = int(h * 0.05)
rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

mask = np.zeros(img.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Encontrar bounding box del objeto segmentado
coords = cv2.findNonZero(mask2)
if coords is not None:
    x, y, cw, ch = cv2.boundingRect(coords)
    vehiculo = img[y:y+ch, x:x+cw]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "vehiculo_recortado.png"), vehiculo)
    print(f"Vehículo segmentado: x={x}, y={y}, w={cw}, h={ch}")
```

### Técnica para mejorar una patente recortada:

```python
import cv2
import numpy as np
import os

# Cargar la patente ya recortada
patente = cv2.imread(os.path.join(OUTPUT_DIR, "patente_recortada.png"))
if patente is not None:
    # Ampliar 3x con interpolación cúbica
    scale = 3
    ampliada = cv2.resize(patente, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(ampliada, cv2.COLOR_BGR2GRAY)
    
    # CLAHE para mejorar contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Umbralización adaptativa para texto
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "patente_mejorada.png"), sharpened)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "patente_binaria.png"), binary)
    print("Patente mejorada y binarizada guardadas")
```

## 3. RESULTADOS
Después de la ejecución del código, presenta los hallazgos usando formato Markdown enriquecido:
- Usa **negrita** para énfasis
- Usa tablas para datos estructurados
- Usa encabezados (##, ###) para organizar secciones
- Referencia las imágenes generadas usando: ![descripción](nombre_archivo.png)

RECUERDA: 
- NUNCA hardcodees coordenadas. SIEMPRE detecta con computer vision.
- Si un método de detección no funciona, intenta otro (contornos, morfología, umbralización, etc.)
- Siempre imprime las coordenadas detectadas para que el usuario pueda verificar.
- Si no puedes detectar automáticamente, explica por qué y sugiere alternativas.

Operaciones disponibles:
- RECORTAR: Extraer regiones específicas usando DETECCIÓN automática
- AMPLIAR: Magnificar áreas para inspección detallada
- MEJORAR: Mejorar contraste, brillo, nitidez con CLAHE, denoising, sharpening
- ANOTAR: Agregar cuadros delimitadores, etiquetas, flechas
- MEDIR: Calcular dimensiones, distancias, ángulos
- COMPARAR: Comparación lado a lado de regiones
- FILTRAR: Aplicar varios filtros de imagen
- DETECTAR: Detección de bordes, búsqueda de contornos, segmentación
- ANÁLISIS DE COLOR: Extraer colores dominantes, histogramas
- PREPARACIÓN OCR: Pre-procesar para reconocimiento de texto
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
        
        # Listar imágenes antes de ejecutar
        images_before = set()
        if os.path.exists(self.output_dir):
            images_before = {
                f for f in os.listdir(self.output_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))
                and not f.startswith('_agent_script')
            }
        
        # Ejecutar
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.output_dir
            )
            
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            success = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "Error: El código excedió el tiempo límite de 120 segundos"
            success = False
        except Exception as e:
            stdout = ""
            stderr = f"Error al ejecutar: {str(e)}"
            success = False
        
        # Detectar nuevas imágenes generadas
        images_after = set()
        if os.path.exists(self.output_dir):
            images_after = {
                f for f in os.listdir(self.output_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))
                and not f.startswith('_agent_script')
            }
        
        new_images = sorted(images_after - images_before)
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": success,
            "generated_images": new_images  # Solo filenames
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
    """Parsea un segmento de texto markdown en partes, extrayendo imágenes inline."""
    parts = []
    
    # Detectar si contiene una sección de plan
    plan_match = re.search(r'(#{1,3}\s*(?:Plan|Planificación|Planning)\s*\n)(.*?)(?=\n#{1,3}\s|\Z)', 
                           text, re.DOTALL | re.IGNORECASE)
    
    if plan_match:
        # Texto antes del plan
        before = text[:plan_match.start()].strip()
        if before:
            _extract_images_and_text(before, parts)
        
        # El plan
        plan_content = (plan_match.group(1) + plan_match.group(2)).strip()
        parts.append(MessagePart(type="plan", content=plan_content))
        
        # Texto después del plan
        after = text[plan_match.end():].strip()
        if after:
            _extract_images_and_text(after, parts)
    else:
        # Sin plan detectado
        if text.strip():
            _extract_images_and_text(text.strip(), parts)
    
    return parts


def _extract_images_and_text(text: str, parts: List[MessagePart]):
    """
    Extrae referencias a imágenes ![caption](filename) del texto markdown
    y las convierte en partes de tipo 'image' separadas, dejando el texto restante
    como partes de tipo 'text'.
    """
    # Pattern para imágenes markdown: ![caption](filename)
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    
    last_end = 0
    for match in img_pattern.finditer(text):
        # Texto antes de la imagen
        before_text = text[last_end:match.start()].strip()
        if before_text:
            parts.append(MessagePart(type="text", content=before_text))
        
        # La imagen
        caption = match.group(1)
        filename = match.group(2)
        # Limpiar el filename - solo quedarnos con el nombre del archivo
        clean_filename = os.path.basename(filename)
        parts.append(MessagePart(
            type="image",
            filename=clean_filename,
            caption=caption or clean_filename.replace("_", " ").replace(".png", "").replace(".jpg", "").title()
        ))
        
        last_end = match.end()
    
    # Texto restante después de la última imagen
    remaining = text[last_end:].strip()
    if remaining:
        parts.append(MessagePart(type="text", content=remaining))


class VisualAgent:
    """Agente visual interactivo para análisis de imágenes"""
    
    def __init__(self, base_dir: str = None, config_loader=None):
        if base_dir is None:
            base_dir = os.path.expanduser("~/.vision_llm_comparator")
        
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "agent_sessions"
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
            title=title or f"Análisis de {image_name}"
        )
        
        # Crear directorio de salida para esta sesión
        session.output_dir = str(self.sessions_dir / session.id / "outputs")
        os.makedirs(session.output_dir, exist_ok=True)
        
        # Mensaje de sistema
        system_msg = AgentMessage(
            role="system",
            content=f"Sesión iniciada. Analizando imagen: {image_name}",
            parts=[MessagePart(type="text", content=f"Sesión iniciada. Analizando imagen: **{image_name}**")]
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
        Implementa auto-retry cuando el código falla.
        """
        session = self._get_session(session_id)
        if not session:
            raise ValueError(f"Sesión no encontrada: {session_id}")
        
        if search_image_path and os.path.exists(search_image_path):
            session.image_path = search_image_path
            session.image_name = os.path.basename(search_image_path)
            user_message = f"[Imagen cargada desde búsqueda: {session.image_name}]\n\n{user_message}"
        
        # Agregar mensaje del usuario
        user_msg = AgentMessage(
            role="user", 
            content=user_message,
            parts=[MessagePart(type="text", content=user_message)]
        )
        session.messages.append(user_msg)
        
        # Obtener configuración LLM
        if not self.config_loader:
            raise ValueError("config_loader es requerido")
        
        llm_config = self.config_loader(session.llm_config_name)
        if not llm_config:
            raise ValueError(f"Configuración LLM no encontrada: {session.llm_config_name}")
        
        # Ejecutar con auto-retry
        assistant_msg = await self._execute_with_retry(session, llm_config, user_message)
        
        session.messages.append(assistant_msg)
        self._save_session(session)
        
        return assistant_msg
    
    async def _execute_with_retry(self, session: AgentSession, llm_config, 
                                   user_message: str, retry_count: int = 0) -> AgentMessage:
        """
        Ejecuta la interacción con el LLM y reintenta automáticamente si el código falla.
        Acumula todas las partes (intentos fallidos + intento exitoso) en un solo mensaje.
        """
        # Construir prompt de conversación
        conversation_prompt = self._build_conversation_prompt(session, user_message)
        
        # Si es un reintento, agregar contexto del error
        if retry_count > 0:
            conversation_prompt += f"\n\n[SISTEMA: El código anterior falló. Por favor corrige el error y vuelve a intentar. Intento {retry_count + 1} de {MAX_CODE_RETRIES + 1}.]"
        
        # Llamar al LLM
        provider = create_provider(llm_config)
        response = await provider.analyze_image(session.image_path, conversation_prompt)
        
        if not response.success:
            return AgentMessage(
                role="assistant",
                content=f"Error al comunicarse con el LLM: {response.error}",
                parts=[MessagePart(type="error", content=f"Error al comunicarse con el LLM: {response.error}")]
            )
        
        # Procesar la respuesta - buscar bloques de código Python
        assistant_content = response.content
        code_blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)\n```', assistant_content, re.DOTALL)
        
        all_generated_images = []
        all_code_results = []
        has_code_error = False
        error_details = ""
        
        if code_blocks:
            executor = CodeExecutor(session.output_dir)
            
            for i, code in enumerate(code_blocks):
                result = executor.execute(code, session.image_path)
                
                output_text = ""
                if result["stdout"]:
                    output_text += result["stdout"]
                if result["stderr"] and not result["success"]:
                    if output_text:
                        output_text += "\n"
                    output_text += f"Error:\n{result['stderr']}"
                    has_code_error = True
                    error_details = result["stderr"]
                
                all_code_results.append({
                    "code": code,
                    "output": output_text,
                    "success": result["success"],
                    "images": result["generated_images"]
                })
                
                all_generated_images.extend(result["generated_images"])
        
        # Si hay error en el código y no hemos agotado los reintentos, reintentar
        if has_code_error and retry_count < MAX_CODE_RETRIES:
            # Construir partes del intento fallido
            failed_parts = parse_response_to_parts(assistant_content, all_code_results, session.id)
            
            # Agregar una nota de reintento
            retry_note = MessagePart(
                type="text",
                content=f"\n\n---\n**Reintentando automáticamente** (intento {retry_count + 2}/{MAX_CODE_RETRIES + 1})...\n\n---\n"
            )
            
            # Agregar el intento fallido al historial de la sesión como contexto
            # para que el LLM sepa qué falló
            error_context_msg = AgentMessage(
                role="assistant",
                content=assistant_content,
                parts=failed_parts,
                images=all_generated_images,
                code="\n\n".join(code_blocks),
                code_output=error_details,
                code_success=False
            )
            session.messages.append(error_context_msg)
            
            # Crear mensaje de sistema con el error para el retry
            error_feedback = f"[SISTEMA: El código falló con el siguiente error:\n{error_details}\n\nPor favor analiza el error, corrige el código y vuelve a intentar. NO redefinas IMAGE_PATH ni OUTPUT_DIR, ya están disponibles. RECUERDA: NUNCA uses coordenadas hardcodeadas, siempre detecta con computer vision.]"
            error_msg = AgentMessage(
                role="user",
                content=error_feedback,
                parts=[MessagePart(type="text", content=error_feedback)]
            )
            session.messages.append(error_msg)
            
            # Reintentar
            retry_result = await self._execute_with_retry(
                session, llm_config, error_feedback, retry_count + 1
            )
            
            return retry_result
        
        # Parsear la respuesta en partes estructuradas
        parts = parse_response_to_parts(assistant_content, all_code_results, session.id)
        
        # Recopilar todas las imágenes referenciadas en el markdown
        image_refs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', assistant_content)
        for caption, filename in image_refs:
            clean_filename = os.path.basename(filename)
            img_path = os.path.join(session.output_dir, clean_filename)
            if os.path.exists(img_path) and clean_filename not in all_generated_images:
                all_generated_images.append(clean_filename)
        
        # Crear mensaje de respuesta del asistente
        assistant_msg = AgentMessage(
            role="assistant",
            content=assistant_content,
            parts=parts,
            images=all_generated_images,
            code="\n\n".join(code_blocks) if code_blocks else "",
            code_output="\n---\n".join(
                [f"[Bloque {i+1}] {'OK' if co['success'] else 'ERROR'}\n{co['output']}" 
                 for i, co in enumerate(all_code_results)]
            ) if all_code_results else "",
            code_success=all(co["success"] for co in all_code_results) if all_code_results else True
        )
        
        return assistant_msg
    
    def _build_conversation_prompt(self, session: AgentSession, current_message: str) -> str:
        """Construye el prompt de conversación incluyendo historial"""
        parts = [SYSTEM_PROMPT]
        parts.append(f"\nImagen actual: {session.image_name}")
        parts.append(f"Ruta de imagen: {session.image_path}")
        parts.append(f"Directorio de salida: {session.output_dir}\n")
        
        # Incluir historial relevante (últimos N mensajes)
        recent_messages = session.messages[-20:]
        
        for msg in recent_messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                parts.append(f"\nUsuario: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"\nAsistente: {msg.content}")
                if msg.code_output:
                    parts.append(f"\n[Salida de ejecución de código: {msg.code_output}]")
                if msg.images:
                    parts.append(f"\n[Imágenes generadas: {', '.join(msg.images)}]")
        
        parts.append(f"\nUsuario: {current_message}")
        parts.append("\nAsistente:")
        
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
                    "step_name": f"{'Consulta del Usuario' if msg.role == 'user' else 'Respuesta del Agente'} #{i}",
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
            "pipeline_name": f"Sesión de Agente: {session.title}",
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
