# Vision LLM Comparator

Herramienta para comparar LLMs visuales (modelos de lenguaje con capacidad de visión) mediante pipelines configurables. Permite evaluar y comparar el rendimiento de diferentes modelos procesando imágenes con múltiples pasos de análisis.

## Características

- **Soporte Multi-Proveedor**: OpenAI, Anthropic, Google Gemini, Ollama (modelos locales), y cualquier API compatible con OpenAI
- **Pipelines Configurables**: Crea secuencias de prompts que se ejecutan en cadena, usando la salida anterior como contexto
- **Comparación de Rendimiento**: Ejecuta múltiples pipelines sobre la misma imagen y compara métricas
- **Métricas Detalladas**: Latencia, tokens consumidos, costo estimado por paso y total
- **Historial Completo**: Guarda todas las ejecuciones para análisis posterior
- **Interfaz Web Intuitiva**: UI moderna con Bulma CSS para gestionar todo desde el navegador

## Instalación

### Opción 1: Docker (Recomendado)

La forma más sencilla de ejecutar Vision LLM Comparator es usando Docker.

#### Requisitos Previos
- [Docker](https://docs.docker.com/get-docker/) instalado
- [Docker Compose](https://docs.docker.com/compose/install/) instalado

#### Inicio Rápido con Docker

```bash
# Clonar el repositorio
git clone https://github.com/vtomasv/vision-arena.git
cd vision-arena

# Construir y ejecutar
docker compose up -d

# Ver logs (opcional)
docker compose logs -f
```

La aplicación estará disponible en **http://localhost:8000**

#### Configuración con Variables de Entorno

Puedes configurar las API keys mediante variables de entorno:

```bash
# Opción 1: Exportar variables antes de ejecutar
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
docker compose up -d

# Opción 2: Crear archivo .env
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF
docker compose up -d
```

#### Comandos Docker Útiles

```bash
# Iniciar en segundo plano
docker compose up -d

# Ver estado del contenedor
docker compose ps

# Ver logs en tiempo real
docker compose logs -f

# Detener la aplicación
docker compose down

# Reconstruir después de cambios
docker compose build --no-cache
docker compose up -d

# Eliminar todo (incluyendo datos)
docker compose down -v
```

#### Persistencia de Datos

Los datos se almacenan en un volumen Docker llamado `vision_data`. Esto incluye:
- Configuraciones de LLM
- Definiciones de pipelines
- Historial de ejecuciones
- Imágenes subidas

Para hacer backup de los datos:
```bash
# Crear backup
docker run --rm -v vision-arena_vision_data:/data -v $(pwd):/backup alpine tar czf /backup/vision-data-backup.tar.gz -C /data .

# Restaurar backup
docker run --rm -v vision-arena_vision_data:/data -v $(pwd):/backup alpine tar xzf /backup/vision-data-backup.tar.gz -C /data
```

### Opción 2: Instalación Local (Python)

```bash
# Clonar el repositorio
git clone https://github.com/vtomasv/vision-arena.git
cd vision-arena

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

La aplicación estará disponible en `http://localhost:8000`

## Estructura del Proyecto

```
vision-arena/
├── app.py              # Aplicación FastAPI con UI
├── llm_providers.py    # Proveedores de LLM (OpenAI, Anthropic, etc.)
├── pipeline.py         # Sistema de pipelines y comparación
├── storage.py          # Persistencia de datos
├── requirements.txt    # Dependencias Python
├── Dockerfile          # Imagen Docker
├── docker-compose.yml  # Configuración Docker Compose
└── README.md           # Esta documentación
```

## Uso

### 1. Configurar Proveedores LLM

Primero, configura los LLMs que deseas usar:

1. Ve a la pestaña **"Configuraciones LLM"**
2. Completa el formulario:
   - **Nombre**: Identificador único (ej: "GPT-4 Vision")
   - **Proveedor**: Selecciona el tipo (OpenAI, Anthropic, etc.)
   - **Modelo**: Nombre del modelo (ej: "gpt-4o", "claude-3-5-sonnet-20241022")
   - **API Key**: Tu clave de API
   - **API Base URL**: (Opcional) Para APIs compatibles o locales
3. Guarda la configuración

#### Ejemplos de Configuración

**OpenAI GPT-4 Vision:**
```
Proveedor: openai
Modelo: gpt-4o
API Key: sk-...
```

**Anthropic Claude:**
```
Proveedor: anthropic
Modelo: claude-3-5-sonnet-20241022
API Key: sk-ant-...
```

**Ollama (Local):**
```
Proveedor: ollama
Modelo: llava
API Base URL: http://localhost:11434
```

**OpenAI Compatible (LM Studio, vLLM, etc.):**
```
Proveedor: openai-compatible
Modelo: local-model
API Base URL: http://localhost:1234/v1
```

### 2. Crear Pipelines

Un pipeline es una secuencia de pasos de análisis:

1. Ve a la pestaña **"Pipelines"**
2. Completa:
   - **Nombre**: Nombre descriptivo
   - **Descripción**: Qué hace el pipeline
   - **Configuración LLM**: Selecciona un LLM configurado
3. Agrega pasos con el botón **"Agregar Paso"**:
   - **Nombre del paso**: Identificador del paso
   - **Prompt**: Instrucciones para el LLM
   - **Usar salida anterior**: Si incluir el resultado del paso previo como contexto
4. Guarda el pipeline

#### Ejemplo de Pipeline

**Pipeline: "Análisis Completo de Imagen"**

| Paso | Nombre | Prompt |
|------|--------|--------|
| 1 | Descripción General | "Describe detalladamente qué ves en esta imagen." |
| 2 | Identificar Objetos | "Lista todos los objetos visibles en la imagen con su ubicación aproximada." |
| 3 | Análisis de Contexto | "Basándote en la descripción anterior, ¿cuál es el contexto o situación de esta imagen?" |
| 4 | Resumen Estructurado | "Genera un JSON con: {objetos: [], contexto: '', sentimiento: ''}" |

### 3. Ejecutar Comparaciones

1. Ve a la pestaña **"Ejecutar"**
2. Sube una imagen
3. Selecciona los pipelines a comparar
4. Haz clic en **"Ejecutar Comparación"**
5. Revisa los resultados:
   - Tabla comparativa de métricas
   - Resultados detallados por paso
   - Identificación del pipeline más rápido y económico

### 4. Revisar Historial

La pestaña **"Historial"** muestra todas las ejecuciones pasadas con:
- Fecha y hora
- Pipeline utilizado
- Métricas de rendimiento
- Opción de ver detalles completos

### 5. Estadísticas

La pestaña **"Estadísticas"** ofrece:
- Total de ejecuciones
- Tasa de éxito
- Tokens totales consumidos
- Costo total estimado
- Estadísticas por pipeline

## API REST

La aplicación expone una API REST completa:

### Configuraciones LLM
- `POST /api/configs` - Crear configuración
- `GET /api/configs` - Listar configuraciones
- `DELETE /api/configs/{name}` - Eliminar configuración

### Pipelines
- `POST /api/pipelines` - Crear pipeline
- `GET /api/pipelines` - Listar pipelines
- `GET /api/pipelines/{id}` - Obtener pipeline
- `PUT /api/pipelines/{id}` - Actualizar pipeline
- `DELETE /api/pipelines/{id}` - Eliminar pipeline

### Imágenes
- `POST /api/images/upload` - Subir imagen
- `GET /api/images` - Listar imágenes
- `GET /api/images/view/{filename}` - Ver imagen

### Ejecución
- `POST /api/execute` - Ejecutar pipelines
- `GET /api/execute/{id}/status` - Estado de ejecución

### Historial
- `GET /api/history` - Listar historial
- `GET /api/history/{id}` - Detalles de ejecución
- `DELETE /api/history/{id}` - Eliminar ejecución
- `DELETE /api/history` - Limpiar historial

### Estadísticas
- `GET /api/statistics` - Obtener estadísticas

## Uso Programático

```python
import asyncio
from llm_providers import LLMConfig, create_provider
from pipeline import Pipeline, PipelineRunner

# Configurar LLM
config = LLMConfig(
    name="GPT-4 Vision",
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    max_tokens=4096,
    temperature=0.7
)

# Crear pipeline
pipeline = Pipeline(
    id="test-pipeline",
    name="Test Pipeline",
    description="Pipeline de prueba",
    llm_config=config
)

pipeline.add_step("Descripción", "Describe esta imagen en detalle.")
pipeline.add_step("Análisis", "Analiza los elementos principales.")

# Ejecutar
async def main():
    runner = PipelineRunner()
    result = await runner.run(pipeline, "imagen.jpg")
    
    print(f"Latencia total: {result.total_latency_ms}ms")
    print(f"Tokens totales: {result.total_tokens}")
    print(f"Costo estimado: ${result.total_cost}")
    
    for step in result.step_results:
        print(f"\n--- {step.step_name} ---")
        print(step.response.content)

asyncio.run(main())
```

## Almacenamiento

### Con Docker
Los datos se almacenan en el volumen `vision_data` montado en `/data` dentro del contenedor.

### Instalación Local
Los datos se guardan en `~/.vision_llm_comparator/`:
- `configs/` - Configuraciones de LLM
- `pipelines/` - Definiciones de pipelines
- `executions/` - Historial de ejecuciones (organizado por fecha)
- `images/` - Imágenes subidas

## Estimación de Costos

La aplicación estima costos basándose en precios públicos de los proveedores:

| Modelo | Input ($/1M tokens) | Output ($/1M tokens) |
|--------|---------------------|----------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-haiku | $0.25 | $1.25 |
| Ollama (local) | $0.00 | $0.00 |

## Solución de Problemas

### Docker

**El contenedor no inicia:**
```bash
# Ver logs detallados
docker compose logs vision-llm-comparator

# Verificar que el puerto 8000 no está en uso
lsof -i :8000
```

**Problemas de permisos con volúmenes:**
```bash
# En Linux, puede ser necesario ajustar permisos
sudo chown -R 1000:1000 /var/lib/docker/volumes/vision-arena_vision_data
```

**Reconstruir imagen después de cambios:**
```bash
docker compose build --no-cache
docker compose up -d
```

### Instalación Local

**Error de dependencias:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Puerto en uso:**
```bash
# Cambiar el puerto en app.py o usar variable de entorno
PORT=8080 python app.py
```

## Licencia

MIT License
