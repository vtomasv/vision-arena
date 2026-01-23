"""
Vision LLM Comparator - Aplicación Web con FastAPI
Interfaz gráfica para comparar LLMs visuales con pipelines configurables
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_providers import LLMConfig
from pipeline import Pipeline, PipelineStep, PipelineRunner, PipelineComparator
from storage import StorageManager

# Inicializar la aplicación
app = FastAPI(
    title="Vision LLM Comparator",
    description="Herramienta para comparar LLMs visuales con pipelines configurables",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
storage = StorageManager()

# Estado de ejecuciones en progreso
running_executions = {}


# ==================== Modelos Pydantic ====================

class LLMConfigCreate(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7


class PipelineStepCreate(BaseModel):
    name: str
    prompt: str
    use_previous_output: bool = True


class PipelineCreate(BaseModel):
    name: str
    description: str = ""
    llm_config_name: str
    steps: List[PipelineStepCreate]


class PipelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[PipelineStepCreate]] = None


class ExecutionRequest(BaseModel):
    pipeline_ids: List[str]
    image_path: str


# ==================== API Endpoints ====================

# --- LLM Configs ---

@app.post("/api/configs")
async def create_llm_config(config: LLMConfigCreate):
    """Crear una nueva configuración de LLM"""
    llm_config = LLMConfig(
        name=config.name,
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_tokens=config.max_tokens,
        temperature=config.temperature
    )
    storage.save_llm_config(llm_config)
    return {"status": "success", "message": f"Configuración '{config.name}' guardada"}


@app.get("/api/configs")
async def list_llm_configs():
    """Listar todas las configuraciones de LLM"""
    return storage.list_llm_configs()


@app.delete("/api/configs/{name}")
async def delete_llm_config(name: str):
    """Eliminar una configuración de LLM"""
    if storage.delete_llm_config(name):
        return {"status": "success", "message": f"Configuración '{name}' eliminada"}
    raise HTTPException(status_code=404, detail="Configuración no encontrada")


# --- Pipelines ---

@app.post("/api/pipelines")
async def create_pipeline(pipeline_data: PipelineCreate):
    """Crear un nuevo pipeline"""
    # Cargar la configuración de LLM
    llm_config = storage.load_llm_config(pipeline_data.llm_config_name)
    if not llm_config:
        raise HTTPException(status_code=404, detail=f"Configuración LLM '{pipeline_data.llm_config_name}' no encontrada")
    
    pipeline = Pipeline(
        id=str(uuid.uuid4()),
        name=pipeline_data.name,
        description=pipeline_data.description,
        llm_config=llm_config
    )
    
    for step in pipeline_data.steps:
        pipeline.add_step(
            name=step.name,
            prompt=step.prompt,
            use_previous_output=step.use_previous_output
        )
    
    storage.save_pipeline(pipeline)
    return {"status": "success", "pipeline_id": pipeline.id, "message": f"Pipeline '{pipeline.name}' creado"}


@app.get("/api/pipelines")
async def list_pipelines():
    """Listar todos los pipelines"""
    return storage.list_pipelines()


@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Obtener detalles de un pipeline"""
    pipeline = storage.load_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado")
    return pipeline.to_dict()


@app.put("/api/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, update_data: PipelineUpdate):
    """Actualizar un pipeline existente"""
    pipeline = storage.load_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado")
    
    if update_data.name:
        pipeline.name = update_data.name
    if update_data.description is not None:
        pipeline.description = update_data.description
    if update_data.steps is not None:
        pipeline.steps = []
        for step in update_data.steps:
            pipeline.add_step(
                name=step.name,
                prompt=step.prompt,
                use_previous_output=step.use_previous_output
            )
    
    storage.save_pipeline(pipeline)
    return {"status": "success", "message": f"Pipeline '{pipeline.name}' actualizado"}


@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Eliminar un pipeline"""
    if storage.delete_pipeline(pipeline_id):
        return {"status": "success", "message": "Pipeline eliminado"}
    raise HTTPException(status_code=404, detail="Pipeline no encontrado")


# --- Images ---

@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Subir una imagen"""
    # Validar tipo de archivo
    allowed_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
    
    # Guardar temporalmente
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Mover a directorio de imágenes
    saved_path = storage.save_image(temp_path, file.filename)
    os.remove(temp_path)
    
    return {"status": "success", "path": saved_path, "filename": file.filename}


@app.get("/api/images")
async def list_images():
    """Listar imágenes guardadas"""
    return storage.list_images()


@app.get("/api/images/view/{filename}")
async def view_image(filename: str):
    """Obtener una imagen para visualización"""
    image_path = storage.images_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(str(image_path))


# --- Executions ---

@app.post("/api/execute")
async def execute_pipelines(request: ExecutionRequest, background_tasks: BackgroundTasks):
    """Ejecutar uno o más pipelines sobre una imagen"""
    execution_id = str(uuid.uuid4())
    
    # Validar que la imagen existe
    if not Path(request.image_path).exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Cargar pipelines
    pipelines = []
    for pid in request.pipeline_ids:
        pipeline = storage.load_pipeline(pid)
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pid} no encontrado")
        pipelines.append(pipeline)
    
    # Iniciar ejecución en background
    running_executions[execution_id] = {
        "status": "running",
        "progress": "Iniciando...",
        "results": [],
        "started_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(
        run_pipelines_background,
        execution_id,
        pipelines,
        request.image_path
    )
    
    return {"status": "started", "execution_id": execution_id}


async def run_pipelines_background(execution_id: str, pipelines: List[Pipeline], image_path: str):
    """Ejecuta los pipelines en background"""
    comparator = PipelineComparator()
    
    def progress_callback(message: str):
        running_executions[execution_id]["progress"] = message
    
    try:
        results = await comparator.compare(pipelines, image_path, progress_callback)
        
        # Guardar resultados
        for result in results:
            storage.save_execution(result)
        
        # Generar reporte comparativo
        report = PipelineComparator.generate_comparison_report(results)
        
        running_executions[execution_id] = {
            "status": "completed",
            "progress": "Completado",
            "results": [r.to_dict() for r in results],
            "report": report,
            "completed_at": datetime.now().isoformat()
        }
    except Exception as e:
        running_executions[execution_id] = {
            "status": "error",
            "progress": f"Error: {str(e)}",
            "error": str(e)
        }


@app.get("/api/execute/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """Obtener el estado de una ejecución"""
    if execution_id not in running_executions:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return running_executions[execution_id]


# --- History ---

@app.get("/api/history")
async def get_history(limit: int = 50, pipeline_id: str = None):
    """Obtener historial de ejecuciones"""
    return storage.list_executions(limit=limit, pipeline_id=pipeline_id)


@app.get("/api/history/{execution_id}")
async def get_execution_details(execution_id: str):
    """Obtener detalles completos de una ejecución"""
    details = storage.get_execution_details(execution_id)
    if not details:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return details


@app.delete("/api/history/{execution_id}")
async def delete_execution(execution_id: str):
    """Eliminar una ejecución del historial"""
    if storage.delete_execution(execution_id):
        return {"status": "success", "message": "Ejecución eliminada"}
    raise HTTPException(status_code=404, detail="Ejecución no encontrada")


@app.delete("/api/history")
async def clear_history(before_date: str = None):
    """Limpiar historial de ejecuciones"""
    count = storage.clear_history(before_date)
    return {"status": "success", "deleted_count": count}


# --- Statistics ---

@app.get("/api/statistics")
async def get_statistics():
    """Obtener estadísticas de uso"""
    return storage.get_statistics()


# ==================== Frontend HTML ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision LLM Comparator</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #3273dc;
            --success-color: #48c774;
            --danger-color: #f14668;
            --warning-color: #ffdd57;
        }
        
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .main-container {
            padding: 2rem;
        }
        
        .card {
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        .navbar {
            background: rgba(255,255,255,0.95);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .navbar-brand .navbar-item {
            font-weight: bold;
            font-size: 1.3rem;
            color: var(--primary-color);
        }
        
        .tabs-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            object-fit: contain;
        }
        
        .step-card {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid var(--primary-color);
        }
        
        .result-card {
            border-left: 4px solid var(--success-color);
        }
        
        .result-card.error {
            border-left-color: var(--danger-color);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.85rem;
            opacity: 0.9;
        }
        
        .pipeline-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: var(--primary-color);
            color: white;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .loading-card {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            min-width: 300px;
        }
        
        .comparison-table th {
            background: #f5f5f5;
        }
        
        .comparison-table .best {
            background: #effaf3;
            font-weight: bold;
        }
        
        .prompt-textarea {
            min-height: 100px;
            font-family: monospace;
        }
        
        .hidden {
            display: none !important;
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .response-content {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9rem;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .config-form .field {
            margin-bottom: 1rem;
        }
        
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            background: var(--primary-color);
            color: white;
            border-radius: 50%;
            font-weight: bold;
            margin-right: 0.5rem;
        }
    </style>
</head>
<body>
    <nav class="navbar" role="navigation">
        <div class="navbar-brand">
            <a class="navbar-item" href="#">
                <i class="fas fa-eye mr-2"></i> Vision LLM Comparator
            </a>
        </div>
        <div class="navbar-menu">
            <div class="navbar-end">
                <div class="navbar-item">
                    <span class="tag is-info" id="stats-badge">
                        <i class="fas fa-chart-bar mr-1"></i>
                        <span id="total-executions">0</span> ejecuciones
                    </span>
                </div>
            </div>
        </div>
    </nav>

    <div class="main-container">
        <div class="tabs-container">
            <div class="tabs is-centered is-boxed">
                <ul>
                    <li class="is-active" data-tab="execute">
                        <a><i class="fas fa-play mr-2"></i>Ejecutar</a>
                    </li>
                    <li data-tab="pipelines">
                        <a><i class="fas fa-project-diagram mr-2"></i>Pipelines</a>
                    </li>
                    <li data-tab="configs">
                        <a><i class="fas fa-cog mr-2"></i>Configuraciones LLM</a>
                    </li>
                    <li data-tab="history">
                        <a><i class="fas fa-history mr-2"></i>Historial</a>
                    </li>
                    <li data-tab="stats">
                        <a><i class="fas fa-chart-line mr-2"></i>Estadísticas</a>
                    </li>
                </ul>
            </div>
        </div>

        <!-- Tab: Ejecutar -->
        <div id="tab-execute" class="tab-content">
            <div class="columns">
                <div class="column is-4">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-image mr-2"></i>Imagen
                            </p>
                        </div>
                        <div class="card-content">
                            <div class="file has-name is-fullwidth is-boxed">
                                <label class="file-label">
                                    <input class="file-input" type="file" id="image-upload" accept="image/*">
                                    <span class="file-cta">
                                        <span class="file-icon">
                                            <i class="fas fa-upload"></i>
                                        </span>
                                        <span class="file-label">Subir imagen...</span>
                                    </span>
                                    <span class="file-name" id="image-filename">Ningún archivo seleccionado</span>
                                </label>
                            </div>
                            <div class="mt-4 has-text-centered">
                                <img id="image-preview" class="image-preview hidden" alt="Preview">
                            </div>
                            <input type="hidden" id="current-image-path">
                        </div>
                    </div>
                    
                    <div class="card mt-4">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list-check mr-2"></i>Pipelines a ejecutar
                            </p>
                        </div>
                        <div class="card-content">
                            <div id="pipeline-checkboxes">
                                <p class="has-text-grey">Cargando pipelines...</p>
                            </div>
                            <button class="button is-primary is-fullwidth mt-4" id="btn-execute" disabled>
                                <i class="fas fa-play mr-2"></i>Ejecutar Comparación
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="column is-8">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-chart-bar mr-2"></i>Resultados de Comparación
                            </p>
                        </div>
                        <div class="card-content">
                            <div id="execution-results">
                                <div class="has-text-centered has-text-grey py-6">
                                    <i class="fas fa-info-circle fa-2x mb-3"></i>
                                    <p>Selecciona una imagen y pipelines para ejecutar la comparación</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Pipelines -->
        <div id="tab-pipelines" class="tab-content hidden">
            <div class="columns">
                <div class="column is-4">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-plus mr-2"></i>Crear Pipeline
                            </p>
                        </div>
                        <div class="card-content">
                            <div class="field">
                                <label class="label">Nombre</label>
                                <input class="input" type="text" id="pipeline-name" placeholder="Mi Pipeline">
                            </div>
                            <div class="field">
                                <label class="label">Descripción</label>
                                <textarea class="textarea" id="pipeline-description" placeholder="Descripción del pipeline..."></textarea>
                            </div>
                            <div class="field">
                                <label class="label">Configuración LLM</label>
                                <div class="select is-fullwidth">
                                    <select id="pipeline-llm-config">
                                        <option value="">Seleccionar...</option>
                                    </select>
                                </div>
                            </div>
                            
                            <hr>
                            
                            <div class="field">
                                <label class="label">Pasos del Pipeline</label>
                                <div id="pipeline-steps-editor">
                                    <!-- Steps dinámicos -->
                                </div>
                                <button class="button is-small is-info mt-2" id="btn-add-step">
                                    <i class="fas fa-plus mr-1"></i>Agregar Paso
                                </button>
                            </div>
                            
                            <button class="button is-primary is-fullwidth mt-4" id="btn-save-pipeline">
                                <i class="fas fa-save mr-2"></i>Guardar Pipeline
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="column is-8">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list mr-2"></i>Pipelines Guardados
                            </p>
                        </div>
                        <div class="card-content">
                            <div id="pipelines-list">
                                <p class="has-text-grey">Cargando...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Configuraciones LLM -->
        <div id="tab-configs" class="tab-content hidden">
            <div class="columns">
                <div class="column is-5">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-plus mr-2"></i>Nueva Configuración
                            </p>
                        </div>
                        <div class="card-content config-form">
                            <div class="field">
                                <label class="label">Nombre</label>
                                <input class="input" type="text" id="config-name" placeholder="GPT-4 Vision">
                            </div>
                            <div class="field">
                                <label class="label">Proveedor</label>
                                <div class="select is-fullwidth">
                                    <select id="config-provider">
                                        <option value="openai">OpenAI</option>
                                        <option value="openai-compatible">OpenAI Compatible</option>
                                        <option value="anthropic">Anthropic</option>
                                        <option value="google">Google</option>
                                        <option value="ollama">Ollama (Local)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="field">
                                <label class="label">Modelo</label>
                                <input class="input" type="text" id="config-model" placeholder="gpt-4o">
                            </div>
                            <div class="field">
                                <label class="label">API Key</label>
                                <input class="input" type="password" id="config-api-key" placeholder="sk-...">
                            </div>
                            <div class="field">
                                <label class="label">API Base URL (opcional)</label>
                                <input class="input" type="text" id="config-api-base" placeholder="https://api.openai.com/v1">
                            </div>
                            <div class="columns">
                                <div class="column">
                                    <div class="field">
                                        <label class="label">Max Tokens</label>
                                        <input class="input" type="number" id="config-max-tokens" value="4096">
                                    </div>
                                </div>
                                <div class="column">
                                    <div class="field">
                                        <label class="label">Temperature</label>
                                        <input class="input" type="number" id="config-temperature" value="0.7" step="0.1" min="0" max="2">
                                    </div>
                                </div>
                            </div>
                            <button class="button is-primary is-fullwidth" id="btn-save-config">
                                <i class="fas fa-save mr-2"></i>Guardar Configuración
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="column is-7">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list mr-2"></i>Configuraciones Guardadas
                            </p>
                        </div>
                        <div class="card-content">
                            <div id="configs-list">
                                <p class="has-text-grey">Cargando...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Historial -->
        <div id="tab-history" class="tab-content hidden">
            <div class="card">
                <div class="card-header">
                    <p class="card-header-title">
                        <i class="fas fa-history mr-2"></i>Historial de Ejecuciones
                    </p>
                    <button class="button is-small is-danger mr-4 mt-3" id="btn-clear-history">
                        <i class="fas fa-trash mr-1"></i>Limpiar
                    </button>
                </div>
                <div class="card-content">
                    <div id="history-list">
                        <p class="has-text-grey">Cargando...</p>
                    </div>
                </div>
            </div>
            
            <!-- Modal para detalles -->
            <div class="modal" id="execution-detail-modal">
                <div class="modal-background"></div>
                <div class="modal-card" style="width: 90%; max-width: 1200px;">
                    <header class="modal-card-head">
                        <p class="modal-card-title">Detalles de Ejecución</p>
                        <button class="delete" aria-label="close"></button>
                    </header>
                    <section class="modal-card-body" id="execution-detail-content">
                        <!-- Contenido dinámico -->
                    </section>
                </div>
            </div>
        </div>

        <!-- Tab: Estadísticas -->
        <div id="tab-stats" class="tab-content hidden">
            <div class="columns">
                <div class="column is-12">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-chart-pie mr-2"></i>Resumen General
                            </p>
                        </div>
                        <div class="card-content">
                            <div class="metrics-grid" id="stats-overview">
                                <!-- Métricas dinámicas -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="columns mt-4">
                <div class="column is-12">
                    <div class="card">
                        <div class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-project-diagram mr-2"></i>Estadísticas por Pipeline
                            </p>
                        </div>
                        <div class="card-content">
                            <div id="pipeline-stats">
                                <!-- Estadísticas por pipeline -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay hidden" id="loading-overlay">
        <div class="loading-card">
            <div class="mb-4">
                <i class="fas fa-spinner fa-spin fa-3x has-text-primary"></i>
            </div>
            <p class="is-size-5" id="loading-message">Procesando...</p>
            <progress class="progress is-primary mt-3" max="100" id="loading-progress"></progress>
        </div>
    </div>

    <script>
        // ==================== Estado Global ====================
        let currentImagePath = null;
        let selectedPipelines = new Set();
        let stepCounter = 0;

        // ==================== Utilidades ====================
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tabs li').forEach(el => el.classList.remove('is-active'));
            
            document.getElementById(`tab-${tabName}`).classList.remove('hidden');
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('is-active');
            
            // Cargar datos según la pestaña
            if (tabName === 'configs') loadConfigs();
            if (tabName === 'pipelines') { loadConfigs(); loadPipelines(); }
            if (tabName === 'execute') { loadPipelinesForExecution(); }
            if (tabName === 'history') loadHistory();
            if (tabName === 'stats') loadStatistics();
        }

        function showLoading(message = 'Procesando...') {
            document.getElementById('loading-message').textContent = message;
            document.getElementById('loading-overlay').classList.remove('hidden');
        }

        function hideLoading() {
            document.getElementById('loading-overlay').classList.add('hidden');
        }

        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification is-${type} fade-in`;
            notification.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 1001; max-width: 400px;';
            notification.innerHTML = `
                <button class="delete"></button>
                ${message}
            `;
            document.body.appendChild(notification);
            notification.querySelector('.delete').onclick = () => notification.remove();
            setTimeout(() => notification.remove(), 5000);
        }

        function formatNumber(num, decimals = 2) {
            if (num >= 1000000) return (num / 1000000).toFixed(decimals) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(decimals) + 'K';
            return num.toFixed(decimals);
        }

        function formatDuration(ms) {
            if (ms < 1000) return ms.toFixed(0) + ' ms';
            return (ms / 1000).toFixed(2) + ' s';
        }

        // ==================== API Calls ====================
        async function apiCall(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (data) options.body = JSON.stringify(data);
            
            const response = await fetch(`/api/${endpoint}`, options);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Error en la API');
            }
            return response.json();
        }

        // ==================== Configuraciones LLM ====================
        async function loadConfigs() {
            try {
                const configs = await apiCall('configs');
                renderConfigsList(configs);
                renderConfigsSelect(configs);
            } catch (e) {
                console.error('Error cargando configs:', e);
            }
        }

        function renderConfigsList(configs) {
            const container = document.getElementById('configs-list');
            if (configs.length === 0) {
                container.innerHTML = '<p class="has-text-grey">No hay configuraciones guardadas</p>';
                return;
            }
            
            container.innerHTML = configs.map(c => `
                <div class="step-card">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <strong>${c.name}</strong>
                                <br>
                                <span class="tag is-info is-light">${c.provider}</span>
                                <span class="tag is-primary is-light">${c.model}</span>
                                <br>
                                <small class="has-text-grey">API Key: ${c.api_key_preview || 'No configurada'}</small>
                            </div>
                        </div>
                        <div class="level-right">
                            <button class="button is-small is-danger" onclick="deleteConfig('${c.name}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function renderConfigsSelect(configs) {
            const select = document.getElementById('pipeline-llm-config');
            select.innerHTML = '<option value="">Seleccionar...</option>' +
                configs.map(c => `<option value="${c.name}">${c.name} (${c.model})</option>`).join('');
        }

        async function saveConfig() {
            const config = {
                name: document.getElementById('config-name').value,
                provider: document.getElementById('config-provider').value,
                model: document.getElementById('config-model').value,
                api_key: document.getElementById('config-api-key').value,
                api_base: document.getElementById('config-api-base').value,
                max_tokens: parseInt(document.getElementById('config-max-tokens').value),
                temperature: parseFloat(document.getElementById('config-temperature').value)
            };
            
            if (!config.name || !config.model) {
                showNotification('Nombre y modelo son requeridos', 'danger');
                return;
            }
            
            try {
                await apiCall('configs', 'POST', config);
                showNotification('Configuración guardada correctamente');
                loadConfigs();
                // Limpiar formulario
                document.getElementById('config-name').value = '';
                document.getElementById('config-api-key').value = '';
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        async function deleteConfig(name) {
            if (!confirm(`¿Eliminar configuración "${name}"?`)) return;
            try {
                await apiCall(`configs/${encodeURIComponent(name)}`, 'DELETE');
                showNotification('Configuración eliminada');
                loadConfigs();
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        // ==================== Pipelines ====================
        function addPipelineStep() {
            stepCounter++;
            const container = document.getElementById('pipeline-steps-editor');
            const stepDiv = document.createElement('div');
            stepDiv.className = 'step-card fade-in';
            stepDiv.id = `step-${stepCounter}`;
            stepDiv.innerHTML = `
                <div class="level mb-2">
                    <div class="level-left">
                        <span class="step-number">${container.children.length + 1}</span>
                        <input class="input is-small" type="text" placeholder="Nombre del paso" 
                               style="width: 200px;" data-field="name">
                    </div>
                    <div class="level-right">
                        <button class="delete is-small" onclick="removeStep('step-${stepCounter}')"></button>
                    </div>
                </div>
                <textarea class="textarea prompt-textarea" placeholder="Prompt para este paso..." 
                          data-field="prompt"></textarea>
                <label class="checkbox mt-2">
                    <input type="checkbox" checked data-field="use_previous">
                    Usar salida del paso anterior como contexto
                </label>
            `;
            container.appendChild(stepDiv);
        }

        function removeStep(stepId) {
            document.getElementById(stepId).remove();
            // Renumerar pasos
            const steps = document.querySelectorAll('#pipeline-steps-editor .step-card');
            steps.forEach((step, i) => {
                step.querySelector('.step-number').textContent = i + 1;
            });
        }

        function getStepsFromEditor() {
            const steps = [];
            document.querySelectorAll('#pipeline-steps-editor .step-card').forEach(stepDiv => {
                steps.push({
                    name: stepDiv.querySelector('[data-field="name"]').value,
                    prompt: stepDiv.querySelector('[data-field="prompt"]').value,
                    use_previous_output: stepDiv.querySelector('[data-field="use_previous"]').checked
                });
            });
            return steps;
        }

        async function savePipeline() {
            const pipeline = {
                name: document.getElementById('pipeline-name').value,
                description: document.getElementById('pipeline-description').value,
                llm_config_name: document.getElementById('pipeline-llm-config').value,
                steps: getStepsFromEditor()
            };
            
            if (!pipeline.name || !pipeline.llm_config_name || pipeline.steps.length === 0) {
                showNotification('Nombre, configuración LLM y al menos un paso son requeridos', 'danger');
                return;
            }
            
            try {
                await apiCall('pipelines', 'POST', pipeline);
                showNotification('Pipeline guardado correctamente');
                loadPipelines();
                // Limpiar formulario
                document.getElementById('pipeline-name').value = '';
                document.getElementById('pipeline-description').value = '';
                document.getElementById('pipeline-steps-editor').innerHTML = '';
                stepCounter = 0;
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        async function loadPipelines() {
            try {
                const pipelines = await apiCall('pipelines');
                renderPipelinesList(pipelines);
            } catch (e) {
                console.error('Error cargando pipelines:', e);
            }
        }

        function renderPipelinesList(pipelines) {
            const container = document.getElementById('pipelines-list');
            if (pipelines.length === 0) {
                container.innerHTML = '<p class="has-text-grey">No hay pipelines guardados</p>';
                return;
            }
            
            container.innerHTML = pipelines.map(p => `
                <div class="step-card">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <strong>${p.name}</strong>
                                <br>
                                <small class="has-text-grey">${p.description || 'Sin descripción'}</small>
                                <br>
                                <span class="tag is-primary is-light">${p.llm_model}</span>
                                <span class="tag is-info is-light">${p.steps_count} pasos</span>
                            </div>
                        </div>
                        <div class="level-right">
                            <button class="button is-small is-info mr-2" onclick="viewPipeline('${p.id}')">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="button is-small is-danger" onclick="deletePipeline('${p.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        async function viewPipeline(id) {
            try {
                const pipeline = await apiCall(`pipelines/${id}`);
                alert(JSON.stringify(pipeline, null, 2));
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        async function deletePipeline(id) {
            if (!confirm('¿Eliminar este pipeline?')) return;
            try {
                await apiCall(`pipelines/${id}`, 'DELETE');
                showNotification('Pipeline eliminado');
                loadPipelines();
                loadPipelinesForExecution();
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        // ==================== Ejecución ====================
        async function loadPipelinesForExecution() {
            try {
                const pipelines = await apiCall('pipelines');
                renderPipelineCheckboxes(pipelines);
            } catch (e) {
                console.error('Error:', e);
            }
        }

        function renderPipelineCheckboxes(pipelines) {
            const container = document.getElementById('pipeline-checkboxes');
            if (pipelines.length === 0) {
                container.innerHTML = '<p class="has-text-grey">No hay pipelines. Crea uno primero.</p>';
                return;
            }
            
            container.innerHTML = pipelines.map(p => `
                <label class="checkbox is-block mb-2">
                    <input type="checkbox" value="${p.id}" onchange="togglePipeline('${p.id}')">
                    <strong>${p.name}</strong>
                    <br>
                    <small class="has-text-grey ml-4">${p.llm_model} - ${p.steps_count} pasos</small>
                </label>
            `).join('');
        }

        function togglePipeline(id) {
            if (selectedPipelines.has(id)) {
                selectedPipelines.delete(id);
            } else {
                selectedPipelines.add(id);
            }
            updateExecuteButton();
        }

        function updateExecuteButton() {
            const btn = document.getElementById('btn-execute');
            btn.disabled = !currentImagePath || selectedPipelines.size === 0;
        }

        async function uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            showLoading('Subiendo imagen...');
            try {
                const response = await fetch('/api/images/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                currentImagePath = data.path;
                document.getElementById('current-image-path').value = data.path;
                document.getElementById('image-filename').textContent = file.name;
                
                // Mostrar preview
                const preview = document.getElementById('image-preview');
                preview.src = `/api/images/view/${file.name}`;
                preview.classList.remove('hidden');
                
                updateExecuteButton();
                showNotification('Imagen subida correctamente');
            } catch (e) {
                showNotification('Error subiendo imagen: ' + e.message, 'danger');
            } finally {
                hideLoading();
            }
        }

        async function executeComparison() {
            if (!currentImagePath || selectedPipelines.size === 0) return;
            
            showLoading('Iniciando ejecución...');
            
            try {
                const response = await apiCall('execute', 'POST', {
                    pipeline_ids: Array.from(selectedPipelines),
                    image_path: currentImagePath
                });
                
                const executionId = response.execution_id;
                
                // Polling para estado
                const checkStatus = async () => {
                    const status = await apiCall(`execute/${executionId}/status`);
                    document.getElementById('loading-message').textContent = status.progress;
                    
                    if (status.status === 'completed') {
                        hideLoading();
                        renderExecutionResults(status);
                        showNotification('Ejecución completada');
                    } else if (status.status === 'error') {
                        hideLoading();
                        showNotification('Error: ' + status.error, 'danger');
                    } else {
                        setTimeout(checkStatus, 1000);
                    }
                };
                
                checkStatus();
            } catch (e) {
                hideLoading();
                showNotification('Error: ' + e.message, 'danger');
            }
        }

        function renderExecutionResults(status) {
            const container = document.getElementById('execution-results');
            const results = status.results;
            const report = status.report;
            
            if (!results || results.length === 0) {
                container.innerHTML = '<p class="has-text-grey">No hay resultados</p>';
                return;
            }
            
            // Tabla comparativa
            let html = `
                <h4 class="title is-5 mb-4">Comparación de Rendimiento</h4>
                <table class="table is-fullwidth comparison-table">
                    <thead>
                        <tr>
                            <th>Pipeline</th>
                            <th>Latencia</th>
                            <th>Tokens</th>
                            <th>Costo Est.</th>
                            <th>Estado</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            results.forEach(r => {
                const isFastest = report.fastest === r.pipeline_name;
                const isCheapest = report.cheapest === r.pipeline_name;
                
                html += `
                    <tr>
                        <td><strong>${r.pipeline_name}</strong></td>
                        <td class="${isFastest ? 'best' : ''}">${formatDuration(r.total_latency_ms)}</td>
                        <td>${formatNumber(r.total_tokens, 0)}</td>
                        <td class="${isCheapest ? 'best' : ''}">$${r.total_cost.toFixed(6)}</td>
                        <td>
                            ${r.success 
                                ? '<span class="tag is-success">Éxito</span>' 
                                : '<span class="tag is-danger">Error</span>'}
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            
            // Leyenda
            if (report.fastest || report.cheapest) {
                html += '<p class="is-size-7 has-text-grey mt-2">';
                if (report.fastest) html += `<span class="tag is-success is-light mr-2">Más rápido: ${report.fastest}</span>`;
                if (report.cheapest) html += `<span class="tag is-info is-light">Más económico: ${report.cheapest}</span>`;
                html += '</p>';
            }
            
            // Resultados detallados
            html += '<hr><h4 class="title is-5 mb-4">Resultados Detallados</h4>';
            
            results.forEach(r => {
                html += `
                    <div class="step-card result-card ${r.success ? '' : 'error'} mb-4">
                        <h5 class="title is-6">${r.pipeline_name}</h5>
                `;
                
                r.step_results.forEach((step, i) => {
                    html += `
                        <div class="mb-3">
                            <p><span class="step-number">${i + 1}</span><strong>${step.step_name}</strong></p>
                            <div class="tags mt-1">
                                <span class="tag is-light">${formatDuration(step.latency_ms)}</span>
                                <span class="tag is-light">${step.total_tokens} tokens</span>
                            </div>
                            <div class="response-content mt-2">${step.content || step.error || 'Sin respuesta'}</div>
                        </div>
                    `;
                });
                
                html += '</div>';
            });
            
            container.innerHTML = html;
        }

        // ==================== Historial ====================
        async function loadHistory() {
            try {
                const history = await apiCall('history?limit=50');
                renderHistory(history);
            } catch (e) {
                console.error('Error:', e);
            }
        }

        function renderHistory(history) {
            const container = document.getElementById('history-list');
            if (history.length === 0) {
                container.innerHTML = '<p class="has-text-grey">No hay ejecuciones en el historial</p>';
                return;
            }
            
            container.innerHTML = `
                <table class="table is-fullwidth is-hoverable">
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>Pipeline</th>
                            <th>Latencia</th>
                            <th>Tokens</th>
                            <th>Costo</th>
                            <th>Estado</th>
                            <th>Acciones</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${history.map(h => `
                            <tr>
                                <td>${new Date(h.started_at).toLocaleString()}</td>
                                <td><strong>${h.pipeline_name}</strong></td>
                                <td>${formatDuration(h.total_latency_ms)}</td>
                                <td>${formatNumber(h.total_tokens, 0)}</td>
                                <td>$${h.total_cost.toFixed(6)}</td>
                                <td>
                                    ${h.success 
                                        ? '<span class="tag is-success is-light">Éxito</span>' 
                                        : '<span class="tag is-danger is-light">Error</span>'}
                                </td>
                                <td>
                                    <button class="button is-small is-info" onclick="viewExecutionDetails('${h.id}')">
                                        <i class="fas fa-eye"></i>
                                    </button>
                                    <button class="button is-small is-danger" onclick="deleteExecution('${h.id}')">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        async function viewExecutionDetails(id) {
            try {
                const details = await apiCall(`history/${id}`);
                const modal = document.getElementById('execution-detail-modal');
                const content = document.getElementById('execution-detail-content');
                
                let html = `
                    <div class="columns">
                        <div class="column is-4">
                            <p><strong>Pipeline:</strong> ${details.pipeline_name}</p>
                            <p><strong>Fecha:</strong> ${new Date(details.started_at).toLocaleString()}</p>
                            <p><strong>Duración:</strong> ${formatDuration(details.total_latency_ms)}</p>
                            <p><strong>Tokens totales:</strong> ${formatNumber(details.total_tokens, 0)}</p>
                            <p><strong>Costo estimado:</strong> $${details.total_cost.toFixed(6)}</p>
                        </div>
                        <div class="column is-8">
                            <h5 class="title is-6">Resultados por Paso</h5>
                `;
                
                details.step_results.forEach((step, i) => {
                    html += `
                        <div class="step-card mb-3">
                            <p><span class="step-number">${i + 1}</span><strong>${step.step_name}</strong></p>
                            <div class="tags mt-1">
                                <span class="tag is-light">${formatDuration(step.latency_ms)}</span>
                                <span class="tag is-light">${step.total_tokens} tokens</span>
                                <span class="tag is-light">$${step.cost_estimate.toFixed(6)}</span>
                            </div>
                            <div class="response-content mt-2">${step.content || step.error || 'Sin respuesta'}</div>
                        </div>
                    `;
                });
                
                html += '</div></div>';
                content.innerHTML = html;
                modal.classList.add('is-active');
            } catch (e) {
                showNotification('Error cargando detalles: ' + e.message, 'danger');
            }
        }

        async function deleteExecution(id) {
            if (!confirm('¿Eliminar esta ejecución del historial?')) return;
            try {
                await apiCall(`history/${id}`, 'DELETE');
                showNotification('Ejecución eliminada');
                loadHistory();
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        async function clearHistory() {
            if (!confirm('¿Eliminar todo el historial?')) return;
            try {
                await apiCall('history', 'DELETE');
                showNotification('Historial limpiado');
                loadHistory();
            } catch (e) {
                showNotification(e.message, 'danger');
            }
        }

        // ==================== Estadísticas ====================
        async function loadStatistics() {
            try {
                const stats = await apiCall('statistics');
                renderStatistics(stats);
            } catch (e) {
                console.error('Error:', e);
            }
        }

        function renderStatistics(stats) {
            const overview = document.getElementById('stats-overview');
            overview.innerHTML = `
                <div class="metric-box">
                    <div class="metric-value">${stats.total_executions}</div>
                    <div class="metric-label">Ejecuciones Totales</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${stats.success_rate.toFixed(1)}%</div>
                    <div class="metric-label">Tasa de Éxito</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${formatNumber(stats.total_tokens, 0)}</div>
                    <div class="metric-label">Tokens Totales</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">$${stats.total_cost.toFixed(4)}</div>
                    <div class="metric-label">Costo Total</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${formatDuration(stats.avg_latency_ms)}</div>
                    <div class="metric-label">Latencia Promedio</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${stats.pipelines_count}</div>
                    <div class="metric-label">Pipelines</div>
                </div>
            `;
            
            // Estadísticas por pipeline
            const pipelineStats = document.getElementById('pipeline-stats');
            const pipelines = Object.values(stats.pipeline_stats);
            
            if (pipelines.length === 0) {
                pipelineStats.innerHTML = '<p class="has-text-grey">No hay datos de pipelines</p>';
                return;
            }
            
            pipelineStats.innerHTML = `
                <table class="table is-fullwidth">
                    <thead>
                        <tr>
                            <th>Pipeline</th>
                            <th>Ejecuciones</th>
                            <th>Tokens</th>
                            <th>Costo</th>
                            <th>Latencia Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${pipelines.map(p => `
                            <tr>
                                <td><strong>${p.name}</strong></td>
                                <td>${p.executions}</td>
                                <td>${formatNumber(p.tokens, 0)}</td>
                                <td>$${p.cost.toFixed(4)}</td>
                                <td>${formatDuration(p.latency)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            
            // Actualizar badge
            document.getElementById('total-executions').textContent = stats.total_executions;
        }

        // ==================== Event Listeners ====================
        document.addEventListener('DOMContentLoaded', () => {
            // Tabs
            document.querySelectorAll('.tabs li').forEach(tab => {
                tab.addEventListener('click', () => showTab(tab.dataset.tab));
            });
            
            // Image upload
            document.getElementById('image-upload').addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploadImage(e.target.files[0]);
                }
            });
            
            // Buttons
            document.getElementById('btn-save-config').addEventListener('click', saveConfig);
            document.getElementById('btn-add-step').addEventListener('click', addPipelineStep);
            document.getElementById('btn-save-pipeline').addEventListener('click', savePipeline);
            document.getElementById('btn-execute').addEventListener('click', executeComparison);
            document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
            
            // Modal close
            document.querySelectorAll('.modal .delete, .modal-background').forEach(el => {
                el.addEventListener('click', () => {
                    document.querySelectorAll('.modal').forEach(m => m.classList.remove('is-active'));
                });
            });
            
            // Cargar datos iniciales
            loadPipelinesForExecution();
            loadStatistics();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal"""
    return HTML_TEMPLATE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
