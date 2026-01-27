"""
Vision LLM Comparator - Aplicación Web
Interfaz completa para comparar LLMs visuales con pipelines configurables
"""

import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from llm_providers import LLMConfig
from pipeline import Pipeline, PipelineRunner, PipelineComparator, PipelineReviewer
from storage import StorageManager

# Inicializar aplicación
app = FastAPI(title="Vision LLM Comparator", version="2.0.0")
storage = StorageManager()

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
    llm_config_name: Optional[str] = None

class PipelineCreate(BaseModel):
    name: str
    description: str = ""
    default_llm_config_name: str
    steps: List[PipelineStepCreate] = []

class ExecuteRequest(BaseModel):
    pipeline_ids: List[str]
    image_name: str
    context_data: Optional[Dict[str, Any]] = None

class BatchExecuteRequest(BaseModel):
    pipeline_id: str
    images: List[Dict[str, Any]]  # [{"image_name": "...", "context_data": {...}}]

class ReviewStepRequest(BaseModel):
    execution_id: str
    step_id: str
    is_correct: bool

# ==================== API Endpoints ====================

# --- LLM Configs ---

@app.post("/api/configs")
async def create_config(config: LLMConfigCreate):
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
    return {"status": "ok", "name": config.name}

@app.get("/api/configs")
async def list_configs():
    """Listar todas las configuraciones de LLM"""
    return storage.list_llm_configs()

@app.delete("/api/configs/{name}")
async def delete_config(name: str):
    """Eliminar una configuración de LLM"""
    if storage.delete_llm_config(name):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Config not found")

# --- Pipelines ---

@app.post("/api/pipelines")
async def create_pipeline(pipeline_data: PipelineCreate):
    """Crear un nuevo pipeline"""
    pipeline = Pipeline(
        id=str(uuid.uuid4()),
        name=pipeline_data.name,
        description=pipeline_data.description,
        default_llm_config_name=pipeline_data.default_llm_config_name
    )
    
    for step in pipeline_data.steps:
        pipeline.add_step(
            name=step.name,
            prompt=step.prompt,
            use_previous_output=step.use_previous_output,
            llm_config_name=step.llm_config_name
        )
    
    storage.save_pipeline(pipeline)
    return {"status": "ok", "id": pipeline.id}

@app.get("/api/pipelines")
async def list_pipelines():
    """Listar todos los pipelines"""
    return storage.list_pipelines()

@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Obtener un pipeline por ID"""
    pipeline = storage.load_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline.to_dict()

@app.put("/api/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, pipeline_data: PipelineCreate):
    """Actualizar un pipeline existente"""
    existing = storage.load_pipeline(pipeline_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = Pipeline(
        id=pipeline_id,
        name=pipeline_data.name,
        description=pipeline_data.description,
        default_llm_config_name=pipeline_data.default_llm_config_name,
        created_at=existing.created_at
    )
    
    for step in pipeline_data.steps:
        pipeline.add_step(
            name=step.name,
            prompt=step.prompt,
            use_previous_output=step.use_previous_output,
            llm_config_name=step.llm_config_name
        )
    
    storage.save_pipeline(pipeline)
    return {"status": "ok", "id": pipeline.id}

@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Eliminar un pipeline"""
    if storage.delete_pipeline(pipeline_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Pipeline not found")

# --- Images ---

@app.post("/api/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    context_json: Optional[str] = Form(None)
):
    """Subir una imagen con contexto JSON opcional"""
    # Guardar imagen
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    saved_path = storage.save_image(temp_path, file.filename)
    os.remove(temp_path)
    
    # Guardar contexto si se proporciona
    if context_json:
        try:
            context_data = json.loads(context_json)
            context_name = Path(file.filename).stem
            storage.save_context(context_data, context_name)
        except json.JSONDecodeError:
            pass
    
    return {
        "status": "ok",
        "filename": file.filename,
        "path": saved_path
    }

@app.post("/api/images/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Subir múltiples imágenes con sus contextos JSON"""
    results = []
    images = []
    contexts = []
    
    # Separar imágenes y JSONs
    for file in files:
        if file.filename.lower().endswith('.json'):
            contexts.append(file)
        else:
            images.append(file)
    
    # Procesar imágenes
    for image_file in images:
        temp_path = f"/tmp/{image_file.filename}"
        with open(temp_path, "wb") as f:
            content = await image_file.read()
            f.write(content)
        
        saved_path = storage.save_image(temp_path, image_file.filename)
        os.remove(temp_path)
        
        # Buscar contexto correspondiente
        image_stem = Path(image_file.filename).stem
        context_data = None
        
        for ctx_file in contexts:
            ctx_stem = Path(ctx_file.filename).stem
            if ctx_stem == image_stem:
                try:
                    ctx_content = await ctx_file.read()
                    context_data = json.loads(ctx_content.decode('utf-8'))
                    storage.save_context(context_data, image_stem)
                    # Reset file position for potential reuse
                    await ctx_file.seek(0)
                except:
                    pass
                break
        
        results.append({
            "filename": image_file.filename,
            "path": saved_path,
            "has_context": context_data is not None
        })
    
    return {"status": "ok", "images": results}

@app.get("/api/images")
async def list_images():
    """Listar todas las imágenes"""
    return storage.list_images()

@app.get("/api/images/view/{filename}")
async def view_image(filename: str):
    """Ver una imagen"""
    image_path = storage.images_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(image_path))

@app.get("/api/images/{filename}/context")
async def get_image_context(filename: str):
    """Obtener el contexto JSON de una imagen"""
    context_name = Path(filename).stem
    context_data = storage.load_context(context_name)
    return {"context_data": context_data}

# --- Execution ---

execution_status = {}

@app.post("/api/execute")
async def execute_pipelines(request: ExecuteRequest, background_tasks: BackgroundTasks):
    """Ejecutar pipelines sobre una imagen"""
    execution_id = str(uuid.uuid4())
    execution_status[execution_id] = {
        "status": "running",
        "progress": "Iniciando...",
        "results": []
    }
    
    background_tasks.add_task(
        run_execution,
        execution_id,
        request.pipeline_ids,
        request.image_name,
        request.context_data
    )
    
    return {"execution_id": execution_id}

async def run_execution(execution_id: str, pipeline_ids: List[str], 
                        image_name: str, context_data: Optional[Dict[str, Any]]):
    """Ejecutar pipelines en background"""
    try:
        image_info = storage.get_image_with_context(image_name)
        if not image_info:
            execution_status[execution_id] = {
                "status": "error",
                "error": f"Imagen no encontrada: {image_name}"
            }
            return
        
        image_path = image_info["image_path"]
        # Usar contexto proporcionado o el asociado a la imagen
        final_context = context_data or image_info.get("context_data")
        
        pipelines = []
        for pid in pipeline_ids:
            pipeline = storage.load_pipeline(pid)
            if pipeline:
                pipelines.append(pipeline)
        
        if not pipelines:
            execution_status[execution_id] = {
                "status": "error",
                "error": "No se encontraron pipelines válidos"
            }
            return
        
        comparator = PipelineComparator(config_loader=storage.get_config_loader())
        
        def progress_callback(msg):
            execution_status[execution_id]["progress"] = msg
        
        results = await comparator.compare(
            pipelines, 
            image_path, 
            final_context,
            progress_callback
        )
        
        # Guardar resultados
        for result in results:
            storage.save_execution(result)
        
        # Generar reporte
        report = PipelineComparator.generate_comparison_report(results)
        
        execution_status[execution_id] = {
            "status": "completed",
            "results": [r.to_dict() for r in results],
            "report": report
        }
        
    except Exception as e:
        execution_status[execution_id] = {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/execute-batch")
async def execute_batch(request: BatchExecuteRequest, background_tasks: BackgroundTasks):
    """Ejecutar un pipeline sobre múltiples imágenes"""
    execution_id = str(uuid.uuid4())
    execution_status[execution_id] = {
        "status": "running",
        "progress": "Iniciando batch...",
        "results": None
    }
    
    background_tasks.add_task(
        run_batch_execution,
        execution_id,
        request.pipeline_id,
        request.images
    )
    
    return {"execution_id": execution_id}

async def run_batch_execution(execution_id: str, pipeline_id: str, 
                              images: List[Dict[str, Any]]):
    """Ejecutar batch en background"""
    try:
        pipeline = storage.load_pipeline(pipeline_id)
        if not pipeline:
            execution_status[execution_id] = {
                "status": "error",
                "error": f"Pipeline no encontrado: {pipeline_id}"
            }
            return
        
        # Preparar pares imagen-contexto
        image_context_pairs = []
        for img_data in images:
            image_name = img_data.get("image_name")
            image_info = storage.get_image_with_context(image_name)
            
            if image_info:
                context = img_data.get("context_data") or image_info.get("context_data")
                image_context_pairs.append({
                    "image_path": image_info["image_path"],
                    "context_data": context
                })
        
        if not image_context_pairs:
            execution_status[execution_id] = {
                "status": "error",
                "error": "No se encontraron imágenes válidas"
            }
            return
        
        runner = PipelineRunner(config_loader=storage.get_config_loader())
        
        def progress_callback(msg):
            execution_status[execution_id]["progress"] = msg
        
        batch_result = await runner.run_batch(
            pipeline,
            image_context_pairs,
            progress_callback
        )
        
        # Guardar batch
        storage.save_batch_execution(batch_result)
        
        execution_status[execution_id] = {
            "status": "completed",
            "results": batch_result.to_dict()
        }
        
    except Exception as e:
        execution_status[execution_id] = {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/execute/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """Obtener estado de una ejecución"""
    if execution_id not in execution_status:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution_status[execution_id]

# --- History ---

@app.get("/api/history")
async def list_history(
    limit: int = 100,
    pipeline_id: Optional[str] = None,
    review_status: Optional[str] = None
):
    """Listar historial de ejecuciones"""
    return storage.list_executions(limit, pipeline_id, review_status)

@app.get("/api/history/{execution_id}")
async def get_execution_details(execution_id: str):
    """Obtener detalles de una ejecución"""
    details = storage.get_execution_details(execution_id)
    if not details:
        raise HTTPException(status_code=404, detail="Execution not found")
    return details

@app.delete("/api/history/{execution_id}")
async def delete_execution(execution_id: str):
    """Eliminar una ejecución"""
    if storage.delete_execution(execution_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Execution not found")

@app.delete("/api/history")
async def clear_history():
    """Limpiar todo el historial"""
    count = storage.clear_history()
    return {"status": "ok", "deleted": count}

# --- Reviews ---

@app.post("/api/reviews")
async def review_step(request: ReviewStepRequest):
    """Revisar un paso de una ejecución"""
    from pipeline import PipelineExecution
    
    exec_data = storage.get_execution_details(request.execution_id)
    if not exec_data:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = PipelineExecution.from_dict(exec_data)
    
    # Reconstruir step_results para tener acceso completo
    execution.step_reviews = exec_data.get("step_reviews", {})
    
    PipelineReviewer.review_step(execution, request.step_id, request.is_correct)
    
    # Actualizar en storage
    exec_data["step_reviews"] = execution.step_reviews
    exec_data["review_status"] = execution.review_status
    
    # Guardar actualización
    for date_dir in storage.executions_dir.iterdir():
        if date_dir.is_dir():
            filepath = date_dir / f"{request.execution_id}.json"
            if filepath.exists():
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(exec_data, f, indent=2, ensure_ascii=False)
                break
    
    return {
        "status": "ok",
        "review_status": execution.review_status,
        "step_reviews": execution.step_reviews
    }

@app.get("/api/reviews/statistics")
async def get_review_statistics():
    """Obtener estadísticas de revisiones"""
    return storage.get_review_statistics()

# --- Statistics ---

@app.get("/api/statistics")
async def get_statistics():
    """Obtener estadísticas generales"""
    return storage.get_statistics()

# --- Batch History ---

@app.get("/api/batch-history")
async def list_batch_history(limit: int = 50):
    """Listar historial de ejecuciones batch"""
    return storage.list_batch_executions(limit)


# ==================== Interfaz Web ====================

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
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: rgba(255,255,255,0.95);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .main-container {
            padding: 2rem;
        }
        .card {
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .card-header {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
            border-radius: 12px 12px 0 0;
        }
        .tabs-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        .tab-content {
            display: none;
        }
        .tab-content.is-active {
            display: block;
        }
        .step-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-box .value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .metric-box .label {
            font-size: 0.85rem;
            opacity: 0.9;
        }
        .execution-card {
            border-left: 4px solid #667eea;
            transition: all 0.3s;
        }
        .execution-card:hover {
            transform: translateX(5px);
        }
        .execution-card.success {
            border-left-color: #48c774;
        }
        .execution-card.error {
            border-left-color: #f14668;
        }
        .review-btn {
            margin: 0.25rem;
        }
        .review-btn.correct {
            background-color: #48c774;
            color: white;
        }
        .review-btn.incorrect {
            background-color: #f14668;
            color: white;
        }
        .context-preview {
            background: #1a1a2e;
            color: #00ff88;
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
        }
        .prompt-preview {
            background: #f5f5f5;
            padding: 0.75rem;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }
        .model-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
        }
        .accuracy-bar {
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            overflow: hidden;
        }
        .accuracy-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #48c774, #00d1b2);
            transition: width 0.5s;
        }
        .image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .step-result-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .step-result-card .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #eee;
        }
        .counter-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
        }
        .help-text {
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <nav class="navbar" role="navigation">
        <div class="navbar-brand">
            <a class="navbar-item" href="#">
                <i class="fas fa-eye mr-2"></i>
                <strong>Vision LLM Comparator</strong>
            </a>
        </div>
        <div class="navbar-end">
            <div class="navbar-item">
                <span class="counter-badge" id="execution-counter">
                    <i class="fas fa-chart-line mr-1"></i>
                    <span id="total-executions">0</span> ejecuciones
                </span>
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
                    <li data-tab="reviews">
                        <a><i class="fas fa-check-double mr-2"></i>Revisiones</a>
                    </li>
                    <li data-tab="statistics">
                        <a><i class="fas fa-chart-bar mr-2"></i>Estadísticas</a>
                    </li>
                </ul>
            </div>
        </div>

        <!-- Tab: Ejecutar -->
        <div id="tab-execute" class="tab-content is-active">
            <div class="columns">
                <div class="column is-5">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-upload mr-2"></i>Subir Imagen
                            </p>
                        </header>
                        <div class="card-content">
                            <div class="tabs is-small is-toggle">
                                <ul>
                                    <li class="is-active" data-upload-tab="single">
                                        <a href="javascript:void(0)">Imagen Individual</a>
                                    </li>
                                    <li data-upload-tab="batch">
                                        <a href="javascript:void(0)">Batch (Múltiples)</a>
                                    </li>
                                </ul>
                            </div>
                            
                            <!-- Single Upload -->
                            <div id="upload-single" class="upload-tab-content">
                                <div class="file has-name is-fullwidth is-primary mb-3">
                                    <label class="file-label">
                                        <input class="file-input" type="file" id="image-file" accept="image/*">
                                        <span class="file-cta">
                                            <span class="file-icon"><i class="fas fa-image"></i></span>
                                            <span class="file-label">Seleccionar imagen...</span>
                                        </span>
                                        <span class="file-name" id="image-file-name">Ningún archivo</span>
                                    </label>
                                </div>
                                
                                <div class="field">
                                    <label class="label">Contexto JSON (opcional)</label>
                                    <p class="help-text">Variables que se sustituirán en los prompts usando $variable o $data.campo</p>
                                    <div class="control">
                                        <textarea class="textarea" id="context-json" rows="6" 
                                            placeholder='{"vehiclePlate": "ABC123", "data": {"color": "rojo"}}'></textarea>
                                    </div>
                                </div>
                                
                                <button class="button is-primary is-fullwidth" id="btn-upload-single">
                                    <i class="fas fa-cloud-upload-alt mr-2"></i>Subir Imagen
                                </button>
                            </div>
                            
                            <!-- Batch Upload -->
                            <div id="upload-batch" class="upload-tab-content" style="display:none;">
                                <div class="notification is-info is-light">
                                    <p><strong>Formato batch:</strong> Sube imágenes junto con archivos JSON del mismo nombre.</p>
                                    <p class="mt-1">Ejemplo: <code>foto01.jpg</code> + <code>foto01.json</code></p>
                                </div>
                                
                                <div class="file has-name is-fullwidth is-primary mb-3">
                                    <label class="file-label">
                                        <input class="file-input" type="file" id="batch-files" multiple accept="image/*,.json">
                                        <span class="file-cta">
                                            <span class="file-icon"><i class="fas fa-folder-open"></i></span>
                                            <span class="file-label">Seleccionar archivos...</span>
                                        </span>
                                        <span class="file-name" id="batch-file-name">Ningún archivo</span>
                                    </label>
                                </div>
                                
                                <button class="button is-primary is-fullwidth" id="btn-upload-batch">
                                    <i class="fas fa-cloud-upload-alt mr-2"></i>Subir Batch
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-images mr-2"></i>Imágenes Disponibles
                            </p>
                        </header>
                        <div class="card-content">
                            <div id="images-list" style="max-height: 300px; overflow-y: auto;"></div>
                        </div>
                    </div>
                </div>
                
                <div class="column is-7">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-play-circle mr-2"></i>Ejecutar Comparación
                            </p>
                        </header>
                        <div class="card-content">
                            <div class="field">
                                <label class="label">Imagen Seleccionada</label>
                                <div class="control">
                                    <input class="input" type="text" id="selected-image" readonly placeholder="Selecciona una imagen">
                                </div>
                            </div>
                            
                            <div id="selected-image-preview" class="mb-4" style="display:none;">
                                <img id="preview-img" class="image-preview">
                            </div>
                            
                            <div id="selected-context-preview" class="mb-4" style="display:none;">
                                <label class="label">Contexto Asociado</label>
                                <pre class="context-preview" id="context-preview-content"></pre>
                            </div>
                            
                            <div class="field">
                                <label class="label">Pipelines a Ejecutar</label>
                                <div id="pipeline-checkboxes"></div>
                            </div>
                            
                            <button class="button is-success is-fullwidth is-medium" id="btn-execute">
                                <i class="fas fa-rocket mr-2"></i>Ejecutar Comparación
                            </button>
                            
                            <button class="button is-info is-fullwidth mt-2" id="btn-execute-batch" style="display:none;">
                                <i class="fas fa-layer-group mr-2"></i>Ejecutar Batch
                            </button>
                        </div>
                    </div>
                    
                    <div class="card" id="results-card" style="display:none;">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-poll mr-2"></i>Resultados
                            </p>
                        </header>
                        <div class="card-content">
                            <div id="execution-progress" style="display:none;">
                                <progress class="progress is-primary" max="100"></progress>
                                <p class="has-text-centered" id="progress-text">Procesando...</p>
                            </div>
                            <div id="execution-results"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Pipelines -->
        <div id="tab-pipelines" class="tab-content">
            <div class="columns">
                <div class="column is-5">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-plus mr-2"></i>Crear Pipeline
                            </p>
                        </header>
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
                                <label class="label">Configuración LLM por Defecto</label>
                                <div class="select is-fullwidth">
                                    <select id="pipeline-default-llm"></select>
                                </div>
                            </div>
                            
                            <div class="field">
                                <label class="label">Pasos del Pipeline</label>
                                <div id="pipeline-steps"></div>
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
                
                <div class="column is-7">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list mr-2"></i>Pipelines Guardados
                            </p>
                        </header>
                        <div class="card-content">
                            <div id="pipelines-list"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Configuraciones LLM -->
        <div id="tab-configs" class="tab-content">
            <div class="columns">
                <div class="column is-5">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-plus mr-2"></i>Nueva Configuración
                            </p>
                        </header>
                        <div class="card-content">
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
                                <p class="help-text">Para Ollama en Docker: http://host.docker.internal:11434</p>
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
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list mr-2"></i>Configuraciones Guardadas
                            </p>
                        </header>
                        <div class="card-content">
                            <div id="configs-list"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Historial -->
        <div id="tab-history" class="tab-content">
            <div class="card">
                <header class="card-header">
                    <p class="card-header-title">
                        <i class="fas fa-history mr-2"></i>Historial de Ejecuciones
                    </p>
                    <div class="card-header-icon">
                        <button class="button is-small is-danger" id="btn-clear-history">
                            <i class="fas fa-trash mr-1"></i>Limpiar
                        </button>
                    </div>
                </header>
                <div class="card-content">
                    <div id="history-list"></div>
                </div>
            </div>
            
            <div class="card mt-4" id="execution-detail-card" style="display:none;">
                <header class="card-header">
                    <p class="card-header-title">
                        <i class="fas fa-info-circle mr-2"></i>Detalle de Ejecución
                    </p>
                    <button class="delete" id="close-detail"></button>
                </header>
                <div class="card-content" id="execution-detail-content"></div>
            </div>
        </div>

        <!-- Tab: Revisiones -->
        <div id="tab-reviews" class="tab-content">
            <div class="columns">
                <div class="column is-8">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-clipboard-check mr-2"></i>Ejecuciones Pendientes de Revisión
                            </p>
                        </header>
                        <div class="card-content">
                            <div id="pending-reviews-list"></div>
                        </div>
                    </div>
                    
                    <div class="card mt-4" id="review-detail-card" style="display:none;">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-search mr-2"></i>Revisar Ejecución
                            </p>
                        </header>
                        <div class="card-content" id="review-detail-content"></div>
                    </div>
                </div>
                
                <div class="column is-4">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-chart-pie mr-2"></i>Métricas de Precisión
                            </p>
                        </header>
                        <div class="card-content" id="accuracy-metrics"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Estadísticas -->
        <div id="tab-statistics" class="tab-content">
            <div class="card">
                <header class="card-header">
                    <p class="card-header-title">
                        <i class="fas fa-globe mr-2"></i>Resumen General
                    </p>
                </header>
                <div class="card-content">
                    <div class="columns" id="stats-summary"></div>
                </div>
            </div>
            
            <div class="columns mt-4">
                <div class="column">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-project-diagram mr-2"></i>Estadísticas por Pipeline
                            </p>
                        </header>
                        <div class="card-content" id="pipeline-stats"></div>
                    </div>
                </div>
                <div class="column">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-robot mr-2"></i>Estadísticas por Modelo
                            </p>
                        </header>
                        <div class="card-content" id="model-stats"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ==================== State ====================
        let selectedImage = null;
        let selectedImageContext = null;
        let pipelineSteps = [];
        let editingPipelineId = null;

        // ==================== Tab Navigation ====================
        document.querySelectorAll('.tabs li').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.tab;
                document.querySelectorAll('.tabs li').forEach(t => t.classList.remove('is-active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('is-active'));
                tab.classList.add('is-active');
                document.getElementById('tab-' + tabId).classList.add('is-active');
                
                // Cargar datos según la pestaña
                if (tabId === 'configs') loadConfigs();
                if (tabId === 'pipelines') { loadConfigs(); loadPipelines(); }
                if (tabId === 'execute') { loadImages(); loadPipelinesForExecution(); }
                if (tabId === 'history') loadHistory();
                if (tabId === 'reviews') { loadPendingReviews(); loadAccuracyMetrics(); }
                if (tabId === 'statistics') loadStatistics();
            });
        });

        // Upload tab switching
        document.querySelectorAll('[data-upload-tab]').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const tabId = tab.dataset.uploadTab;
                document.querySelectorAll('[data-upload-tab]').forEach(t => t.classList.remove('is-active'));
                tab.classList.add('is-active');
                document.getElementById('upload-single').style.display = tabId === 'single' ? 'block' : 'none';
                document.getElementById('upload-batch').style.display = tabId === 'batch' ? 'block' : 'none';
            });
        });

        // ==================== Configs ====================
        async function loadConfigs() {
            const res = await fetch('/api/configs');
            const configs = await res.json();
            
            const list = document.getElementById('configs-list');
            const select = document.getElementById('pipeline-default-llm');
            
            list.innerHTML = configs.map(c => `
                <div class="box">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <p class="title is-5">${c.name}</p>
                                <p class="subtitle is-6">
                                    <span class="model-badge">${c.provider}</span>
                                    ${c.model}
                                </p>
                                <p class="is-size-7 has-text-grey">
                                    API Base: ${c.api_base || 'default'} | 
                                    Max Tokens: ${c.max_tokens} | 
                                    Temp: ${c.temperature}
                                </p>
                            </div>
                        </div>
                        <div class="level-right">
                            <button class="button is-small is-danger" onclick="deleteConfig('${c.name}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `).join('') || '<p class="has-text-grey">No hay configuraciones guardadas</p>';
            
            select.innerHTML = '<option value="">Seleccionar...</option>' + 
                configs.map(c => `<option value="${c.name}">${c.name} (${c.model})</option>`).join('');
        }

        document.getElementById('btn-save-config').addEventListener('click', async () => {
            const config = {
                name: document.getElementById('config-name').value,
                provider: document.getElementById('config-provider').value,
                model: document.getElementById('config-model').value,
                api_key: document.getElementById('config-api-key').value,
                api_base: document.getElementById('config-api-base').value,
                max_tokens: parseInt(document.getElementById('config-max-tokens').value),
                temperature: parseFloat(document.getElementById('config-temperature').value)
            };
            
            await fetch('/api/configs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            
            loadConfigs();
            document.getElementById('config-name').value = '';
            document.getElementById('config-api-key').value = '';
        });

        async function deleteConfig(name) {
            if (confirm('¿Eliminar esta configuración?')) {
                await fetch('/api/configs/' + encodeURIComponent(name), {method: 'DELETE'});
                loadConfigs();
            }
        }

        // ==================== Pipelines ====================
        function renderPipelineSteps() {
            const container = document.getElementById('pipeline-steps');
            container.innerHTML = pipelineSteps.map((step, i) => `
                <div class="step-card">
                    <div class="level mb-2">
                        <div class="level-left">
                            <span class="tag is-primary">Paso ${i + 1}</span>
                        </div>
                        <div class="level-right">
                            <button class="button is-small is-danger" onclick="removeStep(${i})">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="field">
                        <input class="input is-small" type="text" placeholder="Nombre del paso" 
                            value="${step.name}" onchange="updateStep(${i}, 'name', this.value)">
                    </div>
                    <div class="field">
                        <textarea class="textarea is-small" rows="3" placeholder="Prompt (usa $variable para variables JSON)"
                            onchange="updateStep(${i}, 'prompt', this.value)">${step.prompt}</textarea>
                    </div>
                    <div class="field">
                        <label class="label is-small">LLM para este paso (opcional)</label>
                        <div class="select is-small is-fullwidth">
                            <select onchange="updateStep(${i}, 'llm_config_name', this.value)">
                                <option value="">Usar LLM por defecto</option>
                                ${getConfigOptions(step.llm_config_name)}
                            </select>
                        </div>
                    </div>
                    <label class="checkbox">
                        <input type="checkbox" ${step.use_previous_output ? 'checked' : ''} 
                            onchange="updateStep(${i}, 'use_previous_output', this.checked)">
                        Usar salida del paso anterior como contexto
                    </label>
                </div>
            `).join('');
        }

        function getConfigOptions(selected) {
            const select = document.getElementById('pipeline-default-llm');
            return Array.from(select.options).slice(1).map(opt => 
                `<option value="${opt.value}" ${opt.value === selected ? 'selected' : ''}>${opt.text}</option>`
            ).join('');
        }

        document.getElementById('btn-add-step').addEventListener('click', () => {
            pipelineSteps.push({name: '', prompt: '', use_previous_output: true, llm_config_name: null});
            renderPipelineSteps();
        });

        function removeStep(index) {
            pipelineSteps.splice(index, 1);
            renderPipelineSteps();
        }

        function updateStep(index, field, value) {
            pipelineSteps[index][field] = value;
        }

        document.getElementById('btn-save-pipeline').addEventListener('click', async () => {
            const pipeline = {
                name: document.getElementById('pipeline-name').value,
                description: document.getElementById('pipeline-description').value,
                default_llm_config_name: document.getElementById('pipeline-default-llm').value,
                steps: pipelineSteps
            };
            
            const url = editingPipelineId ? '/api/pipelines/' + editingPipelineId : '/api/pipelines';
            const method = editingPipelineId ? 'PUT' : 'POST';
            
            await fetch(url, {
                method: method,
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(pipeline)
            });
            
            loadPipelines();
            clearPipelineForm();
        });

        function clearPipelineForm() {
            document.getElementById('pipeline-name').value = '';
            document.getElementById('pipeline-description').value = '';
            document.getElementById('pipeline-default-llm').value = '';
            pipelineSteps = [];
            editingPipelineId = null;
            renderPipelineSteps();
        }

        async function loadPipelines() {
            const res = await fetch('/api/pipelines');
            const pipelines = await res.json();
            
            document.getElementById('pipelines-list').innerHTML = pipelines.map(p => `
                <div class="box">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <p class="title is-5">${p.name}</p>
                                <p class="subtitle is-6">${p.description || 'Sin descripción'}</p>
                                <p class="is-size-7">
                                    <span class="tag is-info">${p.steps_count} pasos</span>
                                    <span class="tag is-light">${p.default_llm_config_name}</span>
                                </p>
                            </div>
                        </div>
                        <div class="level-right">
                            <button class="button is-small is-info mr-2" onclick="editPipeline('${p.id}')">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="button is-small is-danger" onclick="deletePipeline('${p.id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `).join('') || '<p class="has-text-grey">No hay pipelines guardados</p>';
        }

        async function editPipeline(id) {
            const res = await fetch('/api/pipelines/' + id);
            const pipeline = await res.json();
            
            document.getElementById('pipeline-name').value = pipeline.name;
            document.getElementById('pipeline-description').value = pipeline.description;
            document.getElementById('pipeline-default-llm').value = pipeline.default_llm_config_name;
            pipelineSteps = pipeline.steps;
            editingPipelineId = id;
            renderPipelineSteps();
        }

        async function deletePipeline(id) {
            if (confirm('¿Eliminar este pipeline?')) {
                await fetch('/api/pipelines/' + id, {method: 'DELETE'});
                loadPipelines();
            }
        }

        // ==================== Images ====================
        document.getElementById('image-file').addEventListener('change', (e) => {
            document.getElementById('image-file-name').textContent = e.target.files[0]?.name || 'Ningún archivo';
        });

        document.getElementById('batch-files').addEventListener('change', (e) => {
            const count = e.target.files.length;
            document.getElementById('batch-file-name').textContent = count > 0 ? `${count} archivos seleccionados` : 'Ningún archivo';
        });

        document.getElementById('btn-upload-single').addEventListener('click', async () => {
            const file = document.getElementById('image-file').files[0];
            if (!file) return alert('Selecciona una imagen');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const contextJson = document.getElementById('context-json').value.trim();
            if (contextJson) {
                try {
                    JSON.parse(contextJson);
                    formData.append('context_json', contextJson);
                } catch (e) {
                    return alert('JSON de contexto inválido');
                }
            }
            
            await fetch('/api/images/upload', {method: 'POST', body: formData});
            loadImages();
            document.getElementById('image-file').value = '';
            document.getElementById('image-file-name').textContent = 'Ningún archivo';
            document.getElementById('context-json').value = '';
        });

        document.getElementById('btn-upload-batch').addEventListener('click', async () => {
            const files = document.getElementById('batch-files').files;
            if (files.length === 0) return alert('Selecciona archivos');
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            await fetch('/api/images/upload-batch', {method: 'POST', body: formData});
            loadImages();
            document.getElementById('batch-files').value = '';
            document.getElementById('batch-file-name').textContent = 'Ningún archivo';
        });

        async function loadImages() {
            const res = await fetch('/api/images');
            const images = await res.json();
            
            document.getElementById('images-list').innerHTML = images.map(img => `
                <div class="box is-clickable" onclick="selectImage('${img.name}', ${img.has_context})">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <p class="has-text-weight-bold">${img.name}</p>
                                <p class="is-size-7 has-text-grey">
                                    ${(img.size / 1024).toFixed(1)} KB
                                    ${img.has_context ? '<span class="tag is-success is-light ml-2">JSON</span>' : ''}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('') || '<p class="has-text-grey">No hay imágenes</p>';
        }

        async function selectImage(name, hasContext) {
            selectedImage = name;
            document.getElementById('selected-image').value = name;
            document.getElementById('selected-image-preview').style.display = 'block';
            document.getElementById('preview-img').src = '/api/images/view/' + name;
            
            if (hasContext) {
                const res = await fetch('/api/images/' + name + '/context');
                const data = await res.json();
                selectedImageContext = data.context_data;
                document.getElementById('selected-context-preview').style.display = 'block';
                document.getElementById('context-preview-content').textContent = JSON.stringify(data.context_data, null, 2);
            } else {
                selectedImageContext = null;
                document.getElementById('selected-context-preview').style.display = 'none';
            }
        }

        // ==================== Execution ====================
        async function loadPipelinesForExecution() {
            const res = await fetch('/api/pipelines');
            const pipelines = await res.json();
            
            document.getElementById('pipeline-checkboxes').innerHTML = pipelines.map(p => `
                <label class="checkbox is-block mb-2">
                    <input type="checkbox" value="${p.id}" class="pipeline-checkbox">
                    ${p.name} <span class="tag is-light">${p.steps_count} pasos</span>
                </label>
            `).join('') || '<p class="has-text-grey">No hay pipelines</p>';
        }

        document.getElementById('btn-execute').addEventListener('click', async () => {
            if (!selectedImage) return alert('Selecciona una imagen');
            
            const selectedPipelines = Array.from(document.querySelectorAll('.pipeline-checkbox:checked'))
                .map(cb => cb.value);
            
            if (selectedPipelines.length === 0) return alert('Selecciona al menos un pipeline');
            
            document.getElementById('results-card').style.display = 'block';
            document.getElementById('execution-progress').style.display = 'block';
            document.getElementById('execution-results').innerHTML = '';
            
            const res = await fetch('/api/execute', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    pipeline_ids: selectedPipelines,
                    image_name: selectedImage,
                    context_data: selectedImageContext
                })
            });
            
            const {execution_id} = await res.json();
            pollExecutionStatus(execution_id);
        });

        async function pollExecutionStatus(executionId) {
            const interval = setInterval(async () => {
                const res = await fetch('/api/execute/' + executionId + '/status');
                const status = await res.json();
                
                document.getElementById('progress-text').textContent = status.progress || 'Procesando...';
                
                if (status.status === 'completed') {
                    clearInterval(interval);
                    document.getElementById('execution-progress').style.display = 'none';
                    renderExecutionResults(status.results, status.report);
                    loadStatistics();
                } else if (status.status === 'error') {
                    clearInterval(interval);
                    document.getElementById('execution-progress').style.display = 'none';
                    document.getElementById('execution-results').innerHTML = `
                        <div class="notification is-danger">${status.error}</div>
                    `;
                }
            }, 1000);
        }

        function renderExecutionResults(results, report) {
            let html = '';
            
            if (report) {
                html += `
                    <div class="notification is-info is-light mb-4">
                        <p><strong>Resumen:</strong></p>
                        <p>Más rápido: <strong>${report.fastest || 'N/A'}</strong></p>
                        <p>Más económico: <strong>${report.cheapest || 'N/A'}</strong></p>
                        <p>Menos tokens: <strong>${report.least_tokens || 'N/A'}</strong></p>
                    </div>
                `;
            }
            
            for (const result of results) {
                html += `
                    <div class="box execution-card ${result.success ? 'success' : 'error'}">
                        <div class="level">
                            <div class="level-left">
                                <div>
                                    <p class="title is-5">${result.pipeline_name}</p>
                                    <p class="is-size-7">
                                        ${result.models_used.map(m => `<span class="model-badge">${m}</span>`).join('')}
                                    </p>
                                </div>
                            </div>
                            <div class="level-right">
                                <div class="has-text-right">
                                    <p><strong>${result.total_latency_ms.toFixed(0)}ms</strong></p>
                                    <p class="is-size-7">${result.total_tokens} tokens | $${result.total_cost.toFixed(4)}</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            ${result.step_results.map((step, i) => `
                                <div class="step-result-card">
                                    <div class="step-header">
                                        <div>
                                            <span class="tag is-primary">Paso ${i+1}</span>
                                            <strong class="ml-2">${step.step_name}</strong>
                                            <span class="model-badge ml-2">${step.model_used}</span>
                                        </div>
                                        <div class="is-size-7">
                                            ${step.latency_ms.toFixed(0)}ms | ${step.total_tokens} tokens
                                        </div>
                                    </div>
                                    <details>
                                        <summary class="is-size-7 has-text-grey">Ver prompt usado</summary>
                                        <pre class="prompt-preview mt-2">${escapeHtml(step.prompt_used)}</pre>
                                    </details>
                                    <div class="content mt-2">
                                        <pre style="white-space: pre-wrap; background: #f5f5f5; padding: 0.75rem; border-radius: 4px;">${escapeHtml(step.content)}</pre>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
            
            document.getElementById('execution-results').innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ==================== History ====================
        async function loadHistory() {
            const res = await fetch('/api/history?limit=50');
            const history = await res.json();
            
            document.getElementById('history-list').innerHTML = history.map(h => `
                <div class="box execution-card ${h.success ? 'success' : 'error'} is-clickable" 
                     onclick="showExecutionDetail('${h.id}')">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <p class="has-text-weight-bold">${h.pipeline_name}</p>
                                <p class="is-size-7 has-text-grey">${h.image_name}</p>
                                <p class="is-size-7">
                                    ${h.models_used.map(m => `<span class="model-badge">${m}</span>`).join('')}
                                    ${h.review_status === 'reviewed' ? '<span class="tag is-success ml-2">Revisado</span>' : 
                                      h.review_status === 'pending' ? '<span class="tag is-warning ml-2">Pendiente</span>' : ''}
                                </p>
                            </div>
                        </div>
                        <div class="level-right">
                            <div class="has-text-right">
                                <p><strong>${h.total_latency_ms.toFixed(0)}ms</strong></p>
                                <p class="is-size-7">${h.total_tokens} tokens | $${h.total_cost.toFixed(4)}</p>
                                <p class="is-size-7 has-text-grey">${new Date(h.started_at).toLocaleString()}</p>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('') || '<p class="has-text-grey">No hay historial</p>';
            
            // Update counter
            document.getElementById('total-executions').textContent = history.length;
        }

        async function showExecutionDetail(id) {
            const res = await fetch('/api/history/' + id);
            const detail = await res.json();
            
            document.getElementById('execution-detail-card').style.display = 'block';
            document.getElementById('execution-detail-content').innerHTML = `
                <div class="columns">
                    <div class="column is-4">
                        <img src="/api/images/view/${detail.image_name}" class="image-preview">
                    </div>
                    <div class="column">
                        <p><strong>Pipeline:</strong> ${detail.pipeline_name}</p>
                        <p><strong>Imagen:</strong> ${detail.image_name}</p>
                        <p><strong>Latencia:</strong> ${detail.total_latency_ms.toFixed(0)}ms</p>
                        <p><strong>Tokens:</strong> ${detail.total_input_tokens} in / ${detail.total_output_tokens} out</p>
                        <p><strong>Costo:</strong> $${detail.total_cost.toFixed(4)}</p>
                        ${detail.context_data ? `
                            <p class="mt-2"><strong>Contexto:</strong></p>
                            <pre class="context-preview">${JSON.stringify(detail.context_data, null, 2)}</pre>
                        ` : ''}
                    </div>
                </div>
                
                <h4 class="title is-5 mt-4">Pasos</h4>
                ${detail.step_results.map((step, i) => `
                    <div class="step-result-card">
                        <div class="step-header">
                            <div>
                                <span class="tag is-primary">Paso ${i+1}</span>
                                <strong class="ml-2">${step.step_name}</strong>
                                <span class="model-badge ml-2">${step.provider_used}:${step.model_used}</span>
                            </div>
                            <div class="is-size-7">
                                ${step.latency_ms.toFixed(0)}ms | ${step.input_tokens}/${step.output_tokens} tokens | $${step.cost_estimate.toFixed(4)}
                            </div>
                        </div>
                        <details>
                            <summary class="is-size-7 has-text-grey">Ver prompt</summary>
                            <pre class="prompt-preview mt-2">${escapeHtml(step.prompt_used)}</pre>
                        </details>
                        <div class="content mt-2">
                            <pre style="white-space: pre-wrap; background: #f5f5f5; padding: 0.75rem; border-radius: 4px;">${escapeHtml(step.content)}</pre>
                        </div>
                    </div>
                `).join('')}
            `;
        }

        document.getElementById('close-detail').addEventListener('click', () => {
            document.getElementById('execution-detail-card').style.display = 'none';
        });

        document.getElementById('btn-clear-history').addEventListener('click', async () => {
            if (confirm('¿Eliminar todo el historial?')) {
                await fetch('/api/history', {method: 'DELETE'});
                loadHistory();
            }
        });

        // ==================== Reviews ====================
        async function loadPendingReviews() {
            const res = await fetch('/api/history?review_status=pending&limit=50');
            const pending = await res.json();
            
            document.getElementById('pending-reviews-list').innerHTML = pending.map(h => `
                <div class="box is-clickable" onclick="showReviewDetail('${h.id}')">
                    <div class="level">
                        <div class="level-left">
                            <div>
                                <p class="has-text-weight-bold">${h.pipeline_name}</p>
                                <p class="is-size-7">${h.image_name} | ${h.steps_count} pasos</p>
                            </div>
                        </div>
                        <div class="level-right">
                            <span class="tag is-warning">Pendiente</span>
                        </div>
                    </div>
                </div>
            `).join('') || '<p class="has-text-grey">No hay ejecuciones pendientes de revisión</p>';
        }

        async function showReviewDetail(id) {
            const res = await fetch('/api/history/' + id);
            const detail = await res.json();
            
            document.getElementById('review-detail-card').style.display = 'block';
            document.getElementById('review-detail-content').innerHTML = `
                <div class="columns">
                    <div class="column is-4">
                        <img src="/api/images/view/${detail.image_name}" class="image-preview">
                    </div>
                    <div class="column">
                        <p><strong>Pipeline:</strong> ${detail.pipeline_name}</p>
                        <p><strong>Imagen:</strong> ${detail.image_name}</p>
                    </div>
                </div>
                
                <h4 class="title is-5 mt-4">Revisar Pasos</h4>
                ${detail.step_results.map((step, i) => {
                    const reviewed = detail.step_reviews && detail.step_reviews[step.step_id] !== undefined;
                    const isCorrect = detail.step_reviews && detail.step_reviews[step.step_id];
                    return `
                        <div class="step-result-card" id="review-step-${step.step_id}">
                            <div class="step-header">
                                <div>
                                    <span class="tag is-primary">Paso ${i+1}</span>
                                    <strong class="ml-2">${step.step_name}</strong>
                                    <span class="model-badge ml-2">${step.model_used}</span>
                                </div>
                                <div>
                                    <button class="button is-small review-btn ${reviewed && isCorrect ? 'correct' : ''}" 
                                            onclick="reviewStep('${id}', '${step.step_id}', true)">
                                        <i class="fas fa-check"></i> Correcto
                                    </button>
                                    <button class="button is-small review-btn ${reviewed && !isCorrect ? 'incorrect' : ''}"
                                            onclick="reviewStep('${id}', '${step.step_id}', false)">
                                        <i class="fas fa-times"></i> Incorrecto
                                    </button>
                                </div>
                            </div>
                            <div class="content mt-2">
                                <pre style="white-space: pre-wrap; background: #f5f5f5; padding: 0.75rem; border-radius: 4px;">${escapeHtml(step.content)}</pre>
                            </div>
                        </div>
                    `;
                }).join('')}
            `;
        }

        async function reviewStep(executionId, stepId, isCorrect) {
            await fetch('/api/reviews', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    execution_id: executionId,
                    step_id: stepId,
                    is_correct: isCorrect
                })
            });
            
            // Refresh
            showReviewDetail(executionId);
            loadPendingReviews();
            loadAccuracyMetrics();
        }

        async function loadAccuracyMetrics() {
            const res = await fetch('/api/reviews/statistics');
            const stats = await res.json();
            
            if (stats.message) {
                document.getElementById('accuracy-metrics').innerHTML = `<p class="has-text-grey">${stats.message}</p>`;
                return;
            }
            
            let html = `<p class="mb-3"><strong>Total revisadas:</strong> ${stats.total_reviewed}</p>`;
            
            html += '<h5 class="title is-6">Por Pipeline</h5>';
            for (const [pid, data] of Object.entries(stats.pipeline_accuracy)) {
                html += `
                    <div class="mb-3">
                        <p class="has-text-weight-bold">${data.name}</p>
                        <div class="accuracy-bar">
                            <div class="fill" style="width: ${data.step_accuracy_pct}%"></div>
                        </div>
                        <p class="is-size-7">${data.step_accuracy_pct}% precisión por paso | ${data.execution_accuracy_pct}% ejecuciones perfectas</p>
                    </div>
                `;
            }
            
            html += '<h5 class="title is-6 mt-4">Por Modelo</h5>';
            for (const [model, data] of Object.entries(stats.model_accuracy)) {
                html += `
                    <div class="mb-3">
                        <p class="has-text-weight-bold">${model}</p>
                        <div class="accuracy-bar">
                            <div class="fill" style="width: ${data.accuracy_pct}%"></div>
                        </div>
                        <p class="is-size-7">${data.accuracy_pct}% precisión (${data.correct_steps}/${data.total_steps} pasos)</p>
                    </div>
                `;
            }
            
            document.getElementById('accuracy-metrics').innerHTML = html;
        }

        // ==================== Statistics ====================
        async function loadStatistics() {
            const res = await fetch('/api/statistics');
            const stats = await res.json();
            
            document.getElementById('stats-summary').innerHTML = `
                <div class="column">
                    <div class="metric-box">
                        <p class="value">${stats.total_executions}</p>
                        <p class="label">Ejecuciones</p>
                    </div>
                </div>
                <div class="column">
                    <div class="metric-box">
                        <p class="value">${stats.success_rate.toFixed(1)}%</p>
                        <p class="label">Tasa de Éxito</p>
                    </div>
                </div>
                <div class="column">
                    <div class="metric-box">
                        <p class="value">${(stats.total_tokens / 1000).toFixed(1)}K</p>
                        <p class="label">Tokens Totales</p>
                    </div>
                </div>
                <div class="column">
                    <div class="metric-box">
                        <p class="value">$${stats.total_cost.toFixed(2)}</p>
                        <p class="label">Costo Total</p>
                    </div>
                </div>
                <div class="column">
                    <div class="metric-box">
                        <p class="value">${stats.avg_latency_ms.toFixed(0)}ms</p>
                        <p class="label">Latencia Promedio</p>
                    </div>
                </div>
            `;
            
            // Pipeline stats
            let pipelineHtml = '';
            for (const [pid, data] of Object.entries(stats.pipeline_stats)) {
                pipelineHtml += `
                    <div class="box">
                        <p class="has-text-weight-bold">${data.name}</p>
                        <p class="is-size-7">
                            ${data.executions} ejecuciones | 
                            ${data.tokens} tokens | 
                            $${data.cost.toFixed(4)} | 
                            ${(data.latency / data.executions).toFixed(0)}ms avg
                            ${data.accuracy !== null ? ` | ${data.accuracy}% precisión` : ''}
                        </p>
                    </div>
                `;
            }
            document.getElementById('pipeline-stats').innerHTML = pipelineHtml || '<p class="has-text-grey">Sin datos</p>';
            
            // Model stats
            let modelHtml = '';
            for (const [model, data] of Object.entries(stats.model_stats)) {
                modelHtml += `
                    <div class="box">
                        <p class="has-text-weight-bold">${model}</p>
                        <p class="is-size-7">${data.executions} usos</p>
                    </div>
                `;
            }
            document.getElementById('model-stats').innerHTML = modelHtml || '<p class="has-text-grey">Sin datos</p>';
            
            // Update counter
            document.getElementById('total-executions').textContent = stats.total_executions;
        }

        // ==================== Init ====================
        loadConfigs();
        loadImages();
        loadPipelinesForExecution();
        loadStatistics();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

# ==================== Main ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
