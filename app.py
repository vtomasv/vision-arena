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
from pdf_generator import generate_forensic_report
from semantic_search import SemanticSearchEngine
from visual_agent import VisualAgent, ensure_image_packages

# Inicializar aplicación
app = FastAPI(title="Vision LLM Comparator", version="3.0.0")
storage = StorageManager()

# Motor de búsqueda semántica
# Use the same base directory as StorageManager for consistency
data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.expanduser("~"), ".vision_llm_comparator"))
search_engine = SemanticSearchEngine(os.path.join(data_dir, "search_index"))

# Agente visual
visual_agent = VisualAgent(
    sessions_dir=os.path.join(data_dir, "agent_sessions"),
    config_loader=None  # Se configura después de inicializar storage
)

@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al arrancar"""
    # Configurar config_loader del agente (usa get_llm_config_full para obtener API key completa)
    def agent_config_loader(name):
        config = storage.get_llm_config_full(name)
        if config:
            return LLMConfig(**config) if isinstance(config, dict) else config
        return None
    visual_agent.config_loader = agent_config_loader
    
    # Instalar paquetes de imagen si es necesario
    try:
        ensure_image_packages()
    except Exception:
        pass

# ==================== Modelos Pydantic ====================

class LLMConfigCreate(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: Dict[str, Any] = {}

class PipelineStepCreate(BaseModel):
    name: str
    prompt: str
    use_previous_output: bool = True
    llm_config_name: Optional[str] = None
    index_for_search: bool = False

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
        temperature=config.temperature,
        extra_params=config.extra_params
    )
    storage.save_llm_config(llm_config)
    return {"status": "ok", "name": config.name}

@app.get("/api/configs")
async def list_configs():
    """Listar todas las configuraciones de LLM"""
    return storage.list_llm_configs()

@app.get("/api/configs/{name}")
async def get_config(name: str):
    """Obtener una configuración de LLM por nombre (con API key completa para edición)"""
    config = storage.get_llm_config_full(name)
    if config:
        return config
    raise HTTPException(status_code=404, detail="Config not found")

@app.put("/api/configs/{name}")
async def update_config(name: str, config: LLMConfigCreate):
    """Actualizar una configuración de LLM existente"""
    existing = storage.get_llm_config(name)
    if not existing:
        raise HTTPException(status_code=404, detail="Config not found")
    
    llm_config = LLMConfig(
        name=config.name,
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        extra_params=config.extra_params
    )
    
    # Si el nombre cambió, eliminar el anterior
    if name != config.name:
        storage.delete_llm_config(name)
    
    storage.save_llm_config(llm_config)
    return {"status": "ok", "name": config.name}

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
            llm_config_name=step.llm_config_name,
            index_for_search=step.index_for_search
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
            llm_config_name=step.llm_config_name,
            index_for_search=step.index_for_search
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
        
        # Indexar semánticamente si algún paso tiene index_for_search
        try:
            for result in results:
                result_dict = result.to_dict()
                pipeline = None
                for p in pipelines:
                    if p.name == result_dict.get("pipeline_name"):
                        pipeline = p
                        break
                if pipeline:
                    index_texts = []
                    for step_result in result_dict.get("step_results", []):
                        step_id = step_result.get("step_id", "")
                        for step in pipeline.steps:
                            if step.id == step_id and step.index_for_search:
                                content = step_result.get("content", "")
                                if content:
                                    index_texts.append(content)
                    if index_texts:
                        combined_text = "\n".join(index_texts)
                        metadata = {
                            "image_name": image_name,
                            "image_path": image_path,
                            "pipeline_name": result_dict.get("pipeline_name", ""),
                            "execution_id": result_dict.get("id", ""),
                            "context_data": final_context or {}
                        }
                        search_engine.index_image(image_name, image_path, combined_text, metadata)
        except Exception as e:
            print(f"Warning: Semantic indexing failed: {e}")
        
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

# --- Forensic PDF Report ---

@app.get("/api/history/{execution_id}/pdf")
async def generate_pdf_report(execution_id: str):
    """Generar reporte PDF forense para una ejecución"""
    # Obtener detalles de la ejecución
    exec_data = storage.get_execution_details(execution_id)
    if not exec_data:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # Obtener ruta de la imagen
    image_name = exec_data.get("image_name", "")
    image_path = storage.images_dir / image_name if image_name else None
    
    if not image_path or not image_path.exists():
        # Intentar buscar en el directorio de imágenes
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            potential_path = storage.images_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
    
    if not image_path:
        image_path = storage.images_dir / image_name  # Usar el nombre original aunque no exista
    
    # Preparar contexto
    context = exec_data.get("context_data", {})
    
    # Preparar pasos
    steps = []
    for step_result in exec_data.get("step_results", []):
        steps.append({
            "step_name": step_result.get("step_name", "Unknown Step"),
            "llm_used": step_result.get("llm_config_name", "Unknown"),
            "prompt": step_result.get("prompt_used", ""),
            "response": step_result.get("response", ""),
            "latency_ms": step_result.get("latency_ms", 0),
            "input_tokens": step_result.get("input_tokens", 0),
            "output_tokens": step_result.get("output_tokens", 0)
        })
    
    # Generar PDF
    pipeline_name = exec_data.get("pipeline_name", "Vehicle Analysis Pipeline")
    
    try:
        result = generate_forensic_report(
            execution_id=execution_id,
            image_path=str(image_path) if image_path else "",
            context=context,
            steps=steps,
            output_dir=str(storage.reports_dir),
            pipeline_name=pipeline_name
        )
        
        # Devolver el archivo PDF
        return FileResponse(
            result["report_path"],
            media_type="application/pdf",
            filename=f"forensic_report_{execution_id}.pdf",
            headers={
                "X-Report-Hash": result["report_hash"],
                "X-Image-Hash": result.get("image_hash", ""),
                "X-Timestamp": result["timestamp"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

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


# --- Semantic Search ---

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    filters: Optional[Dict[str, Any]] = None

@app.post("/api/search")
async def semantic_search(request: SearchRequest):
    """Buscar imágenes por descripción semántica"""
    try:
        results = search_engine.search(request.query, request.top_k, filters=request.filters)
        return {"results": results, "query": request.query, "total": len(results)}
    except Exception as e:
        return {"results": [], "query": request.query, "total": 0, "error": str(e)}

@app.get("/api/search/stats")
async def search_stats():
    """Obtener estadísticas del índice semántico"""
    stats = search_engine.get_stats()
    # Map field names for frontend compatibility
    return {
        "total_images": stats.get("total_indexed_images", 0),
        "total_indexed_images": stats.get("total_indexed_images", 0),
        "total_descriptions": stats.get("total_descriptions", 0),
        "total_attributes": stats.get("total_attributes", 0),
        "total_documents": stats.get("total_descriptions", 0),
        "unique_attribute_keys": stats.get("unique_attribute_keys", 0),
        "faiss_vectors": stats.get("faiss_vectors", 0),
        "vector_dimension": stats.get("config", {}).get("embedding_dim", 384),
        "embeddings_loaded": stats.get("embeddings_loaded", False),
        "config": stats.get("config", {}),
        "last_updated": "N/A"
    }

@app.post("/api/search/reindex")
async def reindex_all():
    """Reindexar todas las ejecuciones existentes"""
    count = 0
    errors = []
    try:
        executions = storage.list_executions(limit=1000)
        for exec_summary in executions:
            exec_id = exec_summary.get("id", "")
            try:
                exec_data = storage.get_execution_details(exec_id)
                if not exec_data:
                    continue
                image_name = exec_data.get("image_name", "")
                image_path = str(storage.images_dir / image_name)
                pipeline_name = exec_data.get("pipeline_name", "")
                
                # Recopilar textos de todos los pasos
                index_texts = []
                for step_result in exec_data.get("step_results", []):
                    content = step_result.get("content", "")
                    if content:
                        index_texts.append(content)
                
                if index_texts:
                    combined_text = "\n".join(index_texts)
                    metadata = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "pipeline_name": pipeline_name,
                        "execution_id": exec_id,
                        "context_data": exec_data.get("context_data", {})
                    }
                    search_engine.index_image(image_name, image_path, combined_text, metadata)
                    count += 1
            except Exception as e:
                errors.append(f"{exec_id}: {str(e)}")
    except Exception as e:
        return {"status": "error", "error": str(e), "indexed": count}
    
    result = {"status": "ok", "indexed": count}
    if errors:
        result["errors"] = errors[:10]  # Limit error messages
    return result

@app.get("/api/search/image-details/{image_name}")
async def get_image_search_details(image_name: str):
    """Obtener todos los datos indexados de una imagen por nombre"""
    details = search_engine.get_image_details_by_name(image_name)
    if not details:
        # Fallback: try by ID
        details = search_engine.get_image_details(image_name)
    if not details:
        raise HTTPException(status_code=404, detail="Image not found in index")
    # Map fields for frontend compatibility
    return {
        "image_id": details.get("id", ""),
        "image_name": details.get("image_name", image_name),
        "image_path": details.get("image_path", ""),
        "image_hash": details.get("image_hash", ""),
        "description": details.get("combined_text", ""),
        "combined_text": details.get("combined_text", ""),
        "pipeline_name": details.get("pipeline_name", ""),
        "execution_id": details.get("execution_id", ""),
        "indexed_at": details.get("indexed_at", ""),
        "descriptions": details.get("descriptions", []),
        "attributes": details.get("attributes", []),
        "metadata": {
            "context_data": details.get("context_data", {})
        }
    }

# --- Visual Agent ---

class AgentSessionCreate(BaseModel):
    image_name: str
    llm_config_name: str
    title: str = ""

class AgentMessageRequest(BaseModel):
    message: str
    search_image_path: Optional[str] = None

@app.post("/api/agent/sessions")
async def create_agent_session(request: AgentSessionCreate):
    """Crear una nueva sesión de agente"""
    image_info = storage.get_image_with_context(request.image_name)
    if not image_info:
        raise HTTPException(status_code=404, detail="Image not found")
    
    session = visual_agent.create_session(
        image_path=image_info["image_path"],
        image_name=request.image_name,
        llm_config_name=request.llm_config_name,
        title=request.title
    )
    return {"status": "ok", "session": session.to_dict()}

@app.get("/api/agent/sessions")
async def list_agent_sessions():
    """Listar todas las sesiones de agente"""
    return visual_agent.list_sessions()

@app.get("/api/agent/sessions/{session_id}")
async def get_agent_session(session_id: str):
    """Obtener una sesión de agente"""
    session = visual_agent.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()

@app.post("/api/agent/sessions/{session_id}/messages")
async def send_agent_message(session_id: str, request: AgentMessageRequest):
    """Enviar un mensaje al agente"""
    try:
        response = await visual_agent.send_message(
            session_id, 
            request.message,
            request.search_image_path
        )
        return {"status": "ok", "message": response.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/agent/sessions/{session_id}")
async def delete_agent_session(session_id: str):
    """Eliminar una sesión de agente"""
    if visual_agent.delete_session(session_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/agent/sessions/{session_id}/pdf")
async def generate_agent_report(session_id: str):
    """Generar reporte PDF de una sesión de agente"""
    report_path = visual_agent.generate_session_report(session_id)
    if not report_path:
        raise HTTPException(status_code=500, detail="Error generating report")
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"agent_report_{session_id[:8]}.pdf"
    )

@app.get("/api/agent/outputs/{session_id}/{filename}")
async def get_agent_output(session_id: str, filename: str):
    """Obtener un archivo de salida del agente"""
    output_path = Path(data_dir) / "agent_sessions" / session_id / "outputs" / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(output_path))


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
    <!-- Markdown rendering and syntax highlighting -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
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
        /* Agent Chat Markdown Styles */
        .agent-msg { max-width: 85%; padding: 1rem 1.25rem; border-radius: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .agent-msg-user { background: linear-gradient(135deg, #e3f2fd, #bbdefb); margin-left: auto; border-bottom-right-radius: 4px; }
        .agent-msg-assistant { background: #ffffff; border: 1px solid #e8e8e8; border-bottom-left-radius: 4px; }
        .agent-msg-system { background: #f0f4f8; text-align: center; max-width: 100%; font-size: 0.85rem; color: #666; padding: 0.5rem 1rem; border-radius: 8px; }
        .agent-msg-error { background: #fff5f5; border: 1px solid #f14668; }
        .agent-msg .md-content h1, .agent-msg .md-content h2, .agent-msg .md-content h3 { margin-top: 0.75rem; margin-bottom: 0.5rem; color: #363636; }
        .agent-msg .md-content h1 { font-size: 1.3rem; }
        .agent-msg .md-content h2 { font-size: 1.15rem; }
        .agent-msg .md-content h3 { font-size: 1.05rem; }
        .agent-msg .md-content p { margin-bottom: 0.5rem; line-height: 1.6; }
        .agent-msg .md-content ul, .agent-msg .md-content ol { margin-left: 1.5rem; margin-bottom: 0.5rem; }
        .agent-msg .md-content li { margin-bottom: 0.25rem; }
        .agent-msg .md-content table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
        .agent-msg .md-content table th { background: #f5f5f5; padding: 0.5rem 0.75rem; border: 1px solid #ddd; text-align: left; font-weight: 600; }
        .agent-msg .md-content table td { padding: 0.5rem 0.75rem; border: 1px solid #ddd; }
        .agent-msg .md-content table tr:nth-child(even) { background: #fafafa; }
        .agent-msg .md-content strong { color: #363636; }
        .agent-msg .md-content em { color: #555; }
        .agent-msg .md-content blockquote { border-left: 4px solid #667eea; padding: 0.5rem 1rem; margin: 0.5rem 0; background: #f8f9ff; }
        .agent-msg .md-content img { max-width: 100%; border-radius: 8px; margin: 0.5rem 0; cursor: pointer; border: 2px solid #eee; transition: border-color 0.2s; }
        .agent-msg .md-content img:hover { border-color: #667eea; }
        .agent-msg .md-content code:not(pre code) { background: #f0f0f0; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; color: #e74c3c; }
        .agent-msg .md-content pre { margin: 0.75rem 0; border-radius: 8px; overflow: hidden; }
        .agent-msg .md-content pre code { font-size: 0.85rem; line-height: 1.5; }
        .agent-plan { background: linear-gradient(135deg, #f0f4ff, #e8ecff); border: 1px solid #c5cae9; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
        .agent-plan::before { content: '\f0ae  Plan'; font-family: 'Font Awesome 6 Free'; font-weight: 900; display: block; font-size: 0.8rem; color: #667eea; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
        .agent-code-block { margin: 0.75rem 0; border-radius: 10px; overflow: hidden; border: 1px solid #2d2d2d; }
        .agent-code-header { background: #1e1e2e; color: #cdd6f4; padding: 0.4rem 0.75rem; font-size: 0.75rem; display: flex; justify-content: space-between; align-items: center; }
        .agent-code-header .lang-badge { background: #45475a; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; }
        .agent-code-output { background: #1a1b26; color: #a9b1d6; padding: 0.75rem; font-family: 'Fira Code', 'Cascadia Code', monospace; font-size: 0.8rem; border-top: 1px solid #333; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; }
        .agent-code-output.success { border-left: 3px solid #48c774; }
        .agent-code-output.error { border-left: 3px solid #f14668; color: #f7768e; }
        .agent-image-output { margin: 0.75rem 0; text-align: center; }
        .agent-image-output img { max-width: 100%; max-height: 500px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); cursor: pointer; transition: transform 0.2s; }
        .agent-image-output img:hover { transform: scale(1.02); }
        .agent-image-caption { font-size: 0.8rem; color: #888; margin-top: 0.25rem; }
        .agent-msg-meta { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; font-size: 0.75rem; color: #999; }
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
                    <li data-tab="search">
                        <a><i class="fas fa-search mr-2"></i>Búsqueda</a>
                    </li>
                    <li data-tab="agent">
                        <a><i class="fas fa-robot mr-2"></i>Agente Visual</a>
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
                                
                                <!-- Preview de archivos seleccionados -->
                                <div id="batch-preview" class="mb-3" style="display:none;">
                                    <label class="label">Archivos a subir:</label>
                                    <div id="batch-preview-list" style="max-height: 150px; overflow-y: auto;"></div>
                                </div>
                                
                                <button class="button is-primary is-fullwidth mb-2" id="btn-upload-batch">
                                    <i class="fas fa-cloud-upload-alt mr-2"></i>Subir Batch
                                </button>
                                
                                <!-- Progreso de subida -->
                                <div id="batch-upload-progress" class="mb-3" style="display:none;">
                                    <progress class="progress is-primary" id="batch-progress-bar" max="100">0%</progress>
                                    <p class="has-text-centered is-size-7" id="batch-progress-text">Subiendo...</p>
                                </div>
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
                            <!-- Tabs para modo individual o batch -->
                            <div class="tabs is-small is-toggle mb-4">
                                <ul>
                                    <li class="is-active" data-exec-tab="single">
                                        <a href="javascript:void(0)">Imagen Individual</a>
                                    </li>
                                    <li data-exec-tab="batch">
                                        <a href="javascript:void(0)">Batch (Múltiples)</a>
                                    </li>
                                </ul>
                            </div>
                            
                            <!-- Modo Individual -->
                            <div id="exec-single" class="exec-tab-content">
                                <div class="field">
                                    <label class="label">Imagen Seleccionada</label>
                                    <div class="control">
                                        <input class="input" type="text" id="selected-image" readonly placeholder="Selecciona una imagen de la lista">
                                    </div>
                                </div>
                                
                                <div id="selected-image-preview" class="mb-4" style="display:none;">
                                    <img id="preview-img" class="image-preview">
                                </div>
                                
                                <div id="selected-context-preview" class="mb-4" style="display:none;">
                                    <label class="label">Contexto Asociado</label>
                                    <pre class="context-preview" id="context-preview-content"></pre>
                                </div>
                            </div>
                            
                            <!-- Modo Batch -->
                            <div id="exec-batch" class="exec-tab-content" style="display:none;">
                                <div class="notification is-info is-light mb-3">
                                    <p class="is-size-7"><strong>Modo Batch:</strong> Selecciona múltiples imágenes de la lista para procesarlas con un pipeline.</p>
                                </div>
                                
                                <div class="field">
                                    <label class="label">Imágenes Seleccionadas (<span id="batch-count">0</span>)</label>
                                    <div id="batch-images-list" style="max-height: 150px; overflow-y: auto;">
                                        <p class="has-text-grey is-size-7">Haz click en las imágenes de la lista para agregarlas</p>
                                    </div>
                                </div>
                                
                                <button class="button is-small is-warning mb-3" id="btn-clear-batch">
                                    <i class="fas fa-times mr-1"></i>Limpiar Selección
                                </button>
                            </div>
                            
                            <div class="field">
                                <label class="label">Pipeline a Ejecutar</label>
                                <div id="pipeline-checkboxes"></div>
                            </div>
                            
                            <button class="button is-success is-fullwidth is-medium" id="btn-execute">
                                <i class="fas fa-rocket mr-2"></i>Ejecutar Comparación
                            </button>
                            
                            <button class="button is-info is-fullwidth is-medium mt-2" id="btn-execute-batch" style="display:none;">
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
                            
                            <!-- Parámetros Avanzados -->
                            <div class="field">
                                <label class="label">
                                    <a onclick="document.getElementById('advanced-params').classList.toggle('is-hidden')" style="cursor:pointer;">
                                        <i class="fas fa-cog mr-1"></i>Parámetros Avanzados <i class="fas fa-chevron-down is-size-7"></i>
                                    </a>
                                </label>
                            </div>
                            
                            <div id="advanced-params" class="is-hidden">
                                <div class="notification is-light is-info is-size-7 mb-3">
                                    <p><strong>top_p:</strong> Nucleus sampling (0.0-1.0). Alternativa a temperature.</p>
                                    <p><strong>top_k:</strong> Limita tokens considerados (Anthropic/Ollama).</p>
                                    <p><strong>frequency_penalty:</strong> Penaliza repetición de tokens (-2.0 a 2.0, OpenAI).</p>
                                    <p><strong>presence_penalty:</strong> Penaliza temas ya mencionados (-2.0 a 2.0, OpenAI).</p>
                                    <p><strong>repeat_penalty:</strong> Penaliza repeticiones (Ollama, >1.0).</p>
                                    <p><strong>seed:</strong> Para resultados reproducibles.</p>
                                </div>
                                
                                <div class="columns">
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">top_p</label>
                                            <input class="input is-small" type="number" id="config-top-p" placeholder="0.9" step="0.05" min="0" max="1">
                                        </div>
                                    </div>
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">top_k</label>
                                            <input class="input is-small" type="number" id="config-top-k" placeholder="40" min="1">
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="columns">
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">frequency_penalty</label>
                                            <input class="input is-small" type="number" id="config-frequency-penalty" placeholder="0" step="0.1" min="-2" max="2">
                                        </div>
                                    </div>
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">presence_penalty</label>
                                            <input class="input is-small" type="number" id="config-presence-penalty" placeholder="0" step="0.1" min="-2" max="2">
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="columns">
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">repeat_penalty (Ollama)</label>
                                            <input class="input is-small" type="number" id="config-repeat-penalty" placeholder="1.1" step="0.1" min="0">
                                        </div>
                                    </div>
                                    <div class="column">
                                        <div class="field">
                                            <label class="label is-small">seed</label>
                                            <input class="input is-small" type="number" id="config-seed" placeholder="">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <input type="hidden" id="config-edit-mode" value="">
                            
                            <div class="buttons">
                                <button class="button is-primary is-expanded" id="btn-save-config">
                                    <i class="fas fa-save mr-2"></i><span id="btn-save-config-text">Guardar Configuración</span>
                                </button>
                                <button class="button is-light" id="btn-cancel-edit" style="display:none;">
                                    <i class="fas fa-times mr-2"></i>Cancelar
                                </button>
                            </div>
                            
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

        <!-- Tab: Búsqueda Semántica -->
        <div id="tab-search" class="tab-content">
            <div class="columns">
                <div class="column is-8">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-search mr-2"></i>Búsqueda Semántica de Imágenes
                            </p>
                        </header>
                        <div class="card-content">
                            <div class="field has-addons">
                                <div class="control is-expanded">
                                    <input class="input" type="text" id="search-query" 
                                        placeholder="Ej: vehículo rojo con parachoque dañado, conductor con ropa verde...">
                                </div>
                                <div class="control">
                                    <button class="button is-primary" id="btn-search">
                                        <i class="fas fa-search mr-1"></i>Buscar
                                    </button>
                                </div>
                            </div>
                            <div class="field">
                                <label class="label is-small">Resultados máximos</label>
                                <div class="control">
                                    <input class="input is-small" type="number" id="search-top-k" value="20" min="1" max="100" style="width:100px">
                                </div>
                            </div>
                            <div id="search-results" class="mt-4"></div>
                        </div>
                    </div>
                </div>
                <div class="column is-4">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-database mr-2"></i>Índice Semántico
                            </p>
                        </header>
                        <div class="card-content" id="search-stats-content">
                            <p class="has-text-grey">Cargando estadísticas...</p>
                        </div>
                        <footer class="card-footer">
                            <a class="card-footer-item" id="btn-reindex">
                                <i class="fas fa-sync mr-1"></i>Reindexar Todo
                            </a>
                        </footer>
                    </div>
                </div>
            </div>
            <!-- Modal de detalle de imagen -->
            <div class="modal" id="image-detail-modal">
                <div class="modal-background"></div>
                <div class="modal-card" style="width:90%;max-width:1200px;max-height:90vh">
                    <header class="modal-card-head">
                        <p class="modal-card-title" id="modal-image-title">Detalle de Imagen</p>
                        <button class="delete" aria-label="close" id="btn-close-image-modal"></button>
                    </header>
                    <section class="modal-card-body" id="modal-image-body">
                    </section>
                </div>
            </div>
        </div>

        <!-- Tab: Agente Visual -->
        <div id="tab-agent" class="tab-content">
            <div class="columns">
                <div class="column is-4">
                    <div class="card">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-plus mr-2"></i>Nueva Sesión
                            </p>
                        </header>
                        <div class="card-content">
                            <div class="field">
                                <label class="label">Título (opcional)</label>
                                <input class="input" type="text" id="agent-session-title" placeholder="Análisis de vehículo...">
                            </div>
                            <div class="field">
                                <label class="label">Imagen</label>
                                <div class="select is-fullwidth">
                                    <select id="agent-image-select">
                                        <option value="">Seleccionar imagen...</option>
                                    </select>
                                </div>
                                <p class="help">O busca una imagen en la base de datos semántica</p>
                                <div class="field has-addons mt-2">
                                    <div class="control is-expanded">
                                        <input class="input is-small" type="text" id="agent-search-image" placeholder="Buscar imagen...">
                                    </div>
                                    <div class="control">
                                        <button class="button is-small is-info" id="btn-agent-search-image">
                                            <i class="fas fa-search"></i>
                                        </button>
                                    </div>
                                </div>
                                <div id="agent-search-results" class="mt-2" style="max-height:200px;overflow-y:auto"></div>
                            </div>
                            <div class="field">
                                <label class="label">Modelo LLM</label>
                                <div class="select is-fullwidth">
                                    <select id="agent-llm-select">
                                        <option value="">Seleccionar modelo...</option>
                                    </select>
                                </div>
                            </div>
                            <button class="button is-primary is-fullwidth" id="btn-create-agent-session">
                                <i class="fas fa-robot mr-2"></i>Iniciar Sesión
                            </button>
                        </div>
                    </div>
                    <div class="card mt-4">
                        <header class="card-header">
                            <p class="card-header-title">
                                <i class="fas fa-list mr-2"></i>Sesiones
                            </p>
                        </header>
                        <div class="card-content" id="agent-sessions-list" style="max-height:400px;overflow-y:auto">
                            <p class="has-text-grey">Sin sesiones</p>
                        </div>
                    </div>
                </div>
                <div class="column is-8">
                    <div class="card" id="agent-chat-card" style="display:none">
                        <header class="card-header">
                            <p class="card-header-title" id="agent-chat-title">
                                <i class="fas fa-comments mr-2"></i>Chat con Agente Visual
                            </p>
                            <div class="card-header-icon">
                                <button class="button is-small is-warning" id="btn-agent-pdf" title="Generar PDF">
                                    <i class="fas fa-file-pdf"></i>
                                </button>
                            </div>
                        </header>
                        <div class="card-content" id="agent-chat-messages" style="height:600px;overflow-y:auto;background:#f5f6fa;border-radius:8px;padding:1.5rem;display:flex;flex-direction:column;gap:0.75rem">
                        </div>
                        <footer class="card-footer" style="padding:1rem">
                            <div class="field has-addons" style="width:100%">
                                <div class="control is-expanded">
                                    <input class="input" type="text" id="agent-message-input" 
                                        placeholder="Escribe tu mensaje... (ej: recorta el vehículo de la imagen, ampliar la patente)">
                                </div>
                                <div class="control">
                                    <button class="button is-primary" id="btn-send-agent-message">
                                        <i class="fas fa-paper-plane"></i>
                                    </button>
                                </div>
                            </div>
                        </footer>
                    </div>
                    <div id="agent-placeholder" class="has-text-centered" style="padding:4rem">
                        <i class="fas fa-robot" style="font-size:4rem;color:#ccc"></i>
                        <p class="mt-4 has-text-grey">Selecciona o crea una sesión para comenzar a interactuar con el Agente Visual</p>
                        <p class="has-text-grey is-size-7 mt-2">El agente puede analizar, recortar, ampliar y procesar imágenes ejecutando código Python</p>
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
        let execMode = 'single'; // 'single' o 'batch'
        let batchSelectedImages = []; // Imágenes seleccionadas para batch
        let allImagesData = []; // Cache de todas las imágenes

        // ==================== Tab Navigation ====================
        document.querySelectorAll('.tabs.is-boxed li[data-tab]').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.tab;
                if (!tabId) return; // Ignorar si no tiene data-tab
                document.querySelectorAll('.tabs.is-boxed li[data-tab]').forEach(t => t.classList.remove('is-active'));
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
                if (tabId === 'search') loadSearchStats();
                if (tabId === 'agent') { loadAgentSessions(); loadAgentImages(); loadAgentLLMs(); }
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
        
        // Execution mode tab switching (single/batch)
        document.querySelectorAll('[data-exec-tab]').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const tabId = tab.dataset.execTab;
                execMode = tabId;
                
                document.querySelectorAll('[data-exec-tab]').forEach(t => t.classList.remove('is-active'));
                tab.classList.add('is-active');
                document.getElementById('exec-single').style.display = tabId === 'single' ? 'block' : 'none';
                document.getElementById('exec-batch').style.display = tabId === 'batch' ? 'block' : 'none';
                
                // Mostrar/ocultar botones según modo
                document.getElementById('btn-execute').style.display = tabId === 'single' ? 'block' : 'none';
                document.getElementById('btn-execute-batch').style.display = tabId === 'batch' ? 'block' : 'none';
                
                // Actualizar lista de imágenes para mostrar checkboxes en modo batch
                loadImages();
            });
        });
        
        // Limpiar selección batch
        document.getElementById('btn-clear-batch').addEventListener('click', () => {
            batchSelectedImages = [];
            updateBatchSelectionUI();
            loadImages();
        });

        // ==================== Configs ====================
        async function loadConfigs() {
            const res = await fetch('/api/configs');
            const configs = await res.json();
            
            const list = document.getElementById('configs-list');
            const select = document.getElementById('pipeline-default-llm');
            
            list.innerHTML = configs.map(c => {
                const extraParams = c.extra_params || {};
                const extraInfo = Object.keys(extraParams).length > 0 
                    ? `<br><span class="tag is-light is-small">Extra: ${Object.keys(extraParams).join(', ')}</span>` 
                    : '';
                
                return `
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
                                    ${extraInfo}
                                </p>
                            </div>
                        </div>
                        <div class="level-right">
                            <div class="buttons are-small">
                                <button class="button is-info is-outlined" onclick="editConfig('${c.name}')">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button class="button is-danger is-outlined" onclick="deleteConfig('${c.name}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `}).join('') || '<p class="has-text-grey">No hay configuraciones guardadas</p>';
            
            select.innerHTML = '<option value="">Seleccionar...</option>' + 
                configs.map(c => `<option value="${c.name}">${c.name} (${c.model})</option>`).join('');
        }
        
        async function editConfig(name) {
            const res = await fetch('/api/configs/' + encodeURIComponent(name));
            if (!res.ok) {
                alert('Error al cargar la configuración');
                return;
            }
            const config = await res.json();
            
            // Llenar el formulario
            document.getElementById('config-name').value = config.name;
            document.getElementById('config-provider').value = config.provider;
            document.getElementById('config-model').value = config.model;
            document.getElementById('config-api-key').value = config.api_key || '';
            document.getElementById('config-api-base').value = config.api_base || '';
            document.getElementById('config-max-tokens').value = config.max_tokens || 4096;
            document.getElementById('config-temperature').value = config.temperature || 0.7;
            
            // Parámetros avanzados
            const extra = config.extra_params || {};
            document.getElementById('config-top-p').value = extra.top_p || '';
            document.getElementById('config-top-k').value = extra.top_k || '';
            document.getElementById('config-frequency-penalty').value = extra.frequency_penalty || '';
            document.getElementById('config-presence-penalty').value = extra.presence_penalty || '';
            document.getElementById('config-repeat-penalty').value = extra.repeat_penalty || '';
            document.getElementById('config-seed').value = extra.seed || '';
            
            // Mostrar parámetros avanzados si hay alguno
            if (Object.keys(extra).length > 0) {
                document.getElementById('advanced-params').classList.remove('is-hidden');
            }
            
            // Modo edición
            document.getElementById('config-edit-mode').value = name;
            document.getElementById('btn-save-config-text').textContent = 'Actualizar Configuración';
            document.getElementById('btn-cancel-edit').style.display = 'inline-flex';
            
            // Scroll al formulario
            document.getElementById('config-name').scrollIntoView({ behavior: 'smooth' });
        }
        
        function clearConfigForm() {
            document.getElementById('config-name').value = '';
            document.getElementById('config-provider').value = 'openai';
            document.getElementById('config-model').value = '';
            document.getElementById('config-api-key').value = '';
            document.getElementById('config-api-base').value = '';
            document.getElementById('config-max-tokens').value = '4096';
            document.getElementById('config-temperature').value = '0.7';
            document.getElementById('config-top-p').value = '';
            document.getElementById('config-top-k').value = '';
            document.getElementById('config-frequency-penalty').value = '';
            document.getElementById('config-presence-penalty').value = '';
            document.getElementById('config-repeat-penalty').value = '';
            document.getElementById('config-seed').value = '';
            document.getElementById('config-edit-mode').value = '';
            document.getElementById('btn-save-config-text').textContent = 'Guardar Configuración';
            document.getElementById('btn-cancel-edit').style.display = 'none';
            document.getElementById('advanced-params').classList.add('is-hidden');
        }
        
        document.getElementById('btn-cancel-edit').addEventListener('click', clearConfigForm);

        document.getElementById('btn-save-config').addEventListener('click', async () => {
            // Construir extra_params solo con valores no vacíos
            const extra_params = {};
            const topP = document.getElementById('config-top-p').value;
            const topK = document.getElementById('config-top-k').value;
            const freqPenalty = document.getElementById('config-frequency-penalty').value;
            const presPenalty = document.getElementById('config-presence-penalty').value;
            const repeatPenalty = document.getElementById('config-repeat-penalty').value;
            const seed = document.getElementById('config-seed').value;
            
            if (topP) extra_params.top_p = parseFloat(topP);
            if (topK) extra_params.top_k = parseInt(topK);
            if (freqPenalty) extra_params.frequency_penalty = parseFloat(freqPenalty);
            if (presPenalty) extra_params.presence_penalty = parseFloat(presPenalty);
            if (repeatPenalty) extra_params.repeat_penalty = parseFloat(repeatPenalty);
            if (seed) extra_params.seed = parseInt(seed);
            
            const config = {
                name: document.getElementById('config-name').value,
                provider: document.getElementById('config-provider').value,
                model: document.getElementById('config-model').value,
                api_key: document.getElementById('config-api-key').value,
                api_base: document.getElementById('config-api-base').value,
                max_tokens: parseInt(document.getElementById('config-max-tokens').value) || 4096,
                temperature: parseFloat(document.getElementById('config-temperature').value) || 0.7,
                extra_params: extra_params
            };
            
            if (!config.name) {
                alert('El nombre es requerido');
                return;
            }
            
            const editMode = document.getElementById('config-edit-mode').value;
            
            if (editMode) {
                // Modo edición - usar PUT
                await fetch('/api/configs/' + encodeURIComponent(editMode), {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
            } else {
                // Modo creación - usar POST
                await fetch('/api/configs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
            }
            
            loadConfigs();
            clearConfigForm();
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
                    <div class="field">
                        <label class="checkbox">
                            <input type="checkbox" ${step.use_previous_output ? 'checked' : ''} 
                                onchange="updateStep(${i}, 'use_previous_output', this.checked)">
                            Usar salida del paso anterior como contexto
                        </label>
                    </div>
                    <div class="field">
                        <label class="checkbox">
                            <input type="checkbox" ${step.index_for_search ? 'checked' : ''} 
                                onchange="updateStep(${i}, 'index_for_search', this.checked)">
                            <i class="fas fa-search mr-1"></i> Indexar para búsqueda semántica
                        </label>
                        <p class="help">La salida de este paso se usará para indexar la imagen en la búsqueda semántica</p>
                    </div>
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
            pipelineSteps.push({name: '', prompt: '', use_previous_output: true, llm_config_name: null, index_for_search: false});
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
            const files = e.target.files;
            const count = files.length;
            document.getElementById('batch-file-name').textContent = count > 0 ? `${count} archivos seleccionados` : 'Ningún archivo';
            
            // Mostrar preview de archivos
            if (count > 0) {
                const previewDiv = document.getElementById('batch-preview');
                const listDiv = document.getElementById('batch-preview-list');
                previewDiv.style.display = 'block';
                
                // Separar imágenes y JSONs
                const images = [];
                const jsons = [];
                for (let file of files) {
                    if (file.name.toLowerCase().endsWith('.json')) {
                        jsons.push(file.name);
                    } else {
                        images.push(file.name);
                    }
                }
                
                // Mostrar pares imagen-json
                let html = '<table class="table is-narrow is-fullwidth is-size-7"><thead><tr><th>Imagen</th><th>JSON</th></tr></thead><tbody>';
                for (let img of images) {
                    const imgStem = img.substring(0, img.lastIndexOf('.'));
                    const matchingJson = jsons.find(j => j.substring(0, j.lastIndexOf('.')) === imgStem);
                    html += `<tr>
                        <td>${img}</td>
                        <td>${matchingJson ? '<span class="tag is-success is-light">' + matchingJson + '</span>' : '<span class="tag is-warning is-light">Sin JSON</span>'}</td>
                    </tr>`;
                }
                html += '</tbody></table>';
                listDiv.innerHTML = html;
            } else {
                document.getElementById('batch-preview').style.display = 'none';
            }
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
            
            // Mostrar progreso
            const progressDiv = document.getElementById('batch-upload-progress');
            const progressBar = document.getElementById('batch-progress-bar');
            const progressText = document.getElementById('batch-progress-text');
            const uploadBtn = document.getElementById('btn-upload-batch');
            
            progressDiv.style.display = 'block';
            uploadBtn.disabled = true;
            uploadBtn.classList.add('is-loading');
            progressText.textContent = 'Preparando archivos...';
            progressBar.value = 10;
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            try {
                progressText.textContent = `Subiendo ${files.length} archivos...`;
                progressBar.value = 30;
                
                const response = await fetch('/api/images/upload-batch', {method: 'POST', body: formData});
                const result = await response.json();
                
                progressBar.value = 100;
                progressText.textContent = `✓ ${result.images.length} imágenes subidas correctamente`;
                
                // Limpiar después de 2 segundos
                setTimeout(() => {
                    progressDiv.style.display = 'none';
                    document.getElementById('batch-preview').style.display = 'none';
                }, 2000);
                
                loadImages();
            } catch (error) {
                progressText.textContent = 'Error al subir archivos';
                progressBar.classList.add('is-danger');
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.classList.remove('is-loading');
                document.getElementById('batch-files').value = '';
                document.getElementById('batch-file-name').textContent = 'Ningún archivo';
            }
        });

        async function loadImages() {
            const res = await fetch('/api/images');
            const images = await res.json();
            allImagesData = images; // Cache para uso posterior
            
            if (execMode === 'batch') {
                // Modo batch: mostrar checkboxes
                document.getElementById('images-list').innerHTML = images.map(img => {
                    const isSelected = batchSelectedImages.some(b => b.name === img.name);
                    return `
                    <div class="box is-clickable ${isSelected ? 'has-background-success-light' : ''}" 
                         onclick="toggleBatchImage('${img.name}', ${img.has_context})">
                        <div class="level">
                            <div class="level-left">
                                <div>
                                    <label class="checkbox">
                                        <input type="checkbox" ${isSelected ? 'checked' : ''} onclick="event.stopPropagation(); toggleBatchImage('${img.name}', ${img.has_context})">
                                        <strong class="ml-2">${img.name}</strong>
                                    </label>
                                    <p class="is-size-7 has-text-grey ml-4">
                                        ${(img.size / 1024).toFixed(1)} KB
                                        ${img.has_context ? '<span class="tag is-success is-light ml-2">JSON</span>' : ''}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                `}).join('') || '<p class="has-text-grey">No hay imágenes</p>';
            } else {
                // Modo individual: click para seleccionar
                document.getElementById('images-list').innerHTML = images.map(img => `
                    <div class="box is-clickable ${selectedImage === img.name ? 'has-background-primary-light' : ''}" 
                         onclick="selectImage('${img.name}', ${img.has_context})">
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
        }
        
        // Función para toggle de imagen en modo batch
        async function toggleBatchImage(name, hasContext) {
            const existingIndex = batchSelectedImages.findIndex(b => b.name === name);
            
            if (existingIndex >= 0) {
                // Remover de la selección
                batchSelectedImages.splice(existingIndex, 1);
            } else {
                // Agregar a la selección
                let contextData = null;
                if (hasContext) {
                    const res = await fetch('/api/images/' + name + '/context');
                    const data = await res.json();
                    contextData = data.context_data;
                }
                batchSelectedImages.push({ name, has_context: hasContext, context_data: contextData });
            }
            
            updateBatchSelectionUI();
            loadImages(); // Refrescar para actualizar visual
        }
        
        // Actualizar UI de selección batch
        function updateBatchSelectionUI() {
            document.getElementById('batch-count').textContent = batchSelectedImages.length;
            
            const listDiv = document.getElementById('batch-images-list');
            if (batchSelectedImages.length === 0) {
                listDiv.innerHTML = '<p class="has-text-grey is-size-7">Haz click en las imágenes de la lista para agregarlas</p>';
            } else {
                listDiv.innerHTML = batchSelectedImages.map(img => `
                    <div class="tag is-medium is-primary is-light mr-1 mb-1">
                        ${img.name}
                        ${img.has_context ? '<span class="tag is-success is-small ml-1">JSON</span>' : ''}
                        <button class="delete is-small" onclick="removeBatchImage('${img.name}')"></button>
                    </div>
                `).join('');
            }
        }
        
        // Remover imagen de selección batch
        function removeBatchImage(name) {
            batchSelectedImages = batchSelectedImages.filter(b => b.name !== name);
            updateBatchSelectionUI();
            loadImages();
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
        
        // Ejecutar batch
        document.getElementById('btn-execute-batch').addEventListener('click', async () => {
            if (batchSelectedImages.length === 0) return alert('Selecciona al menos una imagen');
            
            const selectedPipelines = Array.from(document.querySelectorAll('.pipeline-checkbox:checked'))
                .map(cb => cb.value);
            
            if (selectedPipelines.length === 0) return alert('Selecciona un pipeline');
            if (selectedPipelines.length > 1) return alert('En modo batch solo puedes seleccionar un pipeline');
            
            const pipelineId = selectedPipelines[0];
            
            document.getElementById('results-card').style.display = 'block';
            document.getElementById('execution-progress').style.display = 'block';
            document.getElementById('execution-results').innerHTML = '';
            document.getElementById('progress-text').textContent = `Procesando ${batchSelectedImages.length} imágenes...`;
            
            // Preparar datos de imágenes para el batch
            const imagesData = batchSelectedImages.map(img => ({
                image_name: img.name,
                context_data: img.context_data
            }));
            
            const res = await fetch('/api/execute-batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    pipeline_id: pipelineId,
                    images: imagesData
                })
            });
            
            const {execution_id} = await res.json();
            pollBatchExecutionStatus(execution_id);
        });
        
        // Poll para ejecución batch
        async function pollBatchExecutionStatus(executionId) {
            const interval = setInterval(async () => {
                const res = await fetch('/api/execute/' + executionId + '/status');
                const status = await res.json();
                
                document.getElementById('progress-text').textContent = status.progress || 'Procesando batch...';
                
                if (status.status === 'completed') {
                    clearInterval(interval);
                    document.getElementById('execution-progress').style.display = 'none';
                    renderBatchResults(status.results);
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
        
        // Renderizar resultados de batch
        function renderBatchResults(batchResult) {
            if (!batchResult) {
                document.getElementById('execution-results').innerHTML = '<p class="has-text-grey">Sin resultados</p>';
                return;
            }
            
            let html = `
                <div class="notification is-info is-light mb-4">
                    <p><strong>Resumen Batch:</strong></p>
                    <p>Pipeline: <strong>${batchResult.pipeline_name || 'N/A'}</strong></p>
                    <p>Imágenes procesadas: <strong>${batchResult.total_images || 0}</strong></p>
                    <p>Exitosas: <strong class="has-text-success">${batchResult.successful || 0}</strong> | 
                       Fallidas: <strong class="has-text-danger">${batchResult.failed || 0}</strong></p>
                    <p>Tiempo total: <strong>${((batchResult.total_time_ms || 0) / 1000).toFixed(2)}s</strong></p>
                    <p>Tokens totales: <strong>${batchResult.total_tokens || 0}</strong></p>
                    <p>Costo total: <strong>$${(batchResult.total_cost || 0).toFixed(4)}</strong></p>
                </div>
            `;
            
            // Mostrar resultados individuales
            if (batchResult.results && batchResult.results.length > 0) {
                html += '<h4 class="title is-5 mt-4">Resultados por Imagen</h4>';
                
                for (const result of batchResult.results) {
                    const statusClass = result.success ? 'success' : 'error';
                    const statusIcon = result.success ? 'fa-check-circle has-text-success' : 'fa-times-circle has-text-danger';
                    
                    html += `
                        <div class="box execution-card ${statusClass} mb-3">
                            <div class="level">
                                <div class="level-left">
                                    <div>
                                        <p class="title is-6">
                                            <i class="fas ${statusIcon} mr-2"></i>
                                            ${result.image_name || 'Imagen'}
                                        </p>
                                    </div>
                                </div>
                                <div class="level-right">
                                    <div class="has-text-right">
                                        <p><strong>${(result.total_latency_ms || 0).toFixed(0)}ms</strong></p>
                                        <p class="is-size-7">${result.total_tokens || 0} tokens | $${(result.total_cost || 0).toFixed(4)}</p>
                                    </div>
                                </div>
                            </div>
                            
                            ${result.error ? `<div class="notification is-danger is-light is-size-7">${result.error}</div>` : ''}
                            
                            ${result.step_results ? `
                                <details>
                                    <summary class="is-size-7 has-text-grey">Ver ${result.step_results.length} pasos</summary>
                                    <div class="mt-2">
                                        ${result.step_results.map((step, i) => `
                                            <div class="step-result-card">
                                                <div class="step-header">
                                                    <div>
                                                        <span class="tag is-primary is-small">Paso ${i+1}</span>
                                                        <strong class="ml-2">${step.step_name}</strong>
                                                    </div>
                                                    <div class="is-size-7">
                                                        ${(step.latency_ms || 0).toFixed(0)}ms | ${step.total_tokens || 0} tokens
                                                    </div>
                                                </div>
                                                <div class="content mt-2">
                                                    <pre style="white-space: pre-wrap; background: #f5f5f5; padding: 0.5rem; border-radius: 4px; font-size: 0.75rem; max-height: 150px; overflow-y: auto;">${escapeHtml(step.content || '')}</pre>
                                                </div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </details>
                            ` : ''}
                        </div>
                    `;
                }
            }
            
            document.getElementById('execution-results').innerHTML = html;
        }

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

        // ==================== Forensic PDF ====================
        async function downloadForensicPDF(executionId) {
            try {
                // Mostrar loading
                const btn = event.target.closest('button');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generando PDF...';
                btn.disabled = true;
                
                const response = await fetch(`/api/history/${executionId}/pdf`);
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Error al generar PDF');
                }
                
                // Obtener el hash del header para mostrar
                const reportHash = response.headers.get('X-Report-Hash');
                
                // Descargar el archivo
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `forensic_report_${executionId}.pdf`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                // Mostrar notificación con hash
                if (reportHash) {
                    alert(`PDF generado exitosamente.\n\nHash SHA-256:\n${reportHash}`);
                }
                
                btn.innerHTML = originalText;
                btn.disabled = false;
            } catch (error) {
                alert('Error: ' + error.message);
                const btn = event.target.closest('button');
                btn.innerHTML = '<i class="fas fa-file-pdf mr-2"></i>Descargar Reporte Forense PDF';
                btn.disabled = false;
            }
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
                        <div class="mt-3">
                            <button class="button is-info" onclick="downloadForensicPDF('${detail.id}')">
                                <i class="fas fa-file-pdf mr-2"></i>Descargar Reporte Forense PDF
                            </button>
                        </div>
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

        // ==================== Semantic Search ====================
        
        async function loadSearchStats() {
            try {
                const res = await fetch('/api/search/stats');
                const stats = await res.json();
                document.getElementById('search-stats-content').innerHTML = `
                    <div class="content">
                        <p><strong>Imágenes indexadas:</strong> ${stats.total_images || 0}</p>
                        <p><strong>Descripciones totales:</strong> ${stats.total_descriptions || 0}</p>
                        <p><strong>Atributos totales:</strong> ${stats.total_attributes || 0}</p>
                        <p><strong>Vectores FAISS:</strong> ${stats.faiss_vectors || 0}</p>
                        <p><strong>Dimensión vectorial:</strong> ${stats.vector_dimension || 'N/A'}</p>
                        <p><strong>Embeddings cargados:</strong> ${stats.embeddings_loaded ? 'Sí' : 'No'}</p>
                    </div>
                `;
            } catch(e) {
                document.getElementById('search-stats-content').innerHTML = '<p class="has-text-danger">Error al cargar estadísticas</p>';
            }
        }

        document.getElementById('btn-search').addEventListener('click', async () => {
            const query = document.getElementById('search-query').value.trim();
            if (!query) return;
            const topK = parseInt(document.getElementById('search-top-k').value) || 20;
            const btn = document.getElementById('btn-search');
            btn.classList.add('is-loading');
            try {
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, top_k: topK})
                });
                const data = await res.json();
                renderSearchResults(data.results, query);
            } catch(e) {
                document.getElementById('search-results').innerHTML = '<p class="has-text-danger">Error en la búsqueda</p>';
            } finally {
                btn.classList.remove('is-loading');
            }
        });

        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') document.getElementById('btn-search').click();
        });

        function renderSearchResults(results, query) {
            if (!results || results.length === 0) {
                document.getElementById('search-results').innerHTML = `
                    <div class="notification is-warning is-light">
                        <i class="fas fa-info-circle mr-2"></i>No se encontraron resultados para: <strong>${query}</strong>
                    </div>`;
                return;
            }
            const html = `
                <p class="mb-3"><strong>${results.length}</strong> resultados para: <em>${query}</em></p>
                <div class="columns is-multiline">
                    ${results.map((r, i) => `
                        <div class="column is-4">
                            <div class="card is-clickable" onclick="showImageDetail('${r.image_name}')" style="height:100%">
                                <div class="card-image">
                                    <figure class="image is-4by3">
                                        <img src="/api/images/view/${r.image_name}" alt="${r.image_name}" 
                                            style="object-fit:cover;width:100%;height:100%">
                                    </figure>
                                </div>
                                <div class="card-content" style="padding:0.75rem">
                                    <p class="is-size-7 has-text-weight-bold">${r.image_name}</p>
                                    <div class="level is-mobile mt-1">
                                        <div class="level-left">
                                            <span class="tag is-primary is-light">
                                                <i class="fas fa-percentage mr-1"></i>${(r.score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div class="level-right">
                                            <span class="tag is-info is-light is-size-7">${r.pipeline_name || ''}</span>
                                        </div>
                                    </div>
                                    <p class="is-size-7 has-text-grey mt-1" style="max-height:60px;overflow:hidden">
                                        ${(r.combined_text || r.description || '').substring(0, 150)}...
                                    </p>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>`;
            document.getElementById('search-results').innerHTML = html;
        }

        async function showImageDetail(imageName) {
            const modal = document.getElementById('image-detail-modal');
            modal.classList.add('is-active');
            document.getElementById('modal-image-title').textContent = imageName;
            document.getElementById('modal-image-body').innerHTML = '<p>Cargando...</p>';
            try {
                const res = await fetch(`/api/search/image-details/${imageName}`);
                const details = await res.json();
                let contextHtml = '';
                if (details.metadata && details.metadata.context_data) {
                    const ctx = details.metadata.context_data;
                    contextHtml = `
                        <div class="box">
                            <h5 class="title is-6"><i class="fas fa-database mr-1"></i>Datos de Contexto</h5>
                            <table class="table is-fullwidth is-striped is-size-7">
                                <tbody>
                                    ${Object.entries(ctx).map(([k,v]) => 
                                        `<tr><td><strong>${k}</strong></td><td>${typeof v === 'object' ? JSON.stringify(v) : v}</td></tr>`
                                    ).join('')}
                                </tbody>
                            </table>
                        </div>`;
                }
                document.getElementById('modal-image-body').innerHTML = `
                    <div class="columns">
                        <div class="column is-5">
                            <figure class="image">
                                <img src="/api/images/view/${imageName}" alt="${imageName}" style="border-radius:8px;max-height:500px;object-fit:contain">
                            </figure>
                        </div>
                        <div class="column is-7">
                            ${contextHtml}
                            <div class="box">
                                <h5 class="title is-6"><i class="fas fa-file-alt mr-1"></i>Descripción Indexada</h5>
                                <div class="content is-size-7" style="max-height:400px;overflow-y:auto;white-space:pre-wrap">${details.description || 'Sin descripción'}</div>
                            </div>
                        </div>
                    </div>`;
            } catch(e) {
                document.getElementById('modal-image-body').innerHTML = `<p class="has-text-danger">Error al cargar detalles: ${e.message}</p>`;
            }
        }

        document.getElementById('btn-close-image-modal').addEventListener('click', () => {
            document.getElementById('image-detail-modal').classList.remove('is-active');
        });
        document.querySelector('#image-detail-modal .modal-background').addEventListener('click', () => {
            document.getElementById('image-detail-modal').classList.remove('is-active');
        });

        document.getElementById('btn-reindex').addEventListener('click', async () => {
            const btn = document.getElementById('btn-reindex');
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Reindexando...';
            try {
                const res = await fetch('/api/search/reindex', {method: 'POST'});
                const data = await res.json();
                btn.innerHTML = `<i class="fas fa-check mr-1"></i>${data.indexed} indexadas`;
                loadSearchStats();
                setTimeout(() => { btn.innerHTML = '<i class="fas fa-sync mr-1"></i>Reindexar Todo'; }, 3000);
            } catch(e) {
                btn.innerHTML = '<i class="fas fa-times mr-1"></i>Error';
                setTimeout(() => { btn.innerHTML = '<i class="fas fa-sync mr-1"></i>Reindexar Todo'; }, 3000);
            }
        });

        // ==================== Visual Agent ====================
        
        let currentAgentSessionId = null;

        async function loadAgentSessions() {
            try {
                const res = await fetch('/api/agent/sessions');
                const sessions = await res.json();
                const container = document.getElementById('agent-sessions-list');
                if (!sessions || sessions.length === 0) {
                    container.innerHTML = '<p class="has-text-grey">Sin sesiones</p>';
                    return;
                }
                container.innerHTML = sessions.map(s => `
                    <div class="box is-clickable p-3 mb-2 ${currentAgentSessionId === s.id ? 'has-background-primary-light' : ''}" 
                         onclick="openAgentSession('${s.id}')">
                        <div class="level is-mobile">
                            <div class="level-left">
                                <div>
                                    <p class="is-size-7 has-text-weight-bold">${s.title || s.image_name}</p>
                                    <p class="is-size-7 has-text-grey">${s.llm_config_name} | ${s.message_count || 0} msgs</p>
                                </div>
                            </div>
                            <div class="level-right">
                                <button class="delete is-small" onclick="event.stopPropagation(); deleteAgentSession('${s.id}')"></button>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch(e) {
                console.error('Error loading agent sessions:', e);
            }
        }

        async function loadAgentImages() {
            try {
                const res = await fetch('/api/images');
                const images = await res.json();
                const select = document.getElementById('agent-image-select');
                select.innerHTML = '<option value="">Seleccionar imagen...</option>' +
                    images.map(img => `<option value="${img.name}">${img.name}</option>`).join('');
            } catch(e) {}
        }

        async function loadAgentLLMs() {
            try {
                const res = await fetch('/api/configs');
                const configs = await res.json();
                const select = document.getElementById('agent-llm-select');
                select.innerHTML = '<option value="">Seleccionar modelo...</option>' +
                    configs.map(c => `<option value="${c.name}">${c.name} (${c.provider}/${c.model})</option>`).join('');
            } catch(e) {}
        }

        document.getElementById('btn-create-agent-session').addEventListener('click', async () => {
            const imageName = document.getElementById('agent-image-select').value;
            const llmName = document.getElementById('agent-llm-select').value;
            const title = document.getElementById('agent-session-title').value;
            if (!imageName || !llmName) {
                alert('Selecciona una imagen y un modelo LLM');
                return;
            }
            const btn = document.getElementById('btn-create-agent-session');
            btn.classList.add('is-loading');
            try {
                const res = await fetch('/api/agent/sessions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image_name: imageName, llm_config_name: llmName, title})
                });
                const data = await res.json();
                currentAgentSessionId = data.session.id;
                loadAgentSessions();
                openAgentSession(data.session.id);
            } catch(e) {
                alert('Error al crear sesión: ' + e.message);
            } finally {
                btn.classList.remove('is-loading');
            }
        });

        async function openAgentSession(sessionId) {
            currentAgentSessionId = sessionId;
            document.getElementById('agent-chat-card').style.display = 'block';
            document.getElementById('agent-placeholder').style.display = 'none';
            loadAgentSessions();
            try {
                const res = await fetch(`/api/agent/sessions/${sessionId}`);
                const session = await res.json();
                document.getElementById('agent-chat-title').innerHTML = 
                    `<i class="fas fa-comments mr-2"></i>${session.title || session.image_name} <span class="tag is-info ml-2">${session.llm_config_name}</span>`;
                renderAgentMessages(session.messages || []);
            } catch(e) {
                console.error('Error opening session:', e);
            }
        }

        // Configure marked.js for rendering
        (function() {
            if (typeof marked !== 'undefined') {
                marked.setOptions({
                    breaks: true,
                    gfm: true,
                    highlight: function(code, lang) {
                        if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                            try { return hljs.highlight(code, {language: lang}).value; } catch(e) {}
                        }
                        if (typeof hljs !== 'undefined') {
                            try { return hljs.highlightAuto(code).value; } catch(e) {}
                        }
                        return code;
                    }
                });
            }
        })();

        function renderMarkdown(text) {
            if (!text) return '';
            if (typeof marked !== 'undefined') {
                try { return marked.parse(text); } catch(e) { console.error('Markdown parse error:', e); }
            }
            // Fallback: basic escaping with pre-wrap
            return '<div style="white-space:pre-wrap">' + text.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
        }

        function renderAgentMessageParts(msg) {
            // If message has structured parts, render them
            if (msg.parts && msg.parts.length > 0) {
                return msg.parts.map(part => {
                    switch(part.type) {
                        case 'plan':
                            return `<div class="agent-plan"><div class="md-content">${renderMarkdown(part.content)}</div></div>`;
                        
                        case 'code':
                            let codeHtml = `<div class="agent-code-block">`;
                            codeHtml += `<div class="agent-code-header"><span><i class="fas fa-code mr-1"></i> Código ejecutado</span><span class="lang-badge">${part.language || 'python'}</span></div>`;
                            // Render code with syntax highlighting
                            let highlightedCode = part.content;
                            if (typeof hljs !== 'undefined') {
                                try { highlightedCode = hljs.highlight(part.content, {language: part.language || 'python'}).value; } catch(e) {}
                            }
                            codeHtml += `<pre style="margin:0;background:#1e1e2e;padding:0"><code class="hljs" style="padding:0.75rem;display:block;overflow-x:auto">${highlightedCode}</code></pre>`;
                            // Show output if any
                            if (part.output) {
                                const outputClass = part.success !== false ? 'success' : 'error';
                                const icon = part.success !== false ? 'fa-check-circle has-text-success' : 'fa-times-circle has-text-danger';
                                codeHtml += `<div class="agent-code-output ${outputClass}"><i class="fas ${icon} mr-1"></i>${part.output.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
                            }
                            codeHtml += `</div>`;
                            return codeHtml;
                        
                        case 'image':
                            const imgUrl = `/api/agent/outputs/${currentAgentSessionId}/${part.filename}`;
                            return `<div class="agent-image-output">
                                <img src="${imgUrl}" onclick="window.open(this.src,'_blank')" alt="${part.caption || part.filename}" loading="lazy">
                                <div class="agent-image-caption"><i class="fas fa-image mr-1"></i>${part.caption || part.filename}</div>
                            </div>`;
                        
                        case 'error':
                            return `<div class="notification is-danger is-light"><i class="fas fa-exclamation-triangle mr-2"></i>${renderMarkdown(part.content)}</div>`;
                        
                        case 'text':
                        default:
                            return `<div class="md-content">${renderMarkdown(part.content)}</div>`;
                    }
                }).join('');
            }
            
            // Fallback: render raw content as markdown + legacy images
            let html = `<div class="md-content">${renderMarkdown(msg.content || '')}</div>`;
            
            // Legacy: show images from images array
            if (msg.images && msg.images.length > 0) {
                html += msg.images.map(img => {
                    const filename = img.includes('/') ? img.split('/').pop() : img;
                    const imgUrl = `/api/agent/outputs/${currentAgentSessionId}/${filename}`;
                    return `<div class="agent-image-output">
                        <img src="${imgUrl}" onclick="window.open(this.src,'_blank')" alt="${filename}" loading="lazy">
                        <div class="agent-image-caption"><i class="fas fa-image mr-1"></i>${filename}</div>
                    </div>`;
                }).join('');
            }
            
            // Legacy: show code if present
            if (msg.code) {
                let highlightedCode = msg.code;
                if (typeof hljs !== 'undefined') {
                    try { highlightedCode = hljs.highlight(msg.code, {language: 'python'}).value; } catch(e) {}
                }
                html += `<div class="agent-code-block">
                    <div class="agent-code-header"><span><i class="fas fa-code mr-1"></i> Código ejecutado</span><span class="lang-badge">python</span></div>
                    <pre style="margin:0;background:#1e1e2e;padding:0"><code class="hljs" style="padding:0.75rem;display:block;overflow-x:auto">${highlightedCode}</code></pre>
                    ${msg.code_output ? `<div class="agent-code-output ${msg.code_success !== false ? 'success' : 'error'}">${msg.code_output.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>` : ''}
                </div>`;
            }
            
            return html;
        }

        function renderAgentMessages(messages) {
            const container = document.getElementById('agent-chat-messages');
            container.innerHTML = messages.map(msg => {
                if (msg.role === 'system') {
                    return `<div class="agent-msg agent-msg-system">
                        <i class="fas fa-info-circle mr-1"></i>${msg.content || ''}
                    </div>`;
                }
                
                const isUser = msg.role === 'user';
                const msgClass = isUser ? 'agent-msg-user' : 'agent-msg-assistant';
                const icon = isUser ? 'fa-user' : 'fa-robot';
                const label = isUser ? 'Tú' : 'Agente';
                
                const contentHtml = isUser 
                    ? `<div class="md-content">${renderMarkdown(msg.content || '')}</div>`
                    : renderAgentMessageParts(msg);
                
                return `<div class="agent-msg ${msgClass}">
                    <div class="agent-msg-meta">
                        <i class="fas ${icon}"></i>
                        <strong>${label}</strong>
                        <span>${msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}</span>
                    </div>
                    ${contentHtml}
                </div>`;
            }).join('');
            
            container.scrollTop = container.scrollHeight;
            
            // Apply syntax highlighting to any remaining code blocks
            if (typeof hljs !== 'undefined') {
                container.querySelectorAll('pre code:not(.hljs)').forEach(block => {
                    hljs.highlightElement(block);
                });
            }
        }

        document.getElementById('btn-send-agent-message').addEventListener('click', sendAgentMessage);
        document.getElementById('agent-message-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendAgentMessage();
        });

        async function sendAgentMessage() {
            if (!currentAgentSessionId) return;
            const input = document.getElementById('agent-message-input');
            const message = input.value.trim();
            if (!message) return;
            input.value = '';
            
            // Agregar mensaje del usuario al chat inmediatamente
            const container = document.getElementById('agent-chat-messages');
            const userMsgHtml = `<div class="agent-msg agent-msg-user">
                <div class="agent-msg-meta">
                    <i class="fas fa-user"></i>
                    <strong>Tú</strong>
                    <span>${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="md-content">${renderMarkdown(message)}</div>
            </div>`;
            container.innerHTML += userMsgHtml;
            
            // Mostrar indicador de carga con animación
            container.innerHTML += `
                <div id="agent-typing" class="agent-msg agent-msg-assistant" style="max-width:300px">
                    <div class="agent-msg-meta">
                        <i class="fas fa-robot"></i>
                        <strong>Agente</strong>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem">
                        <div class="loader" style="width:20px;height:20px;border:2px solid #ddd;border-top-color:#667eea;border-radius:50%;animation:spin 0.8s linear infinite"></div>
                        <span style="color:#888;font-size:0.9rem">Analizando y ejecutando código...</span>
                    </div>
                    <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
                </div>`;
            container.scrollTop = container.scrollHeight;
            
            const btn = document.getElementById('btn-send-agent-message');
            btn.disabled = true;
            
            try {
                const res = await fetch(`/api/agent/sessions/${currentAgentSessionId}/messages`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                const data = await res.json();
                // Recargar toda la sesión para mostrar la respuesta con formato completo
                openAgentSession(currentAgentSessionId);
            } catch(e) {
                const typing = document.getElementById('agent-typing');
                if (typing) typing.remove();
                container.innerHTML += `
                    <div class="agent-msg agent-msg-error">
                        <i class="fas fa-exclamation-triangle mr-2 has-text-danger"></i>
                        <strong>Error:</strong> ${e.message}
                    </div>`;
            } finally {
                btn.disabled = false;
            }
        }

        async function deleteAgentSession(sessionId) {
            if (!confirm('¿Eliminar esta sesión?')) return;
            try {
                await fetch(`/api/agent/sessions/${sessionId}`, {method: 'DELETE'});
                if (currentAgentSessionId === sessionId) {
                    currentAgentSessionId = null;
                    document.getElementById('agent-chat-card').style.display = 'none';
                    document.getElementById('agent-placeholder').style.display = 'block';
                }
                loadAgentSessions();
            } catch(e) {}
        }

        document.getElementById('btn-agent-pdf').addEventListener('click', async () => {
            if (!currentAgentSessionId) return;
            window.open(`/api/agent/sessions/${currentAgentSessionId}/pdf`, '_blank');
        });

        // Búsqueda de imagen desde el agente
        document.getElementById('btn-agent-search-image').addEventListener('click', async () => {
            const query = document.getElementById('agent-search-image').value.trim();
            if (!query) return;
            try {
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, top_k: 5})
                });
                const data = await res.json();
                const container = document.getElementById('agent-search-results');
                if (!data.results || data.results.length === 0) {
                    container.innerHTML = '<p class="is-size-7 has-text-grey">Sin resultados</p>';
                    return;
                }
                container.innerHTML = data.results.map(r => `
                    <div class="box is-clickable p-2 mb-1" onclick="document.getElementById('agent-image-select').value='${r.image_name}'">
                        <div class="level is-mobile">
                            <div class="level-left">
                                <img src="/api/images/view/${r.image_name}" style="width:40px;height:40px;object-fit:cover;border-radius:4px" class="mr-2">
                                <div>
                                    <p class="is-size-7 has-text-weight-bold">${r.image_name}</p>
                                    <p class="is-size-7 has-text-grey">${(r.score * 100).toFixed(1)}%</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch(e) {}
        });

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
