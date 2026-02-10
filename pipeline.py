"""
Pipeline Module - Sistema de pipelines configurables para procesamiento de imágenes
Soporta modelos diferentes por paso y sustitución de variables JSON en prompts
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from llm_providers import LLMConfig, LLMResponse, create_provider, substitute_variables


@dataclass
class PipelineStep:
    """Un paso individual en el pipeline - puede tener su propio modelo LLM"""
    id: str
    name: str
    prompt: str
    order: int
    use_previous_output: bool = True
    # Configuración LLM específica para este paso (opcional)
    llm_config_name: Optional[str] = None  # Nombre de la configuración LLM a usar
    # Si True, la salida de este paso se usa para indexar semánticamente la imagen
    index_for_search: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "order": self.order,
            "use_previous_output": self.use_previous_output,
            "llm_config_name": self.llm_config_name,
            "index_for_search": self.index_for_search
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            prompt=data["prompt"],
            order=data["order"],
            use_previous_output=data.get("use_previous_output", True),
            llm_config_name=data.get("llm_config_name"),
            index_for_search=data.get("index_for_search", False)
        )


@dataclass
class StepResult:
    """Resultado de un paso del pipeline con métricas detalladas"""
    step_id: str
    step_name: str
    step_order: int
    prompt_used: str  # El prompt después de sustituir variables
    response: LLMResponse
    model_used: str
    provider_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "step_order": self.step_order,
            "prompt_used": self.prompt_used,
            "content": self.response.content,
            "model_used": self.model_used,
            "provider_used": self.provider_used,
            "input_tokens": self.response.input_tokens,
            "output_tokens": self.response.output_tokens,
            "total_tokens": self.response.total_tokens,
            "latency_ms": self.response.latency_ms,
            "cost_estimate": self.response.cost_estimate,
            "success": self.response.success,
            "error": self.response.error,
            "timestamp": self.timestamp
        }


@dataclass
class Pipeline:
    """Pipeline de procesamiento de imágenes con soporte multi-modelo"""
    id: str
    name: str
    description: str
    default_llm_config_name: str  # Nombre de la configuración LLM por defecto
    steps: List[PipelineStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_step(self, name: str, prompt: str, use_previous_output: bool = True,
                 llm_config_name: Optional[str] = None, index_for_search: bool = False) -> PipelineStep:
        """Agrega un nuevo paso al pipeline"""
        step = PipelineStep(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            order=len(self.steps),
            use_previous_output=use_previous_output,
            llm_config_name=llm_config_name,
            index_for_search=index_for_search
        )
        self.steps.append(step)
        return step
    
    def remove_step(self, step_id: str) -> bool:
        """Elimina un paso del pipeline"""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                self.steps.pop(i)
                for j, s in enumerate(self.steps):
                    s.order = j
                return True
        return False
    
    def reorder_steps(self, step_ids: List[str]) -> None:
        """Reordena los pasos según la lista de IDs"""
        step_map = {s.id: s for s in self.steps}
        self.steps = []
        for i, sid in enumerate(step_ids):
            if sid in step_map:
                step_map[sid].order = i
                self.steps.append(step_map[sid])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default_llm_config_name": self.default_llm_config_name,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        pipeline = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            default_llm_config_name=data.get("default_llm_config_name", ""),
            created_at=data.get("created_at", datetime.now().isoformat())
        )
        
        for step_data in data.get("steps", []):
            pipeline.steps.append(PipelineStep.from_dict(step_data))
        
        return pipeline


@dataclass
class PipelineExecution:
    """Resultado completo de una ejecución de pipeline"""
    id: str
    pipeline_id: str
    pipeline_name: str
    image_path: str
    image_name: str
    context_data: Optional[Dict[str, Any]] = None  # JSON de contexto para variables
    step_results: List[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    models_used: List[str] = field(default_factory=list)
    success: bool = True
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    # Para revisión de pipelines
    review_status: Optional[str] = None  # "pending", "reviewed"
    step_reviews: Dict[str, bool] = field(default_factory=dict)  # step_id -> correcto/incorrecto
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "image_path": self.image_path,
            "image_name": self.image_name,
            "context_data": self.context_data,
            "step_results": [r.to_dict() for r in self.step_results],
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "models_used": self.models_used,
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "review_status": self.review_status,
            "step_reviews": self.step_reviews
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineExecution":
        execution = cls(
            id=data["id"],
            pipeline_id=data["pipeline_id"],
            pipeline_name=data["pipeline_name"],
            image_path=data["image_path"],
            image_name=data.get("image_name", ""),
            context_data=data.get("context_data"),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            models_used=data.get("models_used", []),
            success=data.get("success", True),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            review_status=data.get("review_status"),
            step_reviews=data.get("step_reviews", {})
        )
        return execution


@dataclass
class BatchExecution:
    """Resultado de una ejecución batch de múltiples imágenes"""
    id: str
    pipeline_id: str
    pipeline_name: str
    executions: List[PipelineExecution] = field(default_factory=list)
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "executions": [e.to_dict() for e in self.executions],
            "total_images": self.total_images,
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class PipelineRunner:
    """Ejecutor de pipelines con soporte para variables y múltiples modelos"""
    
    def __init__(self, config_loader=None):
        """
        Args:
            config_loader: Función que recibe un nombre de config y devuelve LLMConfig
        """
        self.config_loader = config_loader
        self.current_execution: Optional[PipelineExecution] = None
    
    async def run(self, pipeline: Pipeline, image_path: str,
                  context_data: Optional[Dict[str, Any]] = None,
                  progress_callback=None) -> PipelineExecution:
        """
        Ejecuta un pipeline completo sobre una imagen.
        
        Args:
            pipeline: Pipeline a ejecutar
            image_path: Ruta a la imagen
            context_data: JSON con variables para sustituir en prompts
            progress_callback: Callback para reportar progreso
        """
        from pathlib import Path
        
        execution = PipelineExecution(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            image_path=image_path,
            image_name=Path(image_path).name,
            context_data=context_data,
            review_status="pending"
        )
        self.current_execution = execution
        
        previous_output = ""
        models_used = set()
        
        sorted_steps = sorted(pipeline.steps, key=lambda s: s.order)
        
        for i, step in enumerate(sorted_steps):
            if progress_callback:
                progress_callback(i, len(sorted_steps), step.name)
            
            # Determinar qué configuración LLM usar para este paso
            config_name = step.llm_config_name or pipeline.default_llm_config_name
            
            if not self.config_loader:
                raise ValueError("config_loader es requerido para ejecutar pipelines")
            
            llm_config = self.config_loader(config_name)
            if not llm_config:
                raise ValueError(f"Configuración LLM no encontrada: {config_name}")
            
            provider = create_provider(llm_config)
            models_used.add(f"{llm_config.provider}:{llm_config.model}")
            
            # Construir el prompt con sustitución de variables
            prompt = step.prompt
            if context_data:
                prompt = substitute_variables(prompt, context_data)
            
            if step.use_previous_output and previous_output:
                prompt = f"""Contexto de análisis previo:
{previous_output}

---

{prompt}"""
            
            # Ejecutar el paso
            response = await provider.analyze_image(image_path, prompt)
            
            step_result = StepResult(
                step_id=step.id,
                step_name=step.name,
                step_order=step.order,
                prompt_used=prompt,
                response=response,
                model_used=llm_config.model,
                provider_used=llm_config.provider
            )
            execution.step_results.append(step_result)
            
            # Actualizar métricas
            execution.total_latency_ms += response.latency_ms
            execution.total_tokens += response.total_tokens
            execution.total_input_tokens += response.input_tokens
            execution.total_output_tokens += response.output_tokens
            execution.total_cost += response.cost_estimate
            
            if not response.success:
                execution.success = False
                break
            
            previous_output = response.content
        
        execution.models_used = list(models_used)
        execution.completed_at = datetime.now().isoformat()
        self.current_execution = None
        
        return execution
    
    async def run_batch(self, pipeline: Pipeline, 
                        image_context_pairs: List[Dict[str, Any]],
                        progress_callback=None) -> BatchExecution:
        """
        Ejecuta un pipeline sobre múltiples imágenes con sus contextos.
        
        Args:
            pipeline: Pipeline a ejecutar
            image_context_pairs: Lista de dicts con 'image_path' y opcional 'context_data'
            progress_callback: Callback para reportar progreso
        """
        batch = BatchExecution(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            total_images=len(image_context_pairs)
        )
        
        for i, pair in enumerate(image_context_pairs):
            if progress_callback:
                progress_callback(f"Procesando imagen {i+1}/{len(image_context_pairs)}")
            
            image_path = pair["image_path"]
            context_data = pair.get("context_data")
            
            try:
                execution = await self.run(
                    pipeline, 
                    image_path, 
                    context_data,
                    progress_callback=lambda step, total, name: progress_callback(
                        f"Imagen {i+1}/{len(image_context_pairs)} - Paso {step+1}/{total}: {name}"
                    ) if progress_callback else None
                )
                batch.executions.append(execution)
                
                if execution.success:
                    batch.successful_images += 1
                else:
                    batch.failed_images += 1
                
                batch.total_latency_ms += execution.total_latency_ms
                batch.total_tokens += execution.total_tokens
                batch.total_cost += execution.total_cost
                
            except Exception as e:
                batch.failed_images += 1
        
        batch.completed_at = datetime.now().isoformat()
        return batch


class PipelineComparator:
    """Comparador de múltiples pipelines"""
    
    def __init__(self, config_loader=None):
        self.config_loader = config_loader
    
    async def compare(self, pipelines: List[Pipeline], image_path: str,
                      context_data: Optional[Dict[str, Any]] = None,
                      progress_callback=None) -> List[PipelineExecution]:
        """Ejecuta múltiples pipelines y devuelve los resultados para comparación"""
        results = []
        
        for i, pipeline in enumerate(pipelines):
            if progress_callback:
                progress_callback(f"Ejecutando pipeline {i+1}/{len(pipelines)}: {pipeline.name}")
            
            runner = PipelineRunner(config_loader=self.config_loader)
            execution = await runner.run(
                pipeline, 
                image_path,
                context_data,
                progress_callback=lambda step, total, name: progress_callback(
                    f"Pipeline {pipeline.name} - Paso {step+1}/{total}: {name}"
                ) if progress_callback else None
            )
            results.append(execution)
        
        return results
    
    @staticmethod
    def generate_comparison_report(executions: List[PipelineExecution]) -> Dict[str, Any]:
        """Genera un reporte comparativo de las ejecuciones"""
        if not executions:
            return {}
        
        report = {
            "summary": [],
            "fastest": None,
            "cheapest": None,
            "most_tokens": None,
            "least_tokens": None,
            "by_model": {}
        }
        
        for exec in executions:
            summary = {
                "pipeline_name": exec.pipeline_name,
                "pipeline_id": exec.pipeline_id,
                "total_latency_ms": exec.total_latency_ms,
                "total_tokens": exec.total_tokens,
                "total_input_tokens": exec.total_input_tokens,
                "total_output_tokens": exec.total_output_tokens,
                "total_cost": exec.total_cost,
                "success": exec.success,
                "steps_completed": len(exec.step_results),
                "models_used": exec.models_used,
                "avg_latency_per_step": exec.total_latency_ms / len(exec.step_results) if exec.step_results else 0
            }
            report["summary"].append(summary)
            
            # Agrupar por modelo
            for model in exec.models_used:
                if model not in report["by_model"]:
                    report["by_model"][model] = {
                        "executions": 0,
                        "total_tokens": 0,
                        "total_cost": 0
                    }
                report["by_model"][model]["executions"] += 1
        
        # Encontrar mejores métricas
        successful = [s for s in report["summary"] if s["success"]]
        if successful:
            report["fastest"] = min(successful, key=lambda x: x["total_latency_ms"])["pipeline_name"]
            report["cheapest"] = min(successful, key=lambda x: x["total_cost"])["pipeline_name"]
            report["most_tokens"] = max(successful, key=lambda x: x["total_tokens"])["pipeline_name"]
            report["least_tokens"] = min(successful, key=lambda x: x["total_tokens"])["pipeline_name"]
        
        return report


class PipelineReviewer:
    """Sistema de revisión de pipelines para evaluar correctitud"""
    
    @staticmethod
    def review_step(execution: PipelineExecution, step_id: str, is_correct: bool) -> None:
        """Marca un paso como correcto o incorrecto"""
        execution.step_reviews[step_id] = is_correct
        
        # Actualizar estado de revisión
        if len(execution.step_reviews) == len(execution.step_results):
            execution.review_status = "reviewed"
    
    @staticmethod
    def get_accuracy_metrics(executions: List[PipelineExecution]) -> Dict[str, Any]:
        """Calcula métricas de precisión basadas en las revisiones"""
        reviewed = [e for e in executions if e.review_status == "reviewed"]
        
        if not reviewed:
            return {"message": "No hay ejecuciones revisadas"}
        
        # Métricas por pipeline
        pipeline_metrics = {}
        model_metrics = {}
        
        for exec in reviewed:
            pid = exec.pipeline_id
            if pid not in pipeline_metrics:
                pipeline_metrics[pid] = {
                    "name": exec.pipeline_name,
                    "total_steps": 0,
                    "correct_steps": 0,
                    "executions": 0,
                    "fully_correct": 0
                }
            
            pipeline_metrics[pid]["executions"] += 1
            all_correct = True
            
            for step_result in exec.step_results:
                step_id = step_result.step_id
                is_correct = exec.step_reviews.get(step_id, False)
                
                pipeline_metrics[pid]["total_steps"] += 1
                if is_correct:
                    pipeline_metrics[pid]["correct_steps"] += 1
                else:
                    all_correct = False
                
                # Métricas por modelo
                model_key = f"{step_result.provider_used}:{step_result.model_used}"
                if model_key not in model_metrics:
                    model_metrics[model_key] = {
                        "total_steps": 0,
                        "correct_steps": 0
                    }
                model_metrics[model_key]["total_steps"] += 1
                if is_correct:
                    model_metrics[model_key]["correct_steps"] += 1
            
            if all_correct:
                pipeline_metrics[pid]["fully_correct"] += 1
        
        # Calcular porcentajes
        for pid, metrics in pipeline_metrics.items():
            metrics["step_accuracy"] = (
                metrics["correct_steps"] / metrics["total_steps"] * 100
                if metrics["total_steps"] > 0 else 0
            )
            metrics["execution_accuracy"] = (
                metrics["fully_correct"] / metrics["executions"] * 100
                if metrics["executions"] > 0 else 0
            )
        
        for model, metrics in model_metrics.items():
            metrics["accuracy"] = (
                metrics["correct_steps"] / metrics["total_steps"] * 100
                if metrics["total_steps"] > 0 else 0
            )
        
        return {
            "total_reviewed": len(reviewed),
            "pipeline_metrics": pipeline_metrics,
            "model_metrics": model_metrics
        }
