"""
Pipeline Module - Sistema de pipelines configurables para procesamiento de imágenes
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from llm_providers import LLMConfig, LLMResponse, create_provider


@dataclass
class PipelineStep:
    """Un paso individual en el pipeline"""
    id: str
    name: str
    prompt: str
    order: int
    use_previous_output: bool = True  # Si usar la salida anterior como contexto
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "order": self.order,
            "use_previous_output": self.use_previous_output
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            prompt=data["prompt"],
            order=data["order"],
            use_previous_output=data.get("use_previous_output", True)
        )


@dataclass
class StepResult:
    """Resultado de un paso del pipeline"""
    step_id: str
    step_name: str
    response: LLMResponse
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "content": self.response.content,
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
    """Pipeline de procesamiento de imágenes"""
    id: str
    name: str
    description: str
    llm_config: LLMConfig
    steps: List[PipelineStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_step(self, name: str, prompt: str, use_previous_output: bool = True) -> PipelineStep:
        """Agrega un nuevo paso al pipeline"""
        step = PipelineStep(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            order=len(self.steps),
            use_previous_output=use_previous_output
        )
        self.steps.append(step)
        return step
    
    def remove_step(self, step_id: str) -> bool:
        """Elimina un paso del pipeline"""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                self.steps.pop(i)
                # Reordenar
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
            "llm_config": self.llm_config.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        llm_data = data["llm_config"]
        llm_config = LLMConfig(
            name=llm_data["name"],
            provider=llm_data["provider"],
            model=llm_data["model"],
            api_key=llm_data.get("api_key", ""),
            api_base=llm_data.get("api_base", ""),
            max_tokens=llm_data.get("max_tokens", 4096),
            temperature=llm_data.get("temperature", 0.7),
            extra_params=llm_data.get("extra_params", {})
        )
        
        pipeline = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            llm_config=llm_config,
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
    step_results: List[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    success: bool = True
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "image_path": self.image_path,
            "step_results": [r.to_dict() for r in self.step_results],
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineExecution":
        execution = cls(
            id=data["id"],
            pipeline_id=data["pipeline_id"],
            pipeline_name=data["pipeline_name"],
            image_path=data["image_path"],
            total_latency_ms=data.get("total_latency_ms", 0.0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            success=data.get("success", True),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at")
        )
        return execution


class PipelineRunner:
    """Ejecutor de pipelines"""
    
    def __init__(self):
        self.current_execution: Optional[PipelineExecution] = None
    
    async def run(self, pipeline: Pipeline, image_path: str, 
                  progress_callback=None) -> PipelineExecution:
        """Ejecuta un pipeline completo sobre una imagen"""
        
        execution = PipelineExecution(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            image_path=image_path
        )
        self.current_execution = execution
        
        provider = create_provider(pipeline.llm_config)
        previous_output = ""
        
        sorted_steps = sorted(pipeline.steps, key=lambda s: s.order)
        
        for i, step in enumerate(sorted_steps):
            if progress_callback:
                progress_callback(i, len(sorted_steps), step.name)
            
            # Construir el prompt
            prompt = step.prompt
            if step.use_previous_output and previous_output:
                prompt = f"""Contexto de análisis previo:
{previous_output}

---

{step.prompt}"""
            
            # Ejecutar el paso
            response = await provider.analyze_image(image_path, prompt)
            
            step_result = StepResult(
                step_id=step.id,
                step_name=step.name,
                response=response
            )
            execution.step_results.append(step_result)
            
            # Actualizar métricas
            execution.total_latency_ms += response.latency_ms
            execution.total_tokens += response.total_tokens
            execution.total_cost += response.cost_estimate
            
            if not response.success:
                execution.success = False
                break
            
            previous_output = response.content
        
        execution.completed_at = datetime.now().isoformat()
        self.current_execution = None
        
        return execution


class PipelineComparator:
    """Comparador de múltiples pipelines"""
    
    def __init__(self):
        self.runner = PipelineRunner()
    
    async def compare(self, pipelines: List[Pipeline], image_path: str,
                      progress_callback=None) -> List[PipelineExecution]:
        """Ejecuta múltiples pipelines y devuelve los resultados para comparación"""
        results = []
        
        for i, pipeline in enumerate(pipelines):
            if progress_callback:
                progress_callback(f"Ejecutando pipeline {i+1}/{len(pipelines)}: {pipeline.name}")
            
            execution = await self.runner.run(
                pipeline, 
                image_path,
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
            "least_tokens": None
        }
        
        for exec in executions:
            summary = {
                "pipeline_name": exec.pipeline_name,
                "pipeline_id": exec.pipeline_id,
                "total_latency_ms": exec.total_latency_ms,
                "total_tokens": exec.total_tokens,
                "total_cost": exec.total_cost,
                "success": exec.success,
                "steps_completed": len(exec.step_results),
                "avg_latency_per_step": exec.total_latency_ms / len(exec.step_results) if exec.step_results else 0
            }
            report["summary"].append(summary)
        
        # Encontrar mejores métricas
        successful = [s for s in report["summary"] if s["success"]]
        if successful:
            report["fastest"] = min(successful, key=lambda x: x["total_latency_ms"])["pipeline_name"]
            report["cheapest"] = min(successful, key=lambda x: x["total_cost"])["pipeline_name"]
            report["most_tokens"] = max(successful, key=lambda x: x["total_tokens"])["pipeline_name"]
            report["least_tokens"] = min(successful, key=lambda x: x["total_tokens"])["pipeline_name"]
        
        return report
