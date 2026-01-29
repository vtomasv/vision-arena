"""
Storage Module - Sistema de persistencia para configuraciones, pipelines e historial
Soporta procesamiento batch y sistema de revisión de pipelines
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from llm_providers import LLMConfig
from pipeline import Pipeline, PipelineExecution, BatchExecution


class StorageManager:
    """Gestor de almacenamiento para la aplicación"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.environ.get(
                "VISION_LLM_DATA_DIR",
                os.path.join(os.path.expanduser("~"), ".vision_llm_comparator")
            )
        
        self.base_dir = Path(base_dir)
        self.pipelines_dir = self.base_dir / "pipelines"
        self.executions_dir = self.base_dir / "executions"
        self.batch_dir = self.base_dir / "batch"
        self.configs_dir = self.base_dir / "configs"
        self.images_dir = self.base_dir / "images"
        self.contexts_dir = self.base_dir / "contexts"  # Para archivos JSON de contexto
        self.reports_dir = self.base_dir / "reports"  # Para reportes PDF forenses
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crea los directorios necesarios si no existen"""
        for dir_path in [self.pipelines_dir, self.executions_dir, self.batch_dir,
                         self.configs_dir, self.images_dir, self.contexts_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== LLM Configs ====================
    
    def save_llm_config(self, config: LLMConfig) -> str:
        """Guarda una configuración de LLM"""
        config_data = {
            "name": config.name,
            "provider": config.provider,
            "model": config.model,
            "api_key": config.api_key,
            "api_base": config.api_base,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "extra_params": config.extra_params
        }
        
        filename = f"{config.name.replace(' ', '_').lower()}.json"
        filepath = self.configs_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_llm_config(self, name: str) -> Optional[LLMConfig]:
        """Carga una configuración de LLM por nombre"""
        filename = f"{name.replace(' ', '_').lower()}.json"
        filepath = self.configs_dir / filename
        
        if not filepath.exists():
            # Intentar buscar por nombre exacto en todos los archivos
            for fp in self.configs_dir.glob("*.json"):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("name") == name:
                        return LLMConfig(**data)
                except Exception:
                    continue
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return LLMConfig(**data)
    
    def get_config_loader(self) -> Callable[[str], Optional[LLMConfig]]:
        """Devuelve una función para cargar configuraciones por nombre"""
        return self.load_llm_config
    
    def list_llm_configs(self) -> List[Dict[str, Any]]:
        """Lista todas las configuraciones de LLM guardadas"""
        configs = []
        for filepath in self.configs_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["api_key_preview"] = data["api_key"][:8] + "..." if data.get("api_key") else ""
                    configs.append(data)
            except Exception:
                continue
        return configs
    
    def get_llm_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene una configuración de LLM por nombre (sin API key completa)"""
        filename = f"{name.replace(' ', '_').lower()}.json"
        filepath = self.configs_dir / filename
        
        if not filepath.exists():
            # Intentar buscar por nombre exacto en todos los archivos
            for fp in self.configs_dir.glob("*.json"):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("name") == name:
                        data["api_key"] = data["api_key"][:8] + "..." if data.get("api_key") else ""
                        return data
                except Exception:
                    continue
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["api_key"] = data["api_key"][:8] + "..." if data.get("api_key") else ""
        return data
    
    def get_llm_config_full(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene una configuración de LLM por nombre (con API key completa para edición)"""
        filename = f"{name.replace(' ', '_').lower()}.json"
        filepath = self.configs_dir / filename
        
        if not filepath.exists():
            # Intentar buscar por nombre exacto en todos los archivos
            for fp in self.configs_dir.glob("*.json"):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("name") == name:
                        return data
                except Exception:
                    continue
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def delete_llm_config(self, name: str) -> bool:
        """Elimina una configuración de LLM"""
        filename = f"{name.replace(' ', '_').lower()}.json"
        filepath = self.configs_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    # ==================== Pipelines ====================
    
    def save_pipeline(self, pipeline: Pipeline) -> str:
        """Guarda un pipeline"""
        data = pipeline.to_dict()
        
        filename = f"{pipeline.id}.json"
        filepath = self.pipelines_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Carga un pipeline por ID"""
        filepath = self.pipelines_dir / f"{pipeline_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return Pipeline.from_dict(data)
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """Lista todos los pipelines guardados"""
        pipelines = []
        for filepath in self.pipelines_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Resumen para listado
                    summary = {
                        "id": data["id"],
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "default_llm_config_name": data.get("default_llm_config_name", ""),
                        "steps_count": len(data.get("steps", [])),
                        "steps": data.get("steps", []),
                        "created_at": data.get("created_at", "")
                    }
                    pipelines.append(summary)
            except Exception:
                continue
        
        pipelines.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return pipelines
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Elimina un pipeline"""
        filepath = self.pipelines_dir / f"{pipeline_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    # ==================== Executions (Historial) ====================
    
    def save_execution(self, execution: PipelineExecution) -> str:
        """Guarda una ejecución en el historial"""
        data = execution.to_dict()
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.executions_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        filename = f"{execution.id}.json"
        filepath = date_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def update_execution(self, execution: PipelineExecution) -> bool:
        """Actualiza una ejecución existente (para revisiones)"""
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                filepath = date_dir / f"{execution.id}.json"
                if filepath.exists():
                    data = execution.to_dict()
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return True
        return False
    
    def load_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Carga una ejecución por ID"""
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                filepath = date_dir / f"{execution_id}.json"
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return PipelineExecution.from_dict(data)
        return None
    
    def list_executions(self, limit: int = 100, pipeline_id: str = None,
                        review_status: str = None) -> List[Dict[str, Any]]:
        """Lista las ejecuciones del historial"""
        executions = []
        
        date_dirs = sorted(self.executions_dir.iterdir(), reverse=True)
        
        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue
            
            for filepath in sorted(date_dir.glob("*.json"), reverse=True):
                if len(executions) >= limit:
                    break
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if pipeline_id and data.get("pipeline_id") != pipeline_id:
                        continue
                    
                    if review_status and data.get("review_status") != review_status:
                        continue
                    
                    summary = {
                        "id": data["id"],
                        "pipeline_id": data["pipeline_id"],
                        "pipeline_name": data["pipeline_name"],
                        "image_path": data["image_path"],
                        "image_name": data.get("image_name", ""),
                        "context_data": data.get("context_data"),
                        "total_latency_ms": data.get("total_latency_ms", 0),
                        "total_tokens": data.get("total_tokens", 0),
                        "total_input_tokens": data.get("total_input_tokens", 0),
                        "total_output_tokens": data.get("total_output_tokens", 0),
                        "total_cost": data.get("total_cost", 0),
                        "models_used": data.get("models_used", []),
                        "success": data.get("success", True),
                        "steps_count": len(data.get("step_results", [])),
                        "started_at": data.get("started_at", ""),
                        "completed_at": data.get("completed_at", ""),
                        "review_status": data.get("review_status"),
                        "step_reviews": data.get("step_reviews", {})
                    }
                    executions.append(summary)
                except Exception:
                    continue
            
            if len(executions) >= limit:
                break
        
        return executions
    
    def get_execution_details(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los detalles completos de una ejecución"""
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                filepath = date_dir / f"{execution_id}.json"
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        return json.load(f)
        return None
    
    def delete_execution(self, execution_id: str) -> bool:
        """Elimina una ejecución del historial"""
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                filepath = date_dir / f"{execution_id}.json"
                if filepath.exists():
                    filepath.unlink()
                    return True
        return False
    
    def clear_history(self, before_date: str = None) -> int:
        """Limpia el historial de ejecuciones"""
        count = 0
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                if before_date and date_dir.name >= before_date:
                    continue
                for filepath in date_dir.glob("*.json"):
                    filepath.unlink()
                    count += 1
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()
        return count
    
    # ==================== Batch Executions ====================
    
    def save_batch_execution(self, batch: BatchExecution) -> str:
        """Guarda una ejecución batch"""
        data = batch.to_dict()
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.batch_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        filename = f"{batch.id}.json"
        filepath = date_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # También guardar cada ejecución individual
        for execution in batch.executions:
            self.save_execution(execution)
        
        return str(filepath)
    
    def list_batch_executions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lista las ejecuciones batch"""
        batches = []
        
        date_dirs = sorted(self.batch_dir.iterdir(), reverse=True)
        
        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue
            
            for filepath in sorted(date_dir.glob("*.json"), reverse=True):
                if len(batches) >= limit:
                    break
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    summary = {
                        "id": data["id"],
                        "pipeline_id": data["pipeline_id"],
                        "pipeline_name": data["pipeline_name"],
                        "total_images": data.get("total_images", 0),
                        "successful_images": data.get("successful_images", 0),
                        "failed_images": data.get("failed_images", 0),
                        "total_latency_ms": data.get("total_latency_ms", 0),
                        "total_tokens": data.get("total_tokens", 0),
                        "total_cost": data.get("total_cost", 0),
                        "started_at": data.get("started_at", ""),
                        "completed_at": data.get("completed_at", "")
                    }
                    batches.append(summary)
                except Exception:
                    continue
            
            if len(batches) >= limit:
                break
        
        return batches
    
    # ==================== Images ====================
    
    def save_image(self, source_path: str, new_name: str = None) -> str:
        """Copia una imagen al directorio de imágenes"""
        source = Path(source_path)
        if new_name:
            dest_name = new_name
        else:
            dest_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source.name}"
        
        dest_path = self.images_dir / dest_name
        shutil.copy2(source_path, dest_path)
        return str(dest_path)
    
    def save_context(self, context_data: Dict[str, Any], name: str) -> str:
        """Guarda un archivo de contexto JSON"""
        filepath = self.contexts_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        return str(filepath)
    
    def load_context(self, name: str) -> Optional[Dict[str, Any]]:
        """Carga un archivo de contexto JSON"""
        # Remover extensión si la tiene
        if name.endswith(".json"):
            name = name[:-5]
        
        filepath = self.contexts_dir / f"{name}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_images(self) -> List[Dict[str, Any]]:
        """Lista las imágenes guardadas"""
        images = []
        extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        
        for filepath in self.images_dir.iterdir():
            if filepath.suffix.lower() in extensions:
                # Buscar contexto asociado
                context_name = filepath.stem
                context_path = self.contexts_dir / f"{context_name}.json"
                has_context = context_path.exists()
                
                images.append({
                    "name": filepath.name,
                    "path": str(filepath),
                    "size": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                    "has_context": has_context,
                    "context_name": context_name if has_context else None
                })
        
        images.sort(key=lambda x: x["modified"], reverse=True)
        return images
    
    def get_image_with_context(self, image_name: str) -> Dict[str, Any]:
        """Obtiene una imagen con su contexto asociado si existe"""
        image_path = self.images_dir / image_name
        if not image_path.exists():
            return None
        
        context_name = image_path.stem
        context_data = self.load_context(context_name)
        
        return {
            "image_path": str(image_path),
            "image_name": image_name,
            "context_data": context_data
        }
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales del uso"""
        executions = self.list_executions(limit=1000)
        
        total_executions = len(executions)
        successful = sum(1 for e in executions if e["success"])
        total_tokens = sum(e["total_tokens"] for e in executions)
        total_input_tokens = sum(e.get("total_input_tokens", 0) for e in executions)
        total_output_tokens = sum(e.get("total_output_tokens", 0) for e in executions)
        total_cost = sum(e["total_cost"] for e in executions)
        total_latency = sum(e["total_latency_ms"] for e in executions)
        
        # Estadísticas por pipeline
        pipeline_stats = {}
        for e in executions:
            pid = e["pipeline_id"]
            if pid not in pipeline_stats:
                pipeline_stats[pid] = {
                    "name": e["pipeline_name"],
                    "executions": 0,
                    "successful": 0,
                    "tokens": 0,
                    "cost": 0,
                    "latency": 0,
                    "reviewed": 0,
                    "correct_steps": 0,
                    "total_steps": 0
                }
            pipeline_stats[pid]["executions"] += 1
            if e["success"]:
                pipeline_stats[pid]["successful"] += 1
            pipeline_stats[pid]["tokens"] += e["total_tokens"]
            pipeline_stats[pid]["cost"] += e["total_cost"]
            pipeline_stats[pid]["latency"] += e["total_latency_ms"]
            
            # Estadísticas de revisión
            if e.get("review_status") == "reviewed":
                pipeline_stats[pid]["reviewed"] += 1
                step_reviews = e.get("step_reviews", {})
                pipeline_stats[pid]["correct_steps"] += sum(1 for v in step_reviews.values() if v)
                pipeline_stats[pid]["total_steps"] += len(step_reviews)
        
        # Calcular accuracy por pipeline
        for pid, stats in pipeline_stats.items():
            if stats["total_steps"] > 0:
                stats["accuracy"] = round(stats["correct_steps"] / stats["total_steps"] * 100, 2)
            else:
                stats["accuracy"] = None
        
        # Estadísticas por modelo
        model_stats = {}
        for e in executions:
            for model in e.get("models_used", []):
                if model not in model_stats:
                    model_stats[model] = {
                        "executions": 0,
                        "tokens": 0,
                        "cost": 0
                    }
                model_stats[model]["executions"] += 1
        
        # Conteo de revisiones
        pending_reviews = sum(1 for e in executions if e.get("review_status") == "pending")
        completed_reviews = sum(1 for e in executions if e.get("review_status") == "reviewed")
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": total_executions - successful,
            "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 4),
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(total_latency / total_executions, 2) if total_executions > 0 else 0,
            "pipelines_count": len(self.list_pipelines()),
            "configs_count": len(self.list_llm_configs()),
            "images_count": len(self.list_images()),
            "pipeline_stats": pipeline_stats,
            "model_stats": model_stats,
            "pending_reviews": pending_reviews,
            "completed_reviews": completed_reviews
        }
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de revisiones"""
        executions = self.list_executions(limit=1000, review_status="reviewed")
        
        if not executions:
            return {"message": "No hay ejecuciones revisadas"}
        
        # Cargar detalles completos para análisis
        pipeline_accuracy = {}
        model_accuracy = {}
        
        for exec_summary in executions:
            exec_details = self.get_execution_details(exec_summary["id"])
            if not exec_details:
                continue
            
            pid = exec_details["pipeline_id"]
            pname = exec_details["pipeline_name"]
            
            if pid not in pipeline_accuracy:
                pipeline_accuracy[pid] = {
                    "name": pname,
                    "total_executions": 0,
                    "fully_correct": 0,
                    "total_steps": 0,
                    "correct_steps": 0
                }
            
            pipeline_accuracy[pid]["total_executions"] += 1
            step_reviews = exec_details.get("step_reviews", {})
            step_results = exec_details.get("step_results", [])
            
            all_correct = True
            for step in step_results:
                step_id = step["step_id"]
                is_correct = step_reviews.get(step_id, False)
                model_key = f"{step.get('provider_used', 'unknown')}:{step.get('model_used', 'unknown')}"
                
                pipeline_accuracy[pid]["total_steps"] += 1
                if is_correct:
                    pipeline_accuracy[pid]["correct_steps"] += 1
                else:
                    all_correct = False
                
                if model_key not in model_accuracy:
                    model_accuracy[model_key] = {
                        "total_steps": 0,
                        "correct_steps": 0
                    }
                model_accuracy[model_key]["total_steps"] += 1
                if is_correct:
                    model_accuracy[model_key]["correct_steps"] += 1
            
            if all_correct:
                pipeline_accuracy[pid]["fully_correct"] += 1
        
        # Calcular porcentajes
        for pid, data in pipeline_accuracy.items():
            data["step_accuracy_pct"] = (
                round(data["correct_steps"] / data["total_steps"] * 100, 2)
                if data["total_steps"] > 0 else 0
            )
            data["execution_accuracy_pct"] = (
                round(data["fully_correct"] / data["total_executions"] * 100, 2)
                if data["total_executions"] > 0 else 0
            )
        
        for model, data in model_accuracy.items():
            data["accuracy_pct"] = (
                round(data["correct_steps"] / data["total_steps"] * 100, 2)
                if data["total_steps"] > 0 else 0
            )
        
        return {
            "total_reviewed": len(executions),
            "pipeline_accuracy": pipeline_accuracy,
            "model_accuracy": model_accuracy
        }
