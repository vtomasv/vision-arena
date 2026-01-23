"""
Storage Module - Sistema de persistencia para configuraciones, pipelines e historial
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from llm_providers import LLMConfig
from pipeline import Pipeline, PipelineExecution


class StorageManager:
    """Gestor de almacenamiento para la aplicación"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Usar variable de entorno si está definida, sino usar directorio home
            base_dir = os.environ.get(
                "VISION_LLM_DATA_DIR",
                os.path.join(os.path.expanduser("~"), ".vision_llm_comparator")
            )
        
        self.base_dir = Path(base_dir)
        self.pipelines_dir = self.base_dir / "pipelines"
        self.executions_dir = self.base_dir / "executions"
        self.configs_dir = self.base_dir / "configs"
        self.images_dir = self.base_dir / "images"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crea los directorios necesarios si no existen"""
        for dir_path in [self.pipelines_dir, self.executions_dir, 
                         self.configs_dir, self.images_dir]:
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
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return LLMConfig(**data)
    
    def list_llm_configs(self) -> List[Dict[str, Any]]:
        """Lista todas las configuraciones de LLM guardadas"""
        configs = []
        for filepath in self.configs_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ocultar API key en el listado
                    data["api_key_preview"] = data["api_key"][:8] + "..." if data.get("api_key") else ""
                    configs.append(data)
            except Exception:
                continue
        return configs
    
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
        # Guardar con API key completa para poder ejecutar
        data = pipeline.to_dict()
        data["llm_config"]["api_key"] = pipeline.llm_config.api_key
        
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
                        "llm_name": data["llm_config"]["name"],
                        "llm_model": data["llm_config"]["model"],
                        "steps_count": len(data.get("steps", [])),
                        "created_at": data.get("created_at", "")
                    }
                    pipelines.append(summary)
            except Exception:
                continue
        
        # Ordenar por fecha de creación (más reciente primero)
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
        
        # Crear subdirectorio por fecha
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.executions_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        filename = f"{execution.id}.json"
        filepath = date_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Carga una ejecución por ID"""
        # Buscar en todos los subdirectorios de fecha
        for date_dir in self.executions_dir.iterdir():
            if date_dir.is_dir():
                filepath = date_dir / f"{execution_id}.json"
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return PipelineExecution.from_dict(data)
        return None
    
    def list_executions(self, limit: int = 100, pipeline_id: str = None) -> List[Dict[str, Any]]:
        """Lista las ejecuciones del historial"""
        executions = []
        
        # Recorrer subdirectorios de fecha en orden inverso
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
                    
                    # Filtrar por pipeline si se especifica
                    if pipeline_id and data.get("pipeline_id") != pipeline_id:
                        continue
                    
                    # Resumen para listado
                    summary = {
                        "id": data["id"],
                        "pipeline_id": data["pipeline_id"],
                        "pipeline_name": data["pipeline_name"],
                        "image_path": data["image_path"],
                        "total_latency_ms": data.get("total_latency_ms", 0),
                        "total_tokens": data.get("total_tokens", 0),
                        "total_cost": data.get("total_cost", 0),
                        "success": data.get("success", True),
                        "steps_count": len(data.get("step_results", [])),
                        "started_at": data.get("started_at", ""),
                        "completed_at": data.get("completed_at", "")
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
                # Eliminar directorio vacío
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()
        return count
    
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
    
    def list_images(self) -> List[Dict[str, Any]]:
        """Lista las imágenes guardadas"""
        images = []
        extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        
        for filepath in self.images_dir.iterdir():
            if filepath.suffix.lower() in extensions:
                images.append({
                    "name": filepath.name,
                    "path": str(filepath),
                    "size": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                })
        
        images.sort(key=lambda x: x["modified"], reverse=True)
        return images
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales del uso"""
        executions = self.list_executions(limit=1000)
        
        total_executions = len(executions)
        successful = sum(1 for e in executions if e["success"])
        total_tokens = sum(e["total_tokens"] for e in executions)
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
                    "tokens": 0,
                    "cost": 0,
                    "latency": 0
                }
            pipeline_stats[pid]["executions"] += 1
            pipeline_stats[pid]["tokens"] += e["total_tokens"]
            pipeline_stats[pid]["cost"] += e["total_cost"]
            pipeline_stats[pid]["latency"] += e["total_latency_ms"]
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": total_executions - successful,
            "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(total_latency / total_executions, 2) if total_executions > 0 else 0,
            "pipelines_count": len(self.list_pipelines()),
            "configs_count": len(self.list_llm_configs()),
            "images_count": len(self.list_images()),
            "pipeline_stats": pipeline_stats
        }
