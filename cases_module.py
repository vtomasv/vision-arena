"""
Cases Module - Módulo independiente para gestión de Casos, Tareas, Agentes y Skills.
Sigue los estándares AGENTS.md y Agent Skills (agentskills.io).
"""

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


# ==================== Data Models ====================

@dataclass
class AgentDefinition:
    """Definición de un agente siguiendo el estándar AGENTS.md"""
    id: str
    name: str
    description: str
    icon: str = "fas fa-robot"
    color: str = "#4361ee"
    category: str = "general"
    llm_config_name: Optional[str] = None
    system_prompt: str = ""
    agent_md: str = ""  # Contenido completo del AGENTS.md
    skills: List[str] = field(default_factory=list)  # IDs de skills asociados
    is_default: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "category": self.category,
            "llm_config_name": self.llm_config_name,
            "system_prompt": self.system_prompt,
            "agent_md": self.agent_md,
            "skills": self.skills,
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentDefinition":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SkillDefinition:
    """Definición de un skill siguiendo el estándar Agent Skills (agentskills.io)"""
    id: str
    name: str  # Lowercase, hyphens only (spec)
    description: str
    category: str = "general"
    icon: str = "fas fa-cog"
    color: str = "#4361ee"
    license: str = ""
    compatibility: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: str = ""
    skill_md: str = ""  # Contenido completo del SKILL.md body
    scripts: Dict[str, str] = field(default_factory=dict)  # filename -> content
    references: Dict[str, str] = field(default_factory=dict)
    is_default: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "icon": self.icon,
            "color": self.color,
            "license": self.license,
            "compatibility": self.compatibility,
            "metadata": self.metadata,
            "allowed_tools": self.allowed_tools,
            "skill_md": self.skill_md,
            "scripts": self.scripts,
            "references": self.references,
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillDefinition":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Case:
    """Caso de investigación"""
    id: str
    name: str
    description: str
    case_type: str = "investigacion"  # penal, civil, investigacion_interna
    status: str = "abierto"  # abierto, en_progreso, archivado
    created_by: str = "admin"
    assigned_users: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    archived_at: Optional[str] = None
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "case_type": self.case_type,
            "status": self.status,
            "created_by": self.created_by,
            "assigned_users": self.assigned_users,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
            "audit_log": self.audit_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkflowStep:
    """Paso de un workflow con agente, skills y prompt asignados"""
    id: str
    name: str
    order: int
    agent_id: Optional[str] = None
    skill_ids: List[str] = field(default_factory=list)
    prompt: str = ""
    llm_config_name: Optional[str] = None
    use_previous_output: bool = True
    index_for_search: bool = False
    estimated_duration_min: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "order": self.order,
            "agent_id": self.agent_id,
            "skill_ids": self.skill_ids,
            "prompt": self.prompt,
            "llm_config_name": self.llm_config_name,
            "use_previous_output": self.use_previous_output,
            "index_for_search": self.index_for_search,
            "estimated_duration_min": self.estimated_duration_min,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Task:
    """Tarea asignada a un caso"""
    id: str
    case_id: str
    name: str
    description: str
    task_type: str = "humana"  # humana, agente, workflow
    status: str = "pendiente"  # pendiente, en_progreso, completada, cancelada
    priority: str = "media"  # alta, media, baja
    assigned_to: str = ""  # usuario o agente ID
    agent_id: Optional[str] = None
    workflow_steps: List[WorkflowStep] = field(default_factory=list)
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # diaria, semanal, mensual
    recurrence_time: Optional[str] = None  # HH:MM
    resource_folder_ids: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    due_date: Optional[str] = None
    created_by: str = "admin"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "agent_id": self.agent_id,
            "workflow_steps": [s.to_dict() for s in self.workflow_steps],
            "is_recurring": self.is_recurring,
            "recurrence_pattern": self.recurrence_pattern,
            "recurrence_time": self.recurrence_time,
            "resource_folder_ids": self.resource_folder_ids,
            "results": self.results,
            "due_date": self.due_date,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        obj = cls(
            id=data["id"],
            case_id=data["case_id"],
            name=data["name"],
            description=data.get("description", ""),
            task_type=data.get("task_type", "humana"),
            status=data.get("status", "pendiente"),
            priority=data.get("priority", "media"),
            assigned_to=data.get("assigned_to", ""),
            agent_id=data.get("agent_id"),
            is_recurring=data.get("is_recurring", False),
            recurrence_pattern=data.get("recurrence_pattern"),
            recurrence_time=data.get("recurrence_time"),
            resource_folder_ids=data.get("resource_folder_ids", []),
            results=data.get("results", []),
            due_date=data.get("due_date"),
            created_by=data.get("created_by", "admin"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
        )
        obj.workflow_steps = [
            WorkflowStep.from_dict(s) for s in data.get("workflow_steps", [])
        ]
        return obj


@dataclass
class CaseFolder:
    """Carpeta dentro de un caso para organizar archivos"""
    id: str
    case_id: str
    name: str
    parent_id: Optional[str] = None
    folder_type: str = "general"  # general, evidencias, resultados_agentes, ejecuciones
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "folder_type": self.folder_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseFolder":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CaseFile:
    """Archivo dentro de una carpeta de caso"""
    id: str
    case_id: str
    folder_id: str
    filename: str
    original_name: str
    file_type: str = ""
    file_size: int = 0
    indexed: bool = False
    uploaded_by: str = "admin"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "folder_id": self.folder_id,
            "filename": self.filename,
            "original_name": self.original_name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "indexed": self.indexed,
            "uploaded_by": self.uploaded_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseFile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvidenceRequest:
    """Solicitud de evidencias geográficas"""
    id: str
    case_id: str
    description: str
    polygon_coords: List[Dict[str, float]] = field(default_factory=list)
    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_m: float = 0.0
    cameras_found: int = 0
    status: str = "pendiente"  # pendiente, procesando, completada
    result_pdf: Optional[str] = None
    created_by: str = "admin"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "description": self.description,
            "polygon_coords": self.polygon_coords,
            "center_lat": self.center_lat,
            "center_lng": self.center_lng,
            "radius_m": self.radius_m,
            "cameras_found": self.cameras_found,
            "status": self.status,
            "result_pdf": self.result_pdf,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRequest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==================== Storage ====================

class CasesStorage:
    """Almacenamiento de datos para el módulo de Casos"""

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(
            os.environ.get("DATA_DIR", os.path.join(os.path.expanduser("~"), ".vision_llm_comparator")),
            "cases"
        )
        self._ensure_dirs()

    def _ensure_dirs(self):
        for sub in ["agents", "skills", "cases", "tasks", "folders", "files", "evidence_requests"]:
            os.makedirs(os.path.join(self.base_dir, sub), exist_ok=True)

    # --- Generic CRUD helpers ---
    def _save(self, category: str, obj_id: str, data: dict):
        path = os.path.join(self.base_dir, category, f"{obj_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, category: str, obj_id: str) -> Optional[dict]:
        path = os.path.join(self.base_dir, category, f"{obj_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _list(self, category: str) -> List[dict]:
        folder = os.path.join(self.base_dir, category)
        items = []
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".json"):
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        items.append(json.load(f))
        return items

    def _delete(self, category: str, obj_id: str) -> bool:
        path = os.path.join(self.base_dir, category, f"{obj_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # --- Agents ---
    def save_agent(self, agent: AgentDefinition):
        self._save("agents", agent.id, agent.to_dict())

    def load_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        data = self._load("agents", agent_id)
        return AgentDefinition.from_dict(data) if data else None

    def list_agents(self) -> List[Dict[str, Any]]:
        return sorted(self._list("agents"), key=lambda x: x.get("name", ""))

    def delete_agent(self, agent_id: str) -> bool:
        return self._delete("agents", agent_id)

    # --- Skills ---
    def save_skill(self, skill: SkillDefinition):
        self._save("skills", skill.id, skill.to_dict())

    def load_skill(self, skill_id: str) -> Optional[SkillDefinition]:
        data = self._load("skills", skill_id)
        return SkillDefinition.from_dict(data) if data else None

    def list_skills(self) -> List[Dict[str, Any]]:
        return sorted(self._list("skills"), key=lambda x: x.get("name", ""))

    def delete_skill(self, skill_id: str) -> bool:
        return self._delete("skills", skill_id)

    # --- Cases ---
    def save_case(self, case: Case):
        self._save("cases", case.id, case.to_dict())

    def load_case(self, case_id: str) -> Optional[Case]:
        data = self._load("cases", case_id)
        return Case.from_dict(data) if data else None

    def list_cases(self, status: str = None) -> List[Dict[str, Any]]:
        cases = self._list("cases")
        if status:
            cases = [c for c in cases if c.get("status") == status]
        return sorted(cases, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_case(self, case_id: str) -> bool:
        return self._delete("cases", case_id)

    # --- Tasks ---
    def save_task(self, task: Task):
        self._save("tasks", task.id, task.to_dict())

    def load_task(self, task_id: str) -> Optional[Task]:
        data = self._load("tasks", task_id)
        return Task.from_dict(data) if data else None

    def list_tasks(self, case_id: str = None, status: str = None) -> List[Dict[str, Any]]:
        tasks = self._list("tasks")
        if case_id:
            tasks = [t for t in tasks if t.get("case_id") == case_id]
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        return sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_task(self, task_id: str) -> bool:
        return self._delete("tasks", task_id)

    # --- Folders ---
    def save_folder(self, folder: CaseFolder):
        self._save("folders", folder.id, folder.to_dict())

    def load_folder(self, folder_id: str) -> Optional[CaseFolder]:
        data = self._load("folders", folder_id)
        return CaseFolder.from_dict(data) if data else None

    def list_folders(self, case_id: str) -> List[Dict[str, Any]]:
        folders = self._list("folders")
        return [f for f in folders if f.get("case_id") == case_id]

    def delete_folder(self, folder_id: str) -> bool:
        return self._delete("folders", folder_id)

    # --- Files ---
    def save_file_meta(self, file_meta: CaseFile):
        self._save("files", file_meta.id, file_meta.to_dict())

    def load_file_meta(self, file_id: str) -> Optional[CaseFile]:
        data = self._load("files", file_id)
        return CaseFile.from_dict(data) if data else None

    def list_files(self, case_id: str = None, folder_id: str = None) -> List[Dict[str, Any]]:
        files = self._list("files")
        if case_id:
            files = [f for f in files if f.get("case_id") == case_id]
        if folder_id:
            files = [f for f in files if f.get("folder_id") == folder_id]
        return sorted(files, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_file_meta(self, file_id: str) -> bool:
        return self._delete("files", file_id)

    def get_files_dir(self, case_id: str) -> str:
        d = os.path.join(self.base_dir, "case_files", case_id)
        os.makedirs(d, exist_ok=True)
        return d

    # --- Evidence Requests ---
    def save_evidence_request(self, req: EvidenceRequest):
        self._save("evidence_requests", req.id, req.to_dict())

    def load_evidence_request(self, req_id: str) -> Optional[EvidenceRequest]:
        data = self._load("evidence_requests", req_id)
        return EvidenceRequest.from_dict(data) if data else None

    def list_evidence_requests(self, case_id: str = None) -> List[Dict[str, Any]]:
        reqs = self._list("evidence_requests")
        if case_id:
            reqs = [r for r in reqs if r.get("case_id") == case_id]
        return sorted(reqs, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_evidence_request(self, req_id: str) -> bool:
        return self._delete("evidence_requests", req_id)

    # --- Dashboard Stats ---
    def get_dashboard_stats(self) -> Dict[str, Any]:
        cases = self._list("cases")
        tasks = self._list("tasks")
        active = [c for c in cases if c.get("status") != "archivado"]
        return {
            "total_cases": len(cases),
            "active_cases": len(active),
            "cases_by_status": {
                "abierto": len([c for c in cases if c.get("status") == "abierto"]),
                "en_progreso": len([c for c in cases if c.get("status") == "en_progreso"]),
                "archivado": len([c for c in cases if c.get("status") == "archivado"]),
            },
            "cases_by_type": self._count_by(cases, "case_type"),
            "total_tasks": len(tasks),
            "tasks_by_status": {
                "pendiente": len([t for t in tasks if t.get("status") == "pendiente"]),
                "en_progreso": len([t for t in tasks if t.get("status") == "en_progreso"]),
                "completada": len([t for t in tasks if t.get("status") == "completada"]),
            },
            "pending_tasks": [t for t in tasks if t.get("status") == "pendiente"][:10],
            "total_agents": len(self._list("agents")),
            "total_skills": len(self._list("skills")),
        }

    def _count_by(self, items: list, key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            val = item.get(key, "otro")
            counts[val] = counts.get(val, 0) + 1
        return counts


# ==================== Default Data Seeder ====================

def seed_default_agents(storage: CasesStorage):
    """Crear agentes predefinidos si no existen"""
    if storage.list_agents():
        return

    agents = [
        AgentDefinition(
            id="agent-visual-forense",
            name="Visual Forense",
            description="Agente especializado en análisis forense de imágenes. Examina evidencias visuales, detecta manipulaciones, analiza metadatos EXIF y genera reportes detallados.",
            icon="fas fa-microscope",
            color="#e63946",
            category="forense",
            system_prompt="Eres un agente forense visual experto. Analiza imágenes con rigor científico, documenta hallazgos con precisión y genera reportes estructurados.",
            agent_md="""# AGENTS.md - Visual Forense

## Rol
Agente especializado en análisis forense de imágenes para investigaciones.

## Capacidades
- Análisis de metadatos EXIF/IPTC
- Detección de manipulación de imágenes (ELA, clonación, splicing)
- Análisis de iluminación y sombras
- Extracción de información geográfica
- Generación de reportes forenses firmados

## Instrucciones
1. Siempre documentar la cadena de custodia digital
2. Usar técnicas no destructivas de análisis
3. Reportar nivel de confianza en cada hallazgo
4. Generar hash SHA-256 de cada evidencia procesada
5. Responder siempre en español

## Formato de Salida
- Resumen ejecutivo
- Hallazgos detallados con evidencia visual
- Conclusiones con nivel de confianza
- Recomendaciones para investigación adicional
""",
            skills=["skill-analisis-color", "skill-mejora-imagen", "skill-deteccion-patentes"],
        ),
        AgentDefinition(
            id="agent-patentes",
            name="Especialista en Patentes",
            description="Agente experto en detección, extracción y mejora de patentes vehiculares en imágenes de vigilancia.",
            icon="fas fa-car",
            color="#457b9d",
            category="vehiculos",
            system_prompt="Eres un especialista en detección de patentes vehiculares. Usa técnicas multi-método para detectar, extraer y mejorar patentes.",
            agent_md="""# AGENTS.md - Especialista en Patentes

## Rol
Agente especializado en detección y lectura de patentes vehiculares.

## Capacidades
- Detección multi-método de patentes (contornos, color, Haar cascades)
- OCR especializado para patentes chilenas y sudamericanas
- Mejora de imagen para patentes borrosas o de baja resolución
- Scoring de confianza por método de detección
- Validación de formato de patente por país

## Instrucciones
1. NUNCA usar coordenadas hardcodeadas
2. Siempre usar detección automática con computer vision
3. Aplicar múltiples métodos y reportar el de mayor confianza
4. Mejorar la imagen antes de OCR si la resolución es baja
5. Validar el formato de la patente detectada

## Formato de Salida
- Patente detectada (texto)
- Método de detección usado
- Score de confianza (0-100)
- Imagen recortada y mejorada de la patente
""",
            skills=["skill-deteccion-patentes", "skill-mejora-imagen"],
        ),
        AgentDefinition(
            id="agent-ocupantes",
            name="Analista de Ocupantes",
            description="Agente especializado en detectar, contar y analizar ocupantes dentro de vehículos desde imágenes de vigilancia.",
            icon="fas fa-users",
            color="#2a9d8f",
            category="personas",
            system_prompt="Eres un analista de ocupantes vehiculares. Detecta y cuenta personas dentro de vehículos usando técnicas de computer vision.",
            agent_md="""# AGENTS.md - Analista de Ocupantes

## Rol
Agente especializado en detección y análisis de ocupantes vehiculares.

## Capacidades
- Detección de rostros y siluetas dentro de vehículos
- Conteo de ocupantes con nivel de confianza
- Estimación de posición (conductor, copiloto, trasero)
- Análisis de visibilidad y obstrucciones
- Detección de objetos en manos de ocupantes

## Instrucciones
1. Segmentar primero el área del vehículo
2. Analizar ventanas y áreas visibles
3. Usar múltiples técnicas de detección
4. Reportar confianza por cada ocupante detectado
5. Documentar limitaciones de visibilidad

## Formato de Salida
- Número de ocupantes detectados
- Posición estimada de cada uno
- Nivel de confianza por detección
- Imagen anotada con bounding boxes
""",
            skills=["skill-deteccion-ocupantes", "skill-segmentacion-vehiculos"],
        ),
        AgentDefinition(
            id="agent-descriptor",
            name="Descriptor de Imágenes",
            description="Agente especializado en generar descripciones exhaustivas de imágenes para indexación en bases de datos vectoriales.",
            icon="fas fa-file-alt",
            color="#e9c46a",
            category="indexacion",
            system_prompt="Eres un descriptor de imágenes experto. Genera descripciones exhaustivas y estructuradas optimizadas para búsqueda semántica.",
            agent_md="""# AGENTS.md - Descriptor de Imágenes

## Rol
Agente especializado en generación de descripciones para indexación vectorial.

## Capacidades
- Descripción exhaustiva de contenido visual
- Extracción de atributos estructurados
- Generación de embeddings textuales optimizados
- Clasificación multi-etiqueta
- Descripción de contexto y escena

## Instrucciones
1. Describir todos los elementos visibles
2. Incluir colores, formas, posiciones relativas
3. Describir el contexto y la escena general
4. Usar vocabulario consistente y preciso
5. Estructurar la descripción para máxima recuperabilidad

## Formato de Salida
- Descripción general de la escena
- Objetos detectados con atributos
- Colores dominantes
- Condiciones ambientales
- Texto visible (si aplica)
- Tags/etiquetas sugeridas
""",
            skills=["skill-descripcion-vectorial", "skill-analisis-color"],
        ),
    ]

    for agent in agents:
        storage.save_agent(agent)


def seed_default_skills(storage: CasesStorage):
    """Crear skills predefinidos si no existen"""
    if storage.list_skills():
        return

    skills = [
        SkillDefinition(
            id="skill-analisis-color",
            name="analisis-color-vehicular",
            description="Técnicas especializadas para identificar y clasificar colores de vehículos con precisión, incluyendo colores primarios y secundarios, acabados metálicos y variaciones por iluminación.",
            category="analisis",
            icon="fas fa-palette",
            color="#e63946",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Paso 1: Segmentación del vehículo
Aislar el vehículo del fondo usando técnicas de segmentación.

### Paso 2: Análisis de color dominante
- Convertir a espacio de color HSV para mejor discriminación
- Aplicar K-means clustering para encontrar colores dominantes
- Filtrar píxeles de fondo, sombras y reflejos

### Paso 3: Clasificación
Clasificar el color en categorías estándar:
- Blanco, Negro, Gris/Plata, Rojo, Azul, Verde, Amarillo, Naranja, Marrón, Beige

### Paso 4: Reportar
- Color primario con porcentaje de confianza
- Color secundario si aplica
- Tipo de acabado (metálico, mate, perlado)

### Ejemplo de salida
```
Color primario: Blanco (92%)
Color secundario: Gris plata (8%)
Acabado: Perlado
```
""",
        ),
        SkillDefinition(
            id="skill-segmentacion-vehiculos",
            name="segmentacion-vehiculos",
            description="Técnicas de segmentación para aislar vehículos del fondo usando GrabCut, contornos y análisis de color.",
            category="segmentacion",
            icon="fas fa-crop-alt",
            color="#457b9d",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Técnica 1: GrabCut
1. Definir rectángulo inicial alrededor del vehículo
2. Aplicar GrabCut iterativo (5-10 iteraciones)
3. Refinar máscara con operaciones morfológicas

### Técnica 2: Detección de contornos
1. Convertir a escala de grises
2. Aplicar Canny edge detection
3. Encontrar contornos y filtrar por área
4. Seleccionar el contorno más grande como vehículo

### Técnica 3: Análisis de color
1. Convertir a HSV
2. Aplicar umbralización adaptativa
3. Combinar con detección de bordes

### Resultado esperado
- Máscara binaria del vehículo
- Imagen recortada con fondo transparente
- Bounding box del vehículo
""",
        ),
        SkillDefinition(
            id="skill-mejora-imagen",
            name="mejora-imagenes",
            description="Pipeline completo de mejora de calidad de imagen: CLAHE, denoising, sharpening, super-resolución y preparación para OCR.",
            category="mejora",
            icon="fas fa-magic",
            color="#2a9d8f",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Pipeline de mejora
1. **Denoising**: Aplicar fastNlMeansDenoisingColored
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Sharpening**: Unsharp mask con kernel personalizado
4. **Super-resolución**: Interpolación bicúbica o ESRGAN si disponible
5. **Preparación OCR**: Binarización Otsu + dilatación

### Parámetros recomendados
```python
# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Denoising
cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Sharpening kernel
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
```

### Resultado esperado
- Imagen mejorada en resolución y contraste
- Imagen preparada para OCR (blanco y negro, alto contraste)
""",
        ),
        SkillDefinition(
            id="skill-descripcion-vectorial",
            name="descripcion-base-vectorial",
            description="Genera descripciones exhaustivas y estructuradas de imágenes optimizadas para indexación en bases de datos vectoriales y búsqueda semántica.",
            category="analisis",
            icon="fas fa-database",
            color="#e9c46a",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Estructura de descripción
Generar una descripción con las siguientes secciones:

1. **Escena general**: Tipo de ubicación, hora del día, condiciones
2. **Vehículos**: Tipo, color, marca estimada, estado, posición
3. **Personas**: Cantidad, vestimenta, actividad, posición
4. **Objetos**: Señalización, mobiliario urbano, elementos relevantes
5. **Texto visible**: Patentes, carteles, señales
6. **Contexto**: Tipo de vía, tráfico, clima visible

### Formato de salida
```
ESCENA: [descripción de la escena]
VEHICULOS: [lista de vehículos con atributos]
PERSONAS: [lista de personas con atributos]
OBJETOS: [objetos relevantes]
TEXTO: [texto visible]
TAGS: [lista de etiquetas separadas por coma]
```

### Optimización para búsqueda
- Usar sinónimos y variaciones de términos
- Incluir tanto términos técnicos como coloquiales
- Describir relaciones espaciales entre elementos
""",
        ),
        SkillDefinition(
            id="skill-estimacion-genero-edad",
            name="estimacion-genero-edad",
            description="Técnicas de análisis visual para estimar género y rango etario de personas en imágenes de vigilancia.",
            category="analisis",
            icon="fas fa-user-circle",
            color="#f4a261",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Paso 1: Detección de rostros
- Usar Haar cascades o DNN face detector
- Extraer ROI del rostro con margen del 20%

### Paso 2: Estimación de edad
- Clasificar en rangos: 0-12, 13-17, 18-25, 26-35, 36-50, 51-65, 65+
- Reportar rango más probable con confianza

### Paso 3: Estimación de género
- Clasificar como masculino/femenino
- Reportar confianza de la estimación

### Consideraciones éticas
- Estas son estimaciones basadas en apariencia visual
- No deben usarse como determinación definitiva
- Siempre reportar como "estimación" con nivel de confianza
- Respetar la privacidad y normativas locales

### Resultado esperado
- Género estimado con confianza
- Rango etario estimado con confianza
- Imagen anotada con detecciones
""",
        ),
        SkillDefinition(
            id="skill-analisis-vestimenta",
            name="analisis-vestimenta",
            description="Técnicas para analizar vestimenta, colores, patrones y accesorios de personas detectadas en imágenes.",
            category="analisis",
            icon="fas fa-tshirt",
            color="#264653",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Paso 1: Segmentación de persona
- Detectar y segmentar la persona completa
- Dividir en regiones: cabeza, torso superior, torso inferior, extremidades

### Paso 2: Análisis por región
- **Cabeza**: Sombrero, gorra, cabello visible, gafas
- **Torso superior**: Tipo de prenda, color, patrón, logo visible
- **Torso inferior**: Pantalón/falda, color, tipo
- **Calzado**: Tipo, color

### Paso 3: Colores dominantes
- Extraer colores dominantes por región
- Clasificar en categorías estándar

### Resultado esperado
```
CABEZA: Gorra negra, cabello oscuro
TORSO_SUPERIOR: Chaqueta azul oscuro, tipo deportiva
TORSO_INFERIOR: Jeans azul claro
CALZADO: Zapatillas blancas
ACCESORIOS: Mochila gris
```
""",
        ),
        SkillDefinition(
            id="skill-deteccion-ocupantes",
            name="deteccion-ocupantes",
            description="Técnicas de computer vision para detectar, contar y localizar ocupantes dentro de vehículos desde imágenes de vigilancia.",
            category="deteccion",
            icon="fas fa-user-friends",
            color="#e76f51",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Paso 1: Localización del vehículo
- Detectar el vehículo en la imagen
- Identificar las ventanas (parabrisas, laterales)

### Paso 2: Análisis de ventanas
- Segmentar cada ventana visible
- Analizar transparencia y visibilidad

### Paso 3: Detección de ocupantes
- Buscar rostros/siluetas en cada ventana
- Usar detección de piel y formas humanas
- Analizar sombras y siluetas

### Paso 4: Conteo y localización
- Contar ocupantes detectados
- Asignar posición: conductor, copiloto, trasero izq/centro/der

### Resultado esperado
- Número total de ocupantes
- Posición de cada uno
- Confianza por detección
- Imagen anotada
""",
        ),
        SkillDefinition(
            id="skill-deteccion-patentes",
            name="deteccion-patentes",
            description="Técnica multi-método con scoring para detectar, extraer y mejorar patentes vehiculares chilenas y sudamericanas.",
            category="deteccion",
            icon="fas fa-id-card",
            color="#6a4c93",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones

### Método 1: Detección por contornos (score base: 60)
1. Convertir a escala de grises
2. Aplicar blur y Canny
3. Buscar contornos rectangulares con aspect ratio 2:1 a 5:1
4. Filtrar por tamaño mínimo

### Método 2: Detección por color (score base: 50)
1. Buscar regiones blancas/claras (patentes chilenas)
2. Filtrar por forma rectangular
3. Validar con aspect ratio

### Método 3: Haar Cascades (score base: 70)
1. Usar clasificador entrenado para patentes
2. Detectar en múltiples escalas

### Scoring
- Cada método genera candidatos con score base
- Bonus +10 si aspect ratio es 3:1 ± 0.5
- Bonus +10 si el tamaño es razonable (>1% de la imagen)
- Bonus +15 si OCR detecta patrón de patente válido

### Formato de patente chilena
- Antiguo: XX-0000 (2 letras, 4 números)
- Nuevo: XXXX-00 (4 letras, 2 números)

### Resultado esperado
- Texto de patente detectada
- Score de confianza (0-100)
- Método de detección
- Imagen recortada y mejorada
""",
        ),
        SkillDefinition(
            id="skill-creador-skills",
            name="creador-de-skills",
            description="Skill especial que permite crear nuevos skills siguiendo el estándar Agent Skills (agentskills.io). Guía al agente en la creación de SKILL.md con frontmatter válido e instrucciones estructuradas.",
            category="meta",
            icon="fas fa-plus-circle",
            color="#4361ee",
            metadata={"author": "sitia", "version": "1.0"},
            skill_md="""## Instrucciones para crear un nuevo Skill

### Paso 1: Definir el skill
Solicitar al usuario:
- **Nombre**: Lowercase, solo letras y guiones (ej: `analisis-facial`)
- **Descripción**: Qué hace el skill y cuándo usarlo (max 1024 chars)
- **Categoría**: analisis, deteccion, mejora, segmentacion, meta, general

### Paso 2: Crear el SKILL.md
El archivo debe seguir este formato:

```yaml
---
name: nombre-del-skill
description: Descripción completa del skill.
metadata:
  author: sitia
  version: "1.0"
---
```

### Paso 3: Escribir las instrucciones
El body del SKILL.md debe incluir:
1. **Instrucciones paso a paso**: Cómo ejecutar la tarea
2. **Ejemplos**: Inputs y outputs esperados
3. **Casos edge**: Situaciones especiales a manejar
4. **Parámetros**: Configuraciones recomendadas

### Paso 4: Validar
- El nombre debe coincidir con el directorio
- La descripción no debe estar vacía
- Las instrucciones deben ser claras y accionables

### Paso 5: Registrar
Guardar el skill en el sistema para que esté disponible para los agentes.

### Plantilla de ejemplo
```markdown
## Instrucciones

### Paso 1: [Nombre del paso]
[Descripción detallada]

### Paso 2: [Nombre del paso]
[Descripción detallada]

### Resultado esperado
[Formato de salida esperado]
```
""",
        ),
    ]

    for skill in skills:
        storage.save_skill(skill)
