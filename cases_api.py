"""
Cases API - Rutas FastAPI para el módulo de Casos.
Se monta como un sub-router en la aplicación principal.
"""

import os
import uuid
import shutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from cases_module import (
    AgentDefinition,
    Case,
    CaseFile,
    CaseFolder,
    CasesStorage,
    EvidenceRequest,
    SkillDefinition,
    Task,
    WorkflowStep,
    seed_default_agents,
    seed_default_skills,
)

router = APIRouter(prefix="/api/cases", tags=["cases"])
cases_storage = CasesStorage()

# Seed defaults on import
seed_default_agents(cases_storage)
seed_default_skills(cases_storage)


# ==================== Pydantic Models ====================

class AgentCreate(BaseModel):
    name: str
    description: str
    icon: str = "fas fa-robot"
    color: str = "#4361ee"
    category: str = "general"
    llm_config_name: Optional[str] = None
    system_prompt: str = ""
    agent_md: str = ""
    skills: List[str] = []

class SkillCreate(BaseModel):
    name: str
    description: str
    category: str = "general"
    icon: str = "fas fa-cog"
    color: str = "#4361ee"
    license: str = ""
    compatibility: str = ""
    metadata: Dict[str, str] = {}
    allowed_tools: str = ""
    skill_md: str = ""
    scripts: Dict[str, str] = {}
    references: Dict[str, str] = {}

class CaseCreate(BaseModel):
    name: str
    description: str
    case_type: str = "investigacion"
    assigned_users: List[str] = []
    tags: List[str] = []

class CaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    case_type: Optional[str] = None
    status: Optional[str] = None
    assigned_users: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class WorkflowStepCreate(BaseModel):
    name: str
    order: int = 0
    agent_id: Optional[str] = None
    skill_ids: List[str] = []
    prompt: str = ""
    llm_config_name: Optional[str] = None
    use_previous_output: bool = True
    index_for_search: bool = False
    estimated_duration_min: int = 5

class TaskCreate(BaseModel):
    case_id: str
    name: str
    description: str = ""
    task_type: str = "humana"
    priority: str = "media"
    assigned_to: str = ""
    agent_id: Optional[str] = None
    workflow_steps: List[WorkflowStepCreate] = []
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None
    recurrence_time: Optional[str] = None
    resource_folder_ids: List[str] = []
    due_date: Optional[str] = None

class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_steps: Optional[List[WorkflowStepCreate]] = None
    due_date: Optional[str] = None

class FolderCreate(BaseModel):
    case_id: str
    name: str
    parent_id: Optional[str] = None
    folder_type: str = "general"

class EvidenceRequestCreate(BaseModel):
    case_id: str
    description: str
    polygon_coords: List[Dict[str, float]] = []
    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_m: float = 0.0


# ==================== Dashboard ====================

@router.get("/dashboard")
async def get_dashboard():
    """Obtener estadísticas del dashboard"""
    return cases_storage.get_dashboard_stats()


# ==================== Agents ====================

@router.get("/agents")
async def list_agents():
    return cases_storage.list_agents()

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = cases_storage.load_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()

@router.post("/agents")
async def create_agent(data: AgentCreate):
    agent = AgentDefinition(
        id=f"agent-{uuid.uuid4().hex[:8]}",
        name=data.name,
        description=data.description,
        icon=data.icon,
        color=data.color,
        category=data.category,
        llm_config_name=data.llm_config_name,
        system_prompt=data.system_prompt,
        agent_md=data.agent_md,
        skills=data.skills,
        is_default=False,
    )
    cases_storage.save_agent(agent)
    return {"status": "ok", "id": agent.id}

@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, data: AgentCreate):
    existing = cases_storage.load_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")
    existing.name = data.name
    existing.description = data.description
    existing.icon = data.icon
    existing.color = data.color
    existing.category = data.category
    existing.llm_config_name = data.llm_config_name
    existing.system_prompt = data.system_prompt
    existing.agent_md = data.agent_md
    existing.skills = data.skills
    existing.updated_at = __import__("datetime").datetime.now().isoformat()
    cases_storage.save_agent(existing)
    return {"status": "ok", "id": agent_id}

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if cases_storage.delete_agent(agent_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Agent not found")


# ==================== Skills ====================

@router.get("/skills")
async def list_skills():
    return cases_storage.list_skills()

@router.get("/skills/{skill_id}")
async def get_skill(skill_id: str):
    skill = cases_storage.load_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill.to_dict()

@router.post("/skills")
async def create_skill(data: SkillCreate):
    skill = SkillDefinition(
        id=f"skill-{uuid.uuid4().hex[:8]}",
        name=data.name,
        description=data.description,
        category=data.category,
        icon=data.icon,
        color=data.color,
        license=data.license,
        compatibility=data.compatibility,
        metadata=data.metadata,
        allowed_tools=data.allowed_tools,
        skill_md=data.skill_md,
        scripts=data.scripts,
        references=data.references,
        is_default=False,
    )
    cases_storage.save_skill(skill)
    return {"status": "ok", "id": skill.id}

@router.put("/skills/{skill_id}")
async def update_skill(skill_id: str, data: SkillCreate):
    existing = cases_storage.load_skill(skill_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Skill not found")
    existing.name = data.name
    existing.description = data.description
    existing.category = data.category
    existing.icon = data.icon
    existing.color = data.color
    existing.license = data.license
    existing.compatibility = data.compatibility
    existing.metadata = data.metadata
    existing.allowed_tools = data.allowed_tools
    existing.skill_md = data.skill_md
    existing.scripts = data.scripts
    existing.references = data.references
    existing.updated_at = __import__("datetime").datetime.now().isoformat()
    cases_storage.save_skill(existing)
    return {"status": "ok", "id": skill_id}

@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str):
    if cases_storage.delete_skill(skill_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Skill not found")


# ==================== Cases ====================

@router.get("/list")
async def list_cases(status: Optional[str] = None):
    return cases_storage.list_cases(status=status)

@router.get("/{case_id}")
async def get_case(case_id: str):
    case = cases_storage.load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.to_dict()

@router.post("/")
async def create_case(data: CaseCreate):
    case = Case(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        case_type=data.case_type,
        assigned_users=data.assigned_users,
        tags=data.tags,
        audit_log=[{"action": "created", "by": "admin", "at": __import__("datetime").datetime.now().isoformat()}],
    )
    cases_storage.save_case(case)
    # Create default folders
    for folder_name, folder_type in [("Evidencias", "evidencias"), ("Resultados Agentes", "resultados_agentes")]:
        folder = CaseFolder(
            id=str(uuid.uuid4()),
            case_id=case.id,
            name=folder_name,
            folder_type=folder_type,
        )
        cases_storage.save_folder(folder)
    return {"status": "ok", "id": case.id}

@router.put("/{case_id}")
async def update_case(case_id: str, data: CaseUpdate):
    case = cases_storage.load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    changes = []
    if data.name is not None:
        changes.append(f"name: {case.name} -> {data.name}")
        case.name = data.name
    if data.description is not None:
        case.description = data.description
        changes.append("description updated")
    if data.case_type is not None:
        changes.append(f"type: {case.case_type} -> {data.case_type}")
        case.case_type = data.case_type
    if data.status is not None:
        changes.append(f"status: {case.status} -> {data.status}")
        case.status = data.status
        if data.status == "archivado":
            case.archived_at = __import__("datetime").datetime.now().isoformat()
    if data.assigned_users is not None:
        case.assigned_users = data.assigned_users
        changes.append("users updated")
    if data.tags is not None:
        case.tags = data.tags
        changes.append("tags updated")
    case.updated_at = __import__("datetime").datetime.now().isoformat()
    case.audit_log.append({
        "action": "updated",
        "changes": changes,
        "by": "admin",
        "at": case.updated_at,
    })
    cases_storage.save_case(case)
    return {"status": "ok", "id": case_id}

@router.post("/{case_id}/archive")
async def archive_case(case_id: str):
    case = cases_storage.load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case.status = "archivado"
    case.archived_at = __import__("datetime").datetime.now().isoformat()
    case.updated_at = case.archived_at
    case.audit_log.append({"action": "archived", "by": "admin", "at": case.archived_at})
    cases_storage.save_case(case)
    return {"status": "ok"}

@router.post("/{case_id}/unarchive")
async def unarchive_case(case_id: str):
    case = cases_storage.load_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case.status = "abierto"
    case.archived_at = None
    case.updated_at = __import__("datetime").datetime.now().isoformat()
    case.audit_log.append({"action": "unarchived", "by": "admin", "at": case.updated_at})
    cases_storage.save_case(case)
    return {"status": "ok"}


# ==================== Tasks ====================

@router.get("/tasks/list")
async def list_tasks(case_id: Optional[str] = None, status: Optional[str] = None):
    return cases_storage.list_tasks(case_id=case_id, status=status)

@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = cases_storage.load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()

@router.post("/tasks")
async def create_task(data: TaskCreate):
    workflow_steps = []
    for i, ws in enumerate(data.workflow_steps):
        workflow_steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            name=ws.name,
            order=ws.order if ws.order else i,
            agent_id=ws.agent_id,
            skill_ids=ws.skill_ids,
            prompt=ws.prompt,
            llm_config_name=ws.llm_config_name,
            use_previous_output=ws.use_previous_output,
            index_for_search=ws.index_for_search,
            estimated_duration_min=ws.estimated_duration_min,
        ))
    task = Task(
        id=str(uuid.uuid4()),
        case_id=data.case_id,
        name=data.name,
        description=data.description,
        task_type=data.task_type,
        priority=data.priority,
        assigned_to=data.assigned_to,
        agent_id=data.agent_id,
        workflow_steps=workflow_steps,
        is_recurring=data.is_recurring,
        recurrence_pattern=data.recurrence_pattern,
        recurrence_time=data.recurrence_time,
        resource_folder_ids=data.resource_folder_ids,
        due_date=data.due_date,
    )
    cases_storage.save_task(task)
    return {"status": "ok", "id": task.id}

@router.put("/tasks/{task_id}")
async def update_task(task_id: str, data: TaskUpdate):
    task = cases_storage.load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if data.name is not None:
        task.name = data.name
    if data.description is not None:
        task.description = data.description
    if data.status is not None:
        task.status = data.status
        if data.status == "completada":
            task.completed_at = __import__("datetime").datetime.now().isoformat()
    if data.priority is not None:
        task.priority = data.priority
    if data.assigned_to is not None:
        task.assigned_to = data.assigned_to
    if data.agent_id is not None:
        task.agent_id = data.agent_id
    if data.workflow_steps is not None:
        task.workflow_steps = [
            WorkflowStep(
                id=str(uuid.uuid4()),
                name=ws.name,
                order=ws.order if ws.order else i,
                agent_id=ws.agent_id,
                skill_ids=ws.skill_ids,
                prompt=ws.prompt,
                llm_config_name=ws.llm_config_name,
                use_previous_output=ws.use_previous_output,
                index_for_search=ws.index_for_search,
                estimated_duration_min=ws.estimated_duration_min,
            ) for i, ws in enumerate(data.workflow_steps)
        ]
    if data.due_date is not None:
        task.due_date = data.due_date
    task.updated_at = __import__("datetime").datetime.now().isoformat()
    cases_storage.save_task(task)
    return {"status": "ok", "id": task_id}

@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    if cases_storage.delete_task(task_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Task not found")


# ==================== Folders ====================

@router.get("/folders/{case_id}")
async def list_folders(case_id: str):
    return cases_storage.list_folders(case_id)

@router.post("/folders")
async def create_folder(data: FolderCreate):
    folder = CaseFolder(
        id=str(uuid.uuid4()),
        case_id=data.case_id,
        name=data.name,
        parent_id=data.parent_id,
        folder_type=data.folder_type,
    )
    cases_storage.save_folder(folder)
    return {"status": "ok", "id": folder.id}

@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str):
    if cases_storage.delete_folder(folder_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Folder not found")


# ==================== Files ====================

@router.get("/files/{case_id}")
async def list_files(case_id: str, folder_id: Optional[str] = None):
    return cases_storage.list_files(case_id=case_id, folder_id=folder_id)

@router.post("/files/upload")
async def upload_file(
    case_id: str = Form(...),
    folder_id: str = Form(...),
    file: UploadFile = File(...),
):
    file_id = str(uuid.uuid4())
    files_dir = cases_storage.get_files_dir(case_id)
    ext = os.path.splitext(file.filename)[1] if file.filename else ""
    stored_name = f"{file_id}{ext}"
    file_path = os.path.join(files_dir, stored_name)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    file_meta = CaseFile(
        id=file_id,
        case_id=case_id,
        folder_id=folder_id,
        filename=stored_name,
        original_name=file.filename or "unknown",
        file_type=file.content_type or "",
        file_size=len(content),
    )
    cases_storage.save_file_meta(file_meta)
    return {"status": "ok", "id": file_id, "filename": stored_name}

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    meta = cases_storage.load_file_meta(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")
    # Delete physical file
    files_dir = cases_storage.get_files_dir(meta.case_id)
    file_path = os.path.join(files_dir, meta.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    cases_storage.delete_file_meta(file_id)
    return {"status": "ok"}


# ==================== Evidence Requests ====================

@router.get("/evidence/{case_id}")
async def list_evidence_requests(case_id: str):
    return cases_storage.list_evidence_requests(case_id=case_id)

@router.post("/evidence")
async def create_evidence_request(data: EvidenceRequestCreate):
    req = EvidenceRequest(
        id=str(uuid.uuid4()),
        case_id=data.case_id,
        description=data.description,
        polygon_coords=data.polygon_coords,
        center_lat=data.center_lat,
        center_lng=data.center_lng,
        radius_m=data.radius_m,
    )
    cases_storage.save_evidence_request(req)
    return {"status": "ok", "id": req.id}

@router.delete("/evidence/{req_id}")
async def delete_evidence_request(req_id: str):
    if cases_storage.delete_evidence_request(req_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Evidence request not found")
