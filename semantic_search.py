"""
Semantic Search Module - Indexación y búsqueda semántica de imágenes
Usa FAISS para búsqueda vectorial y SQLite en memoria para búsqueda relacional.
Los embeddings se generan usando el LLM configurado o sentence-transformers local.
"""

import json
import os
import sqlite3
import hashlib
import pickle
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


@dataclass
class SearchConfig:
    """Configuración del motor de búsqueda semántica"""
    embedding_model: str = "all-MiniLM-L6-v2"  # Modelo de sentence-transformers
    embedding_dim: int = 384  # Dimensión del embedding (depende del modelo)
    use_gpu: bool = False
    faiss_index_type: str = "flat"  # flat, ivf, hnsw
    nprobe: int = 10  # Para IVF
    ef_search: int = 64  # Para HNSW
    top_k_default: int = 10
    similarity_threshold: float = 0.3
    
    # Modelos disponibles y sus dimensiones
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "use_gpu": self.use_gpu,
            "faiss_index_type": self.faiss_index_type,
            "nprobe": self.nprobe,
            "ef_search": self.ef_search,
            "top_k_default": self.top_k_default,
            "similarity_threshold": self.similarity_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IndexedImage:
    """Imagen indexada con sus metadatos"""
    id: str
    image_path: str
    image_name: str
    image_hash: str
    descriptions: List[str] = field(default_factory=list)
    combined_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_id: str = ""
    pipeline_name: str = ""
    execution_id: str = ""
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    indexed_at: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Resultado de una búsqueda semántica"""
    image: IndexedImage
    score: float
    rank: int
    match_type: str = "semantic"  # semantic, relational, hybrid


class EmbeddingEngine:
    """Motor de embeddings usando sentence-transformers"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.model = None
        self._loaded = False
    
    def load(self):
        """Carga el modelo de embeddings"""
        if self._loaded:
            return
        
        if not SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Instálalo con: pip install sentence-transformers"
            )
        
        device = "cuda" if self.config.use_gpu else "cpu"
        self.model = SentenceTransformer(self.config.embedding_model, device=device)
        self._loaded = True
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de textos"""
        if not self._loaded:
            self.load()
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Genera embedding para un solo texto"""
        return self.encode([text])[0]


class FAISSIndex:
    """Índice FAISS para búsqueda vectorial"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index = None
        self.id_map: List[str] = []  # Mapeo de posición FAISS -> image_id
        self._initialized = False
    
    def initialize(self):
        """Inicializa el índice FAISS"""
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS no está instalado. "
                "Instálalo con: pip install faiss-cpu"
            )
        
        dim = self.config.embedding_dim
        
        if self.config.faiss_index_type == "flat":
            # Búsqueda exacta - mejor para datasets pequeños/medianos
            self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine con vectores normalizados)
        elif self.config.faiss_index_type == "ivf":
            # IVF - más rápido para datasets grandes
            quantizer = faiss.IndexFlatIP(dim)
            nlist = max(1, min(100, len(self.id_map) // 10)) if self.id_map else 10
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.config.faiss_index_type == "hnsw":
            # HNSW - buen balance velocidad/precisión
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = self.config.ef_search
        else:
            self.index = faiss.IndexFlatIP(dim)
        
        self._initialized = True
    
    def add(self, image_id: str, embedding: np.ndarray):
        """Agrega un embedding al índice"""
        if not self._initialized:
            self.initialize()
        
        embedding = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding)
        self.id_map.append(image_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Busca los top_k vectores más similares"""
        if not self._initialized or self.index.ntotal == 0:
            return []
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        
        if self.config.faiss_index_type == "ivf" and hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.nprobe
        
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        
        return results
    
    def remove(self, image_id: str):
        """Marca una imagen como eliminada (reconstruye el índice sin ella)"""
        if image_id in self.id_map:
            idx = self.id_map.index(image_id)
            self.id_map[idx] = None  # Marcar como eliminada
    
    def save(self, path: str):
        """Guarda el índice FAISS a disco"""
        if self._initialized and self.index is not None:
            faiss.write_index(self.index, path + ".faiss")
            with open(path + ".map", "wb") as f:
                pickle.dump(self.id_map, f)
    
    def load(self, path: str):
        """Carga el índice FAISS desde disco"""
        faiss_path = path + ".faiss"
        map_path = path + ".map"
        
        if os.path.exists(faiss_path) and os.path.exists(map_path):
            self.index = faiss.read_index(faiss_path)
            with open(map_path, "rb") as f:
                self.id_map = pickle.load(f)
            self._initialized = True
            return True
        return False
    
    @property
    def total(self) -> int:
        return self.index.ntotal if self._initialized and self.index else 0


class RelationalDB:
    """Base de datos relacional SQLite en memoria para búsqueda por atributos"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Crea las tablas necesarias"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                image_name TEXT NOT NULL,
                image_hash TEXT,
                combined_text TEXT,
                pipeline_id TEXT,
                pipeline_name TEXT,
                execution_id TEXT,
                indexed_at TEXT,
                context_data_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                step_name TEXT,
                step_order INTEGER,
                description TEXT NOT NULL,
                model_used TEXT,
                provider_used TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                attribute_key TEXT NOT NULL,
                attribute_value TEXT NOT NULL,
                source_step TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        # Índices para búsqueda rápida
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_desc_image ON image_descriptions(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_image ON image_attributes(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_key ON image_attributes(attribute_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_value ON image_attributes(attribute_value)")
        
        # FTS5 para búsqueda full-text
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
                image_id,
                combined_text,
                content='images',
                content_rowid='rowid'
            )
        """)
        
        self.conn.commit()
    
    def insert_image(self, image: IndexedImage):
        """Inserta o actualiza una imagen en la base de datos"""
        cursor = self.conn.cursor()
        
        # Insertar o reemplazar imagen
        cursor.execute("""
            INSERT OR REPLACE INTO images 
            (id, image_path, image_name, image_hash, combined_text, 
             pipeline_id, pipeline_name, execution_id, indexed_at, context_data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image.id, image.image_path, image.image_name, image.image_hash,
            image.combined_text, image.pipeline_id, image.pipeline_name,
            image.execution_id, image.indexed_at,
            json.dumps(image.context_data, ensure_ascii=False) if image.context_data else "{}"
        ))
        
        # Eliminar descripciones y atributos anteriores
        cursor.execute("DELETE FROM image_descriptions WHERE image_id = ?", (image.id,))
        cursor.execute("DELETE FROM image_attributes WHERE image_id = ?", (image.id,))
        
        # Insertar descripciones por paso
        for step in image.step_results:
            cursor.execute("""
                INSERT INTO image_descriptions 
                (image_id, step_name, step_order, description, model_used, provider_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                image.id, step.get("step_name", ""), step.get("step_order", 0),
                step.get("response_content", ""), step.get("model_used", ""),
                step.get("provider_used", "")
            ))
        
        # Extraer y guardar atributos del contexto y respuestas
        self._extract_attributes(image)
        
        # Actualizar FTS
        cursor.execute("""
            INSERT INTO image_fts (image_id, combined_text)
            VALUES (?, ?)
        """, (image.id, image.combined_text))
        
        self.conn.commit()
    
    def _extract_attributes(self, image: IndexedImage):
        """Extrae atributos estructurados de las respuestas y contexto"""
        cursor = self.conn.cursor()
        
        # Atributos del contexto JSON
        if image.context_data:
            self._flatten_and_insert(cursor, image.id, image.context_data, "context")
        
        # Intentar extraer JSON de las respuestas de cada paso
        for step in image.step_results:
            content = step.get("response_content", "")
            step_name = step.get("step_name", "unknown")
            
            # Intentar parsear JSON de la respuesta
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if not json_blocks:
                # Intentar parsear como JSON directo
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        self._flatten_and_insert(cursor, image.id, parsed, step_name)
                except (json.JSONDecodeError, ValueError):
                    pass
            else:
                for block in json_blocks:
                    try:
                        parsed = json.loads(block)
                        if isinstance(parsed, dict):
                            self._flatten_and_insert(cursor, image.id, parsed, step_name)
                    except (json.JSONDecodeError, ValueError):
                        pass
    
    def _flatten_and_insert(self, cursor, image_id: str, data: Dict, source: str, prefix: str = ""):
        """Aplana un diccionario y lo inserta como atributos"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_and_insert(cursor, image_id, value, source, full_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._flatten_and_insert(cursor, image_id, item, source, f"{full_key}[{i}]")
                    else:
                        cursor.execute("""
                            INSERT INTO image_attributes (image_id, attribute_key, attribute_value, source_step)
                            VALUES (?, ?, ?, ?)
                        """, (image_id, full_key, str(item), source))
            else:
                cursor.execute("""
                    INSERT INTO image_attributes (image_id, attribute_key, attribute_value, source_step)
                    VALUES (?, ?, ?, ?)
                """, (image_id, full_key, str(value), source))
    
    def search_fulltext(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Búsqueda full-text en las descripciones"""
        cursor = self.conn.cursor()
        
        # Preparar query para FTS5
        fts_query = " OR ".join(query.split())
        
        try:
            cursor.execute("""
                SELECT image_id, rank
                FROM image_fts
                WHERE image_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "image_id": row[0],
                    "fts_rank": row[1]
                })
            return results
        except Exception:
            # Fallback: búsqueda LIKE simple
            return self._search_like(query, limit)
    
    def _search_like(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Búsqueda LIKE como fallback"""
        cursor = self.conn.cursor()
        terms = query.lower().split()
        
        if not terms:
            return []
        
        conditions = " AND ".join(["LOWER(combined_text) LIKE ?" for _ in terms])
        params = [f"%{term}%" for term in terms]
        params.append(limit)
        
        cursor.execute(f"""
            SELECT id as image_id, 0 as fts_rank
            FROM images
            WHERE {conditions}
            LIMIT ?
        """, params)
        
        return [{"image_id": row[0], "fts_rank": row[1]} for row in cursor.fetchall()]
    
    def search_attributes(self, filters: Dict[str, str], limit: int = 50) -> List[str]:
        """Búsqueda por atributos específicos"""
        cursor = self.conn.cursor()
        
        if not filters:
            return []
        
        conditions = []
        params = []
        
        for key, value in filters.items():
            conditions.append(
                "(attribute_key LIKE ? AND LOWER(attribute_value) LIKE ?)"
            )
            params.extend([f"%{key}%", f"%{value.lower()}%"])
        
        where_clause = " OR ".join(conditions)
        params.append(limit)
        
        cursor.execute(f"""
            SELECT DISTINCT image_id
            FROM image_attributes
            WHERE {where_clause}
            LIMIT ?
        """, params)
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos completos de una imagen indexada"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        image_data = dict(row)
        
        # Obtener descripciones
        cursor.execute(
            "SELECT * FROM image_descriptions WHERE image_id = ? ORDER BY step_order",
            (image_id,)
        )
        image_data["descriptions"] = [dict(r) for r in cursor.fetchall()]
        
        # Obtener atributos
        cursor.execute(
            "SELECT * FROM image_attributes WHERE image_id = ?",
            (image_id,)
        )
        image_data["attributes"] = [dict(r) for r in cursor.fetchall()]
        
        # Parsear context_data_json
        try:
            image_data["context_data"] = json.loads(image_data.get("context_data_json", "{}"))
        except:
            image_data["context_data"] = {}
        
        return image_data
    
    def get_all_images(self) -> List[Dict[str, Any]]:
        """Obtiene todas las imágenes indexadas"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, image_path, image_name, image_hash, pipeline_name, indexed_at FROM images ORDER BY indexed_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM image_descriptions")
        total_descriptions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM image_attributes")
        total_attributes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT attribute_key) FROM image_attributes")
        unique_keys = cursor.fetchone()[0]
        
        return {
            "total_indexed_images": total_images,
            "total_descriptions": total_descriptions,
            "total_attributes": total_attributes,
            "unique_attribute_keys": unique_keys
        }
    
    def delete_image(self, image_id: str):
        """Elimina una imagen y sus datos asociados"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM image_descriptions WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM image_attributes WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self.conn.commit()
    
    def export_data(self) -> Dict[str, Any]:
        """Exporta todos los datos para persistencia"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM images")
        images = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("SELECT * FROM image_descriptions")
        descriptions = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("SELECT * FROM image_attributes")
        attributes = [dict(row) for row in cursor.fetchall()]
        
        return {
            "images": images,
            "descriptions": descriptions,
            "attributes": attributes
        }
    
    def import_data(self, data: Dict[str, Any]):
        """Importa datos desde un export"""
        cursor = self.conn.cursor()
        
        for img in data.get("images", []):
            cursor.execute("""
                INSERT OR REPLACE INTO images 
                (id, image_path, image_name, image_hash, combined_text,
                 pipeline_id, pipeline_name, execution_id, indexed_at, context_data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                img["id"], img["image_path"], img["image_name"],
                img.get("image_hash", ""), img.get("combined_text", ""),
                img.get("pipeline_id", ""), img.get("pipeline_name", ""),
                img.get("execution_id", ""), img.get("indexed_at", ""),
                img.get("context_data_json", "{}")
            ))
        
        for desc in data.get("descriptions", []):
            cursor.execute("""
                INSERT INTO image_descriptions 
                (image_id, step_name, step_order, description, model_used, provider_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                desc["image_id"], desc.get("step_name", ""),
                desc.get("step_order", 0), desc.get("description", ""),
                desc.get("model_used", ""), desc.get("provider_used", "")
            ))
        
        for attr in data.get("attributes", []):
            cursor.execute("""
                INSERT INTO image_attributes 
                (image_id, attribute_key, attribute_value, source_step)
                VALUES (?, ?, ?, ?)
            """, (
                attr["image_id"], attr["attribute_key"],
                attr["attribute_value"], attr.get("source_step", "")
            ))
        
        self.conn.commit()


class SemanticSearchEngine:
    """Motor principal de búsqueda semántica que combina FAISS + SQLite"""
    
    def __init__(self, data_dir: str, config: SearchConfig = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or SearchConfig()
        self._config_path = self.data_dir / "search_config.json"
        self._index_path = str(self.data_dir / "faiss_index")
        self._db_export_path = self.data_dir / "relational_db.json"
        
        # Cargar configuración guardada
        self._load_config()
        
        # Inicializar componentes
        self.embedding_engine = EmbeddingEngine(self.config)
        self.faiss_index = FAISSIndex(self.config)
        
        # SQLite en archivo para persistencia
        db_path = str(self.data_dir / "search.db")
        self.relational_db = RelationalDB(db_path)
        
        # Intentar cargar índice FAISS existente
        self.faiss_index.load(self._index_path)
        
        self._embeddings_loaded = False
    
    def _load_config(self):
        """Carga la configuración guardada"""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    data = json.load(f)
                self.config = SearchConfig.from_dict(data)
            except Exception:
                pass
    
    def save_config(self, config: SearchConfig = None):
        """Guarda la configuración"""
        if config:
            self.config = config
            # Reinicializar componentes con nueva configuración
            self.embedding_engine = EmbeddingEngine(self.config)
            self.faiss_index = FAISSIndex(self.config)
            self.faiss_index.initialize()
        
        with open(self._config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _ensure_embeddings_loaded(self):
        """Asegura que el modelo de embeddings está cargado"""
        if not self._embeddings_loaded:
            try:
                self.embedding_engine.load()
                self._embeddings_loaded = True
            except ImportError as e:
                raise ImportError(str(e))
    
    def _compute_image_hash(self, image_path: str) -> str:
        """Calcula el hash SHA-256 de una imagen"""
        sha256 = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def index_execution(self, execution_data: Dict[str, Any], 
                        indexable_steps: List[str] = None) -> Optional[str]:
        """
        Indexa una ejecución de pipeline.
        
        Args:
            execution_data: Datos completos de la ejecución
            indexable_steps: Lista de step_ids que deben indexarse.
                           Si es None, se indexan todos los pasos marcados como indexables.
        
        Returns:
            ID de la imagen indexada o None si falla
        """
        self._ensure_embeddings_loaded()
        
        image_path = execution_data.get("image_path", "")
        image_name = execution_data.get("image_name", "")
        
        if not image_path or not os.path.exists(image_path):
            return None
        
        # Recopilar textos de los pasos indexables
        descriptions = []
        step_results_data = []
        
        for step in execution_data.get("step_results", []):
            step_id = step.get("step_id", "")
            
            # Si hay lista de pasos indexables, filtrar
            if indexable_steps is not None and step_id not in indexable_steps:
                continue
            
            response = step.get("response", {})
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            
            if content:
                descriptions.append(content)
                step_results_data.append({
                    "step_name": step.get("step_name", ""),
                    "step_order": step.get("step_order", 0),
                    "response_content": content,
                    "model_used": step.get("model_used", ""),
                    "provider_used": step.get("provider_used", "")
                })
        
        if not descriptions:
            return None
        
        # Crear texto combinado
        combined_text = "\n\n".join(descriptions)
        
        # Generar ID único basado en imagen + ejecución
        image_hash = self._compute_image_hash(image_path)
        image_id = f"{image_hash[:16]}_{execution_data.get('id', '')[:8]}"
        
        # Crear objeto IndexedImage
        indexed_image = IndexedImage(
            id=image_id,
            image_path=image_path,
            image_name=image_name,
            image_hash=image_hash,
            descriptions=descriptions,
            combined_text=combined_text,
            metadata={
                "total_latency_ms": execution_data.get("total_latency_ms", 0),
                "total_tokens": execution_data.get("total_tokens", 0),
                "models_used": execution_data.get("models_used", [])
            },
            pipeline_id=execution_data.get("pipeline_id", ""),
            pipeline_name=execution_data.get("pipeline_name", ""),
            execution_id=execution_data.get("id", ""),
            step_results=step_results_data,
            indexed_at=datetime.now().isoformat(),
            context_data=execution_data.get("context_data", {})
        )
        
        # Generar embedding del texto combinado
        embedding = self.embedding_engine.encode_single(combined_text)
        
        # Indexar en FAISS
        if not self.faiss_index._initialized:
            self.faiss_index.initialize()
        self.faiss_index.add(image_id, embedding)
        
        # Indexar en base de datos relacional
        self.relational_db.insert_image(indexed_image)
        
        # Guardar índice FAISS a disco
        self.faiss_index.save(self._index_path)
        
        return image_id
    
    def index_image(self, image_name: str, image_path: str, 
                     combined_text: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Indexa una imagen directamente con su texto descriptivo.
        
        Args:
            image_name: Nombre de la imagen
            image_path: Ruta al archivo de imagen
            combined_text: Texto combinado de las descripciones
            metadata: Metadatos adicionales (pipeline_name, execution_id, context_data, etc.)
        
        Returns:
            ID de la imagen indexada o None si falla
        """
        if not combined_text or not combined_text.strip():
            return None
        
        try:
            self._ensure_embeddings_loaded()
        except ImportError as e:
            print(f"Warning: Cannot load embeddings: {e}")
            return None
        
        metadata = metadata or {}
        
        # Calcular hash de la imagen si existe
        image_hash = ""
        if image_path and os.path.exists(image_path):
            image_hash = self._compute_image_hash(image_path)
        
        # Generar ID único
        execution_id = metadata.get("execution_id", "")
        image_id = f"{image_hash[:16]}_{execution_id[:8]}" if image_hash else f"{hashlib.md5(image_name.encode()).hexdigest()[:16]}_{execution_id[:8]}"
        
        # Crear objeto IndexedImage
        step_results_data = []
        if metadata.get("step_results"):
            step_results_data = metadata["step_results"]
        else:
            # Crear un step_result genérico con el texto combinado
            step_results_data = [{
                "step_name": "combined_analysis",
                "step_order": 0,
                "response_content": combined_text,
                "model_used": metadata.get("model_used", ""),
                "provider_used": metadata.get("provider_used", "")
            }]
        
        indexed_image = IndexedImage(
            id=image_id,
            image_path=image_path,
            image_name=image_name,
            image_hash=image_hash,
            descriptions=[combined_text],
            combined_text=combined_text,
            metadata={
                "total_latency_ms": metadata.get("total_latency_ms", 0),
                "total_tokens": metadata.get("total_tokens", 0),
                "models_used": metadata.get("models_used", [])
            },
            pipeline_id=metadata.get("pipeline_id", ""),
            pipeline_name=metadata.get("pipeline_name", ""),
            execution_id=execution_id,
            step_results=step_results_data,
            indexed_at=datetime.now().isoformat(),
            context_data=metadata.get("context_data", {})
        )
        
        # Generar embedding del texto combinado
        embedding = self.embedding_engine.encode_single(combined_text)
        
        # Indexar en FAISS
        if not self.faiss_index._initialized:
            self.faiss_index.initialize()
        self.faiss_index.add(image_id, embedding)
        
        # Indexar en base de datos relacional
        self.relational_db.insert_image(indexed_image)
        
        # Guardar índice FAISS a disco
        self.faiss_index.save(self._index_path)
        
        return image_id

    def search(self, query: str, top_k: int = None, 
               mode: str = "hybrid", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares a la query.
        
        Args:
            query: Texto de búsqueda
            top_k: Número máximo de resultados
            mode: Modo de búsqueda - "semantic", "relational", "hybrid"
        
        Returns:
            Lista de SearchResult ordenados por relevancia
        """
        if top_k is None:
            top_k = self.config.top_k_default
        
        results_map: Dict[str, float] = {}
        
        # Búsqueda semántica con FAISS
        if mode in ("semantic", "hybrid"):
            try:
                self._ensure_embeddings_loaded()
                query_embedding = self.embedding_engine.encode_single(query)
                faiss_results = self.faiss_index.search(query_embedding, top_k * 2)
                
                for image_id, score in faiss_results:
                    if score >= self.config.similarity_threshold:
                        results_map[image_id] = results_map.get(image_id, 0) + score
            except Exception:
                pass
        
        # Búsqueda relacional con FTS
        if mode in ("relational", "hybrid"):
            fts_results = self.relational_db.search_fulltext(query, top_k * 2)
            
            for result in fts_results:
                image_id = result["image_id"]
                # Normalizar el score FTS (rank negativo, menor es mejor)
                fts_score = max(0, 1.0 + result["fts_rank"] * 0.1)
                results_map[image_id] = results_map.get(image_id, 0) + fts_score * 0.5
        
        # Construir resultados finales
        search_results = []
        for image_id, score in sorted(results_map.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            image_data = self.relational_db.get_image(image_id)
            if image_data:
                search_results.append({
                    "image_id": image_data["id"],
                    "image_name": image_data["image_name"],
                    "image_path": image_data["image_path"],
                    "image_hash": image_data.get("image_hash", ""),
                    "pipeline_name": image_data.get("pipeline_name", ""),
                    "execution_id": image_data.get("execution_id", ""),
                    "indexed_at": image_data.get("indexed_at", ""),
                    "score": round(score, 4),
                    "rank": len(search_results) + 1,
                    "match_type": mode,
                    "combined_text": image_data.get("combined_text", "")[:500],
                    "descriptions": [{
                        "step_name": d["step_name"],
                        "step_order": d["step_order"],
                        "description": d["description"][:300],
                        "model_used": d["model_used"],
                        "provider_used": d["provider_used"]
                    } for d in image_data.get("descriptions", [])],
                    "attributes": [{
                        "key": a["attribute_key"],
                        "value": a["attribute_value"],
                        "source": a.get("source_step", "")
                    } for a in image_data.get("attributes", [])[:50]],
                    "context_data": image_data.get("context_data", {})
                })
        
        return search_results
    
    def get_image_details(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los detalles completos de una imagen indexada por ID"""
        return self.relational_db.get_image(image_id)
    
    def get_image_details_by_name(self, image_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene los detalles completos de una imagen indexada por nombre"""
        cursor = self.relational_db.conn.cursor()
        cursor.execute("SELECT id FROM images WHERE image_name = ? ORDER BY indexed_at DESC LIMIT 1", (image_name,))
        row = cursor.fetchone()
        if row:
            return self.relational_db.get_image(row[0])
        return None
    
    def get_all_indexed(self) -> List[Dict[str, Any]]:
        """Obtiene todas las imágenes indexadas"""
        return self.relational_db.get_all_images()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del motor de búsqueda"""
        db_stats = self.relational_db.get_stats()
        db_stats["faiss_vectors"] = self.faiss_index.total
        db_stats["config"] = self.config.to_dict()
        db_stats["embeddings_loaded"] = self._embeddings_loaded
        return db_stats
    
    def delete_image(self, image_id: str):
        """Elimina una imagen del índice"""
        self.relational_db.delete_image(image_id)
        self.faiss_index.remove(image_id)
        self.faiss_index.save(self._index_path)
    
    def rebuild_index(self):
        """Reconstruye el índice FAISS desde la base de datos relacional"""
        self._ensure_embeddings_loaded()
        
        # Reinicializar FAISS
        self.faiss_index = FAISSIndex(self.config)
        self.faiss_index.initialize()
        
        # Obtener todas las imágenes
        all_images = self.relational_db.get_all_images()
        
        for img_summary in all_images:
            image_data = self.relational_db.get_image(img_summary["id"])
            if image_data and image_data.get("combined_text"):
                embedding = self.embedding_engine.encode_single(image_data["combined_text"])
                self.faiss_index.add(img_summary["id"], embedding)
        
        self.faiss_index.save(self._index_path)
