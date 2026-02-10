# Vision LLM Comparator - Dockerfile
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
# Nota: libgl1 reemplaza a libgl1-mesa-glx en Debian Trixie+
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requisitos primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios para datos persistentes
RUN mkdir -p /data /data/agent_outputs /data/reports

# Establecer variable de entorno para el directorio de datos
ENV VISION_LLM_DATA_DIR=/data

# Exponer el puerto
EXPOSE 8000

# Healthcheck (start-period largo para primera descarga de modelo de embeddings)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/statistics || exit 1

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
