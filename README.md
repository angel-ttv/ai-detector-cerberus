# AI Detector MVP — Setup

## Errores corregidos en esta versión
- ✅ Modelo Gemini corregido: `gemini-2.0-flash` (el 1.5-flash no está en v1beta)
- ✅ Endpoint `/detectar/audio` implementado (faltaba)
- ✅ Función `descargarReporte()` implementada (generaba error silencioso)
- ✅ Modo URL: descarga el archivo antes de enviarlo al backend

---

## Estructura del proyecto

```
ai-detector/
├── main.py          ← Backend FastAPI (Python)
├── .env             ← Tu API Key de Gemini
├── index.html
├── css/
│   └── style.css
└── js/
    ├── config.js    ← URL del backend
    └── app.js
```

---

## Cómo correr el proyecto

### 1. Backend Python

```powershell
# Activa tu entorno virtual
.\venv\Scripts\activate

# Instala dependencias (solo la primera vez)
pip install fastapi uvicorn python-dotenv google-genai c2pa-python pillow python-multipart

# Corre el servidor
uvicorn main:app --reload
```

El backend queda en `http://localhost:8000`

### 2. Frontend

Abre `index.html` con **Live Server** en VS Code (extensión gratuita).

---

## Variables de entorno (.env)

```
GEMINI_API_KEY=tu_clave_aqui
```

---

## Modelos disponibles en la nueva SDK

| Modelo | Uso |
|---|---|
| `gemini-2.0-flash` | ✅ Recomendado (rápido, multimodal) |
| `gemini-2.5-pro` | Para análisis más profundo (más lento) |

---

## Notas importantes

- El modo **URL** funciona solo si el servidor de la imagen permite CORS. Si falla, sube el archivo directamente.
- Para audio, Gemini 2.0 Flash soporta MP3, WAV, OGG y M4A nativamente.
- El PDF se genera en el cliente (html2pdf.js), no necesita el backend.
