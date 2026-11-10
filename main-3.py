from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import json
import os
import uuid
import base64
import c2pa
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Detector - Motor Claude (Anthropic)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cliente = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODELO = "claude-sonnet-4-6"

SYSTEM_PROMPT = """Eres un sistema experto en detección forense de contenido generado con inteligencia artificial.
Analizas imágenes y audios para determinar si son reales o generados por IA (GAN, Diffusion Models, TTS, Voice Cloning).

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin markdown, sin backticks.

Estructura exacta requerida:
{
  "ai_probability": 85,
  "verdict": "AI_GENERATED",
  "confidence": "ALTA",
  "metrics": [
    {"label": "Nombre", "value": "Valor", "note": "Nota breve"},
    {"label": "Nombre", "value": "Valor", "note": "Nota breve"},
    {"label": "Nombre", "value": "Valor", "note": "Nota breve"},
    {"label": "Nombre", "value": "Valor", "note": "Nota breve"}
  ],
  "signals": [
    {"severity": "HIGH", "text": "Descripcion", "tag": "ETIQUETA"},
    {"severity": "HIGH", "text": "Descripcion", "tag": "ETIQUETA"},
    {"severity": "MED",  "text": "Descripcion", "tag": "ETIQUETA"},
    {"severity": "LOW",  "text": "Descripcion", "tag": "ETIQUETA"}
  ],
  "analysis": "Parrafo tecnico detallado de al menos 3 oraciones."
}

Reglas:
- verdict: solo AI_GENERATED, LIKELY_REAL o UNCERTAIN
- confidence: solo ALTA, MEDIA o BAJA
- severity: solo HIGH, MED o LOW
- Siempre exactamente 4 metricas y 4 senales"""


def parsear_json(texto: str) -> dict:
    limpio = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(limpio)


@app.post("/detectar/imagen")
async def analizar_imagen(archivo: UploadFile = File(...)):
    if not archivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contenido_bytes = await archivo.read()

    # CAPA 1: C2PA
    nombre_tmp = f"tmp_{uuid.uuid4().hex}.jpg"
    with open(nombre_tmp, "wb") as f:
        f.write(contenido_bytes)

    try:
        resultado_c2pa = c2pa.verify_from_file_json(nombre_tmp)
        if resultado_c2pa:
            manifest = str(json.loads(resultado_c2pa)).lower()
            firmas_ia = ["ai_generative", "dall", "firefly", "midjourney", "openai", "stable diffusion"]
            if any(firma in manifest for firma in firmas_ia):
                return {
                    "ai_probability": 100,
                    "verdict": "AI_GENERATED",
                    "confidence": "ALTA",
                    "metrics": [
                        {"label": "Firma C2PA",  "value": "Detectada", "note": "Metadato criptografico incrustado"},
                        {"label": "Motor",        "value": "Local",     "note": "Sin consumir cuota de API"},
                        {"label": "Precision",    "value": "100%",      "note": "Verificacion criptografica"},
                        {"label": "Latencia",     "value": "< 0.1s",   "note": "Procesamiento inmediato"},
                    ],
                    "signals": [
                        {"severity": "HIGH", "text": "El archivo contiene una firma oficial C2PA de generacion por IA.", "tag": "C2PA_FOUND"},
                        {"severity": "HIGH", "text": "Datos de origen verificados criptograficamente.", "tag": "CRYPTO_VERIFIED"},
                        {"severity": "LOW",  "text": "Se omitio la llamada a Claude para ahorrar recursos.", "tag": "BYPASS_API"},
                        {"severity": "LOW",  "text": "Analisis completado en el servidor local.", "tag": "LOCAL_EXEC"},
                    ],
                    "analysis": "Deteccion instantanea por firma C2PA. Se encontro una marca de agua criptografica incrustada por el fabricante. Esta firma es confirmacion directa e irrefutable de que la imagen es artificial. No fue necesario consumir cuota de la API de Claude.",
                }
    except Exception:
        pass
    finally:
        if os.path.exists(nombre_tmp):
            os.remove(nombre_tmp)

    # CAPA 2: Claude Vision
    imagen_b64 = base64.standard_b64encode(contenido_bytes).decode("utf-8")

    prompt = """Analiza esta imagen y determina si fue generada con IA o es real.
Evalua: artefactos de modelos de difusion, inconsistencias en manos/ojos/cabello/texto,
ruido digital vs grano fotografico, coherencia de iluminacion y sombras, uniformidad de texturas.
Metricas sugeridas: Coherencia de textura, Artefactos de difusion, Naturalidad del ruido, Consistencia de luz."""

    try:
        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": archivo.content_type,
                            "data": imagen_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return parsear_json(respuesta.content[0].text)

    except Exception as e:
        print(f"--- ERROR EN CLAUDE (IMAGEN) ---: {e}")
        raise HTTPException(status_code=500, detail=f"Error en Claude: {str(e)}")


@app.post("/detectar/audio")
async def analizar_audio(archivo: UploadFile = File(...)):
    if not archivo.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser audio.")

    contenido_bytes = await archivo.read()
    size_kb = len(contenido_bytes) / 1024

    prompt = f"""Analiza este archivo de audio/voz y determina si fue generado con IA o es humano real.

Informacion del archivo:
- Nombre: {archivo.filename}
- Tipo MIME: {archivo.content_type}
- Tamano: {size_kb:.1f} KB

Evalua: naturalidad de la prosodia, artefactos de sintesis vocal, coherencia espectral,
variabilidad emocional, patrones de respiracion naturales vs sinteticos.
Metricas sugeridas: Naturalidad vocal, Coherencia espectral, Variabilidad prosodica, Artefactos de sintesis."""

    try:
        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return parsear_json(respuesta.content[0].text)

    except Exception as e:
        print(f"--- ERROR EN CLAUDE (AUDIO) ---: {e}")
        raise HTTPException(status_code=500, detail=f"Error en Claude: {str(e)}")


@app.get("/")
def health():
    return {"status": "ok", "motor": "Claude Anthropic", "modelo": MODELO}
