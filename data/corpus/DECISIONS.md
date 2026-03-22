# Decisiones de Inclusión y Exclusión del Corpus

Documento de decisiones tomadas durante la Fase 1 (Curación y Recolección de Contenido).

**Librería**: FastMCP 3.1.1
**Fecha**: 2026-03-22

Por ahora solo incluimos una libreria (publica) para poder continuar con el desarrollo. Pero es importante utilizar librerias privadas (que no pertenezcan al conjunto de entrenamiento de los LLMs) para reportar metricas.

---

## 1. Selección de librería

| Decisión | Opciones evaluadas | Resolución |
|----------|--------------------|------------|
| **Librería inicial** | (a) Libs del propio stack (FastMCP, loguru, supabase-py), (b) Libs populares medianas (httpx, click, typer), (c) Código propio | **(a) FastMCP** como primera librería para validar el pipeline. Es relevante al proyecto, tiene buena documentación y tamaño manejable |
| **Pre-entrenamiento** | Las libs open source están en el training data de LLMs, lo que dificulta medir el aporte real del RAG | FastMCP valida el pipeline. **Después se creará una librería privada/sintética** para el experimento real donde el RAG aporte valor medible |
| **Lenguajes** | Python solo vs Python + C++ desde el inicio | **Python primero**. El proyecto solo tiene `tree-sitter-python` instalado. C++ se agrega en una iteración futura (requiere `tree-sitter-cpp` y extender el chunker) |
| **Cantidad de librerías** | 1 para empezar vs 2-3 como sugiere el plan | **1 para empezar**. Validar el pipeline completo con una y expandir después |

---

## 2. Código fuente — Inclusiones

| Contenido | Incluido | Fuente | Justificación |
|-----------|----------|--------|---------------|
| Código fuente (`src/fastmcp/`) | Sí | `pip install fastmcp==3.1.1` (site-packages) | Versión exacta pinchada en el proyecto. 236 archivos, 60,821 líneas |
| Todos los módulos (server, client, cli, tools, resources, prompts, utilities, contrib, experimental) | Sí | — | Cobertura completa de la API para maximizar variedad de chunks |
| **Examples** (`examples/`) | Sí | GitHub repo | 103 archivos Python, 8,440 líneas. Muestran uso real de la API: apps, auth, config, filesystem, echo, memory, etc. Valiosos para queries tipo "cómo uso X" |

## 3. Código fuente — Exclusiones

| Contenido | Excluido | Volumen estimado | Justificación |
|-----------|----------|------------------|---------------|
| **Tests** (`tests/`) | Sí | ~desconocido | Los tests agregan ruido al índice. Sin embargo, los integration tests bien documentados podrían extraerse como "ejemplos de uso" en un paso separado si se necesita mejorar el recall en queries tipo "cómo se usa X" |
| **`__pycache__/`** | Sí | — | Bytecode compilado, no es código fuente |
| **`py.typed`** | Sí | 0 bytes | Marcador PEP 561, sin contenido indexable |
| **Configs** (`pyproject.toml`, `justfile`, `loq.toml`, `uv.lock`) | Sí | — | Configuración del proyecto, no de la API. No aporta al knowledge index |
| **`logo.py`** | Sí | — | Generación de logo, no es parte de la API |

## 4. Documentación — Inclusiones

| Contenido | Incluido | Fuente | Justificación |
|-----------|----------|--------|---------------|
| **Docs completos** (`docs/`) | Sí | GitHub repo (HEAD) | 435 archivos .mdx/.md, 80,134 líneas. Cobertura de getting-started, servers, clients, CLI, deployment, patterns, tutorials, changelog |
| **README.md** | Sí | Raíz del repo | Overview del proyecto, útil para queries generales |
| **CONTRIBUTING.md** | Sí | Raíz del repo | Contiene convenciones de desarrollo y arquitectura |
| **Docs legacy v2** (`docs/v2/`) | Sí | — | Útil para queries de migración v2→v3. El header path contextualizará que es documentación legacy |
| **v3-notes** (`v3-notes/`) | Sí | GitHub repo | 7 archivos markdown (776 líneas) con notas internas de diseño de v3: provider architecture, resource types, visibility, etc. Valiosos para queries sobre decisiones arquitectónicas |
| **Formato MDX** | Tratado como markdown | — | Los archivos .mdx son markdown con componentes JSX embebidos. Los componentes JSX se ignoran durante el parsing; el contenido textual y code blocks se indexan normalmente |

## 5. Documentación — Exclusiones

| Contenido | Excluido | Justificación |
|-----------|----------|---------------|
| **`docs/assets/`** | Sí | Imágenes y assets estáticos. No es texto indexable |
| **`docs/css/`** | Sí | Estilos de la web de docs. No es contenido |
| **`docs/public/`** | Sí | Assets estáticos del sitio de documentación |
| **`docs/snippets/`** | Sí | 3 archivos de fragmentos reutilizables (badges, embeds). Su contenido aparece inlined en los docs principales |
| **`AGENTS.md`, `CLAUDE.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`** | Sí | Meta-archivos del repo, no documentación de la API |
| **`skills/`** | Sí | Prompts para asistentes AI, no documentación técnica de la librería |
| **Docs no-markdown** | Sí (N/A) | Solo se indexa markdown nativo en esta fase. Si se identifican docs en otros formatos (PDF, Confluence, etc.), se evaluará conversión en fases futuras |

---

## 6. Decisiones transversales

| Decisión | Opciones evaluadas | Resolución | Razón |
|----------|--------------------|------------|-------|
| **Versiones** | Solo actual vs múltiples | **Solo versión actual** (3.1.1) | El schema de Supabase soporta versiones, expandir después es trivial. Una versión es suficiente para validar el pipeline |
| **Idioma del contenido** | Normalizar a inglés vs indexar en idioma original | **Idioma original** (inglés en este caso) | Voyage 3.5 es multilingüe. Las queries pueden ser en cualquier idioma y el vector search matchea semánticamente |
| **Tokenizer de referencia** | `voyageai.count_tokens()` vs estimación 4 chars/token vs tiktoken | **`voyageai.count_tokens()`** para estimaciones del corpus | Es el tokenizer real. Para conteos rápidos en runtime (context builder), tiktoken como proxy |
| **Fuente del código** | Clonar repo vs copiar de site-packages | **site-packages** para código (versión exacta), **repo** para docs (no se distribuyen con pip) | Garantiza que el código indexado es exactamente lo que ejecuta el proyecto |

---

## 7. Elementos a reconsiderar en fases futuras

| Elemento | Fase | Condición para incluir |
|----------|------|----------------------|
| Tests como ejemplos de uso | Fase 2 | Extraer integration tests bien documentados como chunks de tipo `example` |
| Librería privada/sintética | Post Fase 2 | Una vez validado el pipeline, crear contenido que no esté en el training data de LLMs |
| Soporte C++ | Post Fase 2 | Agregar `tree-sitter-cpp` y extender el code chunker |
