# Internal Knowledge MCP Server — Plan de Implementación

---

## Resumen del Proyecto

Construcción de un MCP server experimental para conocimiento técnico de librerías internas, implementando tres arquitecturas distintas (retrieval puro, retrieval + síntesis, agentic retrieval) para comparar enfoques de RAG sobre el mismo índice de datos.

**Stack**: Python + FastMCP | Supabase (PostgreSQL + pgvector) | Voyage AI (Code 3 + 3.5) | Anthropic/OpenAI API

**Desarrollo**: Secuencial estricto — cada fase se completa antes de iniciar la siguiente.

---

## Estructura del Proyecto

```
internal-knowledge-mcp/
├── pyproject.toml
├── README.md
├── Makefile
├── .env.example
│
├── src/
│   ├── indexing/                    # Pipeline de indexing
│   │   ├── __init__.py
│   │   ├── code_chunker.py          # AST-aware chunking con tree-sitter
│   │   ├── markdown_chunker.py      # Chunking semántico de markdown
│   │   ├── proposition_extractor.py # Extracción de tarjetas atómicas via LLM
│   │   ├── embeddings.py            # Cliente Voyage AI (code-3 + 3.5)
│   │   ├── ingestion.py             # Orquestador del pipeline completo
│   │   └── hash_tracker.py          # Tracking de hashes para re-indexing incremental
│   │
│   ├── retrieval/                   # Lógica de búsqueda
│   │   ├── __init__.py
│   │   ├── vector_search.py         # Búsqueda semántica en pgvector
│   │   ├── fulltext_search.py       # Full-text search en PostgreSQL
│   │   ├── hybrid_search.py         # Merge de resultados (RRF)
│   │   └── context_builder.py       # Armado de contexto con deduplicación y budget
│   │
│   ├── mcp/                         # MCP server y tools
│   │   ├── __init__.py
│   │   ├── server.py                # FastMCP server principal
│   │   ├── tools_shared.py          # Tools compartidas: resolve_library, get_file_content
│   │   ├── tools_level1.py          # search_library
│   │   ├── tools_level2.py          # query_internal_knowledge, get_retrieval_context
│   │   └── tools_level3.py          # query_internal_knowledge_agent
│   │
│   ├── synthesis/                   # Generación server-side (niveles 2 y 3)
│   │   ├── __init__.py
│   │   ├── llm_client.py            # Cliente unificado Anthropic/OpenAI
│   │   ├── prompts.py               # Templates de prompts para síntesis
│   │   ├── query_store.py           # Almacén de chunks por query_id (TTL-based)
│   │   └── agent.py                 # Loop agentic con tool use (nivel 3)
│   │
│   ├── logger.py                    # Logging estructurado JSONL (loguru)
│   │
│   └── db/                          # Capa de datos
│       ├── __init__.py
│       ├── client.py                # Cliente Supabase
│       └── queries.py               # Queries SQL reutilizables (RRF, hybrid search)
│
├── scripts/
│   ├── index_library.py             # CLI para indexar una librería
│   ├── reindex.py                   # Re-indexing incremental
│   ├── run_eval.py                  # Ejecutar evaluación con golden set
│   └── generate_golden_set.py       # Generar preguntas de evaluación semi-automáticamente
│
├── eval/
│   ├── golden_set.json              # Preguntas + respuestas esperadas
│   ├── results/                     # Resultados por nivel y run
│   └── analysis.py                  # Scripts de análisis y gráficos
│
├── data/
│   └── corpus/                      # Contenido fuente (gitignored si es sensible)
│       ├── manifest.yaml            # Inventario del corpus
│       └── {library_name}/
│           ├── code/
│           └── docs/
│
└── tests/
    ├── test_code_chunker.py
    ├── test_markdown_chunker.py
    ├── test_proposition_extractor.py
    ├── test_hybrid_search.py
    ├── test_context_builder.py
    └── test_mcp_tools.py
```

---

## Makefile

```makefile
.PHONY: install test index serve eval lint

install:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

index:
	python -m scripts.index_library --library $(LIB) --version $(VER) --corpus-path data/corpus/$(LIB)

index-dry:
	python -m scripts.index_library --library $(LIB) --version $(VER) --corpus-path data/corpus/$(LIB) --dry-run

reindex:
	python -m scripts.reindex --library $(LIB)

serve:
	python -m src.mcp.server

eval:
	python -m scripts.run_eval --golden-set eval/golden_set.json --output eval/results/

eval-level:
	python -m scripts.run_eval --golden-set eval/golden_set.json --level $(LEVEL) --output eval/results/
```

---

## Fase 0 — Infraestructura Base

### Objetivo

Tener el esqueleto del proyecto funcionando: proyecto Python configurado, Supabase aprovisionado con schemas, y un MCP server vacío que arranca y se conecta.

### Tareas

#### 0.1 — Setup del proyecto Python

- Inicializar repositorio con la estructura de directorios definida arriba.
- Configurar `pyproject.toml` con dependencias:
  - Core: `fastmcp`, `supabase-py`, `voyageai`, `anthropic`
  - Indexing: `tree-sitter`, `tree-sitter-python`, `mistune`
  - Utils: `tiktoken`, `python-dotenv`, `pyyaml`
  - Dev: `pytest`, `ruff`, `pyright`
- Configurar `.env.example` con variables: `SUPABASE_URL`, `SUPABASE_KEY`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`.
- Configurar `ruff` para linting/formatting, `pytest` para tests.
- Crear `Makefile` con comandos estándar.

#### 0.2 — Aprovisionamiento de Supabase

- Crear proyecto en Supabase (free tier).
- Habilitar extensión pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

- Crear tabla `source_files` (almacena archivos fuente completos para `get_file_content`):

```sql
CREATE TABLE source_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library TEXT NOT NULL,
    version TEXT NOT NULL,
    filepath TEXT NOT NULL,
    content TEXT NOT NULL,
    file_type TEXT NOT NULL CHECK (file_type IN ('python', 'markdown')),
    total_lines INT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(library, version, filepath)
);
```

- Crear tabla `code_chunks`:

```sql
CREATE TABLE code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library TEXT NOT NULL,
    version TEXT NOT NULL,
    file_id UUID REFERENCES source_files(id),
    filepath TEXT NOT NULL,
    module TEXT NOT NULL,
    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('function', 'class', 'method', 'module')),
    symbol_name TEXT NOT NULL,
    signature TEXT,
    docstring TEXT,
    source_code TEXT NOT NULL,
    imports TEXT[],
    dependencies TEXT[],
    start_line INT NOT NULL,
    end_line INT NOT NULL,
    embedding VECTOR(1024),
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

- Crear tabla `doc_chunks`:

```sql
CREATE TABLE doc_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library TEXT NOT NULL,
    version TEXT NOT NULL,
    file_id UUID REFERENCES source_files(id),
    source_file TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('section', 'proposition', 'changelog')),
    header_path TEXT,
    scope TEXT,
    content TEXT NOT NULL,
    original_section TEXT,
    start_line INT,
    end_line INT,
    embedding VECTOR(1024),
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

- Crear índices HNSW para búsqueda vectorial:

```sql
CREATE INDEX code_chunks_embedding_idx ON code_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX doc_chunks_embedding_idx ON doc_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

- Crear columnas tsvector generadas e índices GIN para full-text search. Diccionario `simple` para código (evita stemming de símbolos), `english` para docs:

```sql
-- FTS para código: diccionario simple, replace de underscores para que
-- snake_case matchee (get_user_by_id → "get user by id")
ALTER TABLE code_chunks ADD COLUMN fts TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('simple',
            COALESCE(replace(symbol_name, '_', ' '), '') || ' ' ||
            COALESCE(signature, '') || ' ' ||
            COALESCE(docstring, '')
        )
    ) STORED;

CREATE INDEX code_chunks_fts_idx ON code_chunks USING gin(fts);

-- FTS para docs: diccionario english para stemming de lenguaje natural
ALTER TABLE doc_chunks ADD COLUMN fts TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('english',
            COALESCE(header_path, '') || ' ' ||
            COALESCE(content, '')
        )
    ) STORED;

CREATE INDEX doc_chunks_fts_idx ON doc_chunks USING gin(fts);
```

- Crear índices auxiliares para filtros frecuentes:

```sql
CREATE INDEX code_chunks_library_idx ON code_chunks(library, version);
CREATE INDEX doc_chunks_library_idx ON doc_chunks(library, version);
CREATE INDEX doc_chunks_content_type_idx ON doc_chunks(content_type);
CREATE INDEX code_chunks_file_id_idx ON code_chunks(file_id);
CREATE INDEX doc_chunks_file_id_idx ON doc_chunks(file_id);
```

- Implementar `src/db/client.py` (conexión Supabase) y `src/db/queries.py` (queries SQL reutilizables para hybrid search y RRF).
- Validar conectividad desde Python con supabase-py.

#### 0.3 — Scaffolding de FastMCP

- Implementar MCP server mínimo con FastMCP que exponga un tool de health check.
- Verificar que el servidor arranca, se registra correctamente, y un MCP client puede descubrirlo.
- Configurar logging estructurado JSONL con loguru (`src/logger.py`): timestamp, tipo de operación, tokens, latencia, costo, resultado.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **HNSW vs IVFFlat para pgvector** | IVFFlat requiere entrenamiento previo, HNSW es más robusto para datasets pequeños-medianos | **HNSW**. Para el volumen experimental (~50MB de texto) HNSW es la opción correcta. Re-evaluar si el corpus crece a millones de vectores |
| **Parámetros HNSW** | `m` (conexiones por nodo) y `ef_construction` (calidad del índice) | Empezar con defaults de pgvector (`m=16`, `ef_construction=64`). Tunear después si recall@k es bajo |
| **supabase-py vs SQL directo** | supabase-py es más ergonómico pero limitado para queries complejas (RRF, hybrid search) | **Híbrido**: supabase-py para CRUD, SQL directo (via `rpc`) para búsqueda híbrida y RRF |
| **Operator de distancia en pgvector** | `<=>` (cosine), `<->` (L2), `<#>` (inner product) | **Cosine (`<=>`)** ya que los embeddings de Voyage están normalizados. Es el estándar para similarity search con estos modelos |
| **FTS: diccionario por tabla** | Mismo diccionario para todo vs diferenciado | **Diferenciado**: `simple` para `code_chunks` (evita stemming de símbolos Python), `english` para `doc_chunks` (beneficia queries en lenguaje natural). El replace de underscores en `symbol_name` permite que snake_case matchee en FTS |
| **tsvector: columna generada vs manual** | Columna `GENERATED ALWAYS AS` vs actualizar manualmente | **Generada**. Más limpio, se mantiene automáticamente sincronizada con los datos. No hay overhead de mantenimiento |
| **source_files como tabla separada con FK** | Tabla auxiliar con FK vs almacenar contenido inline en chunks | **Tabla separada con FK**. Evita duplicación de contenido, centraliza el acceso a archivos fuente, y las FKs `file_id` en `code_chunks` y `doc_chunks` permiten navegar eficientemente desde un chunk al archivo completo |

### Entregables

- [x] Repositorio inicializado con estructura de directorios, dependencias, y Makefile
- [x] Supabase configurado con las tres tablas (`source_files`, `code_chunks`, `doc_chunks`), índices HNSW, GIN, y auxiliares
- [x] MCP server que arranca, conecta a Supabase, y responde health check
- [x] Logger JSONL configurado (loguru)
- [x] README con instrucciones de setup local
- [x] Pre-commit hooks configurados (ruff, detect-secrets, validaciones generales)
- [x] Conectividad validada: Supabase, Voyage AI, OpenAI (GPT-5.4 nano)

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Supabase free tier tiene límites de storage y conexiones | Medio — podría ser insuficiente si el corpus crece | Monitorear uso. El free tier da 500MB de DB, que es suficiente para el experimento. Si se excede, migrar a plan Pro ($25/mes) |
| FastMCP puede tener breaking changes (proyecto relativamente nuevo) | Bajo — solo afecta la capa de transporte | Pinear versión de FastMCP en dependencias. La lógica de negocio es independiente del framework MCP |
| Full-text search tokeniza mal nombres Python | Medio — `get_user_by_id` no matchea queries | Diccionario `simple` + `replace(symbol_name, '_', ' ')` en la columna tsvector generada. Validar con queries de prueba |

### Dependencias

- Cuenta en Supabase (gratuita) ✓
- API key de Voyage AI (free tier: 200M tokens por modelo) ✓
- API key de OpenAI (prepago, modelo principal: GPT-5.4 nano para desarrollo) ✓
- API key de Anthropic (opcional, para comparar calidad en fases posteriores)

---

## Fase 1 — Curación y Recolección de Contenido

### Objetivo

Reunir, organizar y validar el corpus que alimentará el indexing pipeline. Producir un manifiesto claro de qué se indexa.

### Tareas

#### 1.1 — Identificación de librerías target

- Seleccionar las librerías internas que van a indexarse (mínimo 2-3 para tener variedad en la evaluación).
- Documentar por cada librería: nombre, versión actual, lenguaje principal, dependencias externas relevantes, volumen estimado de código y docs.

#### 1.2 — Recolección de código fuente

- Clonar o copiar el código fuente de cada librería al directorio `data/corpus/{library_name}/code/`.
- Validar que el código es parseable por tree-sitter (instalar gramáticas necesarias).
- Filtrar archivos irrelevantes: tests (a menos que sean ejemplos de uso), configs, archivos generados, assets binarios.

#### 1.3 — Recolección de documentación

- Reunir documentación en markdown: READMEs, docs/, guides/, changelogs, blogs internos.
- Organizar en `data/corpus/{library_name}/docs/`.
- Validar que el markdown es parseable (no corrupto, encoding correcto).
- Identificar contenido que NO es markdown (PDFs, Confluence, etc.) y decidir si convertir o excluir.

#### 1.4 — Creación del manifiesto

- Generar `data/corpus/manifest.yaml` que liste:
  - Cada librería con su versión
  - Archivos de código a indexar (paths, lenguaje, líneas de código)
  - Archivos de documentación a indexar (paths, formato, tamaño)
  - Estimación total de tokens (código + docs)
  - Exclusiones explícitas y razón

#### 1.5 — Validación del corpus

- Ejecutar conteo de tokens con `voyageai.Client.count_tokens()` para tener estimación real (no la aproximación de 4 chars/token).
- Verificar que el volumen está dentro del free tier de embeddings (200M tokens por modelo).
- Identificar archivos problemáticos: funciones gigantes (>2000 tokens), archivos de documentación sin estructura de headers, código con syntax inválida.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Incluir tests como contenido** | Tests muestran uso real de la API vs agregan ruido | **Excluir tests del indexing principal**. Si hay tests que son claramente ejemplos de uso (integration tests bien documentados), extraerlos como "ejemplos" en un paso separado |
| **Qué hacer con docs no-markdown** | Convertir todo a MD vs indexar solo MD nativo | **Solo markdown nativo en Fase 1**. Agregar conversión de otros formatos como mejora futura si es necesario |
| **Granularidad de versiones** | Indexar solo la versión actual vs múltiples versiones | **Solo versión actual** para la primera iteración. El schema soporta versiones, así que expandir después es trivial |
| **Contenido en español vs inglés** | Docs internas pueden estar en español | **Indexar en el idioma original**. Voyage 3.5 es multilingüe. Las queries pueden ser en cualquier idioma y el vector search matchea semánticamente |
| **Tokenizer de referencia** | `voyageai.Client.count_tokens()` vs estimación de 4 chars/token vs tiktoken | **`voyageai.Client.count_tokens()`** como fuente de verdad para estimaciones del corpus. Es el tokenizer real que va a procesar el texto. Para conteos rápidos en runtime (ej: context builder), usar tiktoken como proxy rápido |

### Entregables

- [ ] Corpus organizado en `data/corpus/` con estructura consistente
- [ ] Manifiesto `manifest.yaml` con inventario completo
- [ ] Estimación real de tokens por índice (usando tokenizer de Voyage AI)
- [ ] Documento de decisiones sobre inclusiones/exclusiones

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Documentación pobre o desactualizada en las librerías internas | Alto — el índice reflejará la calidad del input | Priorizar librerías con buena documentación para el experimento. Si la documentación es pobre, es un datapoint válido: mide cuánto aporta el indexing de código cuando los docs son insuficientes |
| Corpus demasiado pequeño para evaluar retrieval con rigor | Medio — poca variedad en las preguntas del golden set | Apuntar a un mínimo de 200-300 chunks totales entre las tres capas de indexing. Si el corpus natural es muy chico, considerar agregar una librería open source tratada como "interna" para aumentar volumen |
| Código con syntax inválida que tree-sitter no puede parsear | Bajo — tree-sitter es muy tolerante a errores | Tree-sitter produce ASTs parciales para código con errores. Loggear warnings y revisar manualmente los archivos problemáticos |

### Dependencias

- Acceso a los repos de las librerías internas
- Fase 0 completada (para validar token estimates contra el esquema de Supabase)

---

## Fase 2 — Indexing Pipeline

### Objetivo

Construir el pipeline completo que transforma código y documentación en chunks indexados con embeddings en Supabase.

### Tareas

#### 2.1 — Chunking de código con tree-sitter (`src/indexing/code_chunker.py`)

- Instalar tree-sitter y las gramáticas necesarias (Python como mínimo, agregar otros lenguajes según el corpus).
- Implementar extracción AST-aware en tres niveles de granularidad:
  - **Módulo**: archivo completo como chunk
  - **Clase**: clase con todos sus métodos
  - **Función/método**: unidad más granular
- Extraer metadata por chunk: signature con type hints, docstring, imports relevantes (filtrar solo los que usa la función), filepath, módulo, dependencias internas.
- Manejar edge cases de tree-sitter:
  - Funciones anidadas (inner functions)
  - Decorators (`@property`, `@staticmethod`, `@classmethod`, custom decorators)
  - Clases con herencia
  - Funciones con `*args`, `**kwargs`
  - Funciones sin type hints (graceful degradation)
- Implementar lógica para funciones largas (>800 tokens): indexar enteras si ≤ 16K tokens (límite de Voyage Code 3). Split solo si exceden ese límite.
- Generar content_hash (SHA-256 del source code) para re-indexing incremental.
- Tests unitarios sobre código real del corpus para validar que el chunking produce unidades semánticas completas.

#### 2.2 — Chunking de documentación (`src/indexing/markdown_chunker.py`)

- Implementar parser de markdown con `mistune` para obtener el AST.
- Implementar lógica recursiva de chunking:
  - Empezar desde niveles más profundos
  - Si nodo < 100 tokens → merge con padre
  - Si nodo > 800 tokens → chunk individual
  - Usar `voyageai.Client.count_tokens()` para conteo preciso
- Generar header path (breadcrumb) por chunk: `Librería > Sección > Subsección > Título`.
- Implementar overlap semántico: últimas 2-3 oraciones de la sección anterior como prefijo para embedding (NO incluidas en el texto devuelto al LLM).
- Manejar elementos especiales:
  - Code blocks en docs → indexar TAMBIÉN en el índice de código con Voyage Code 3 (dual indexing)
  - Tablas → convertir a texto estructurado antes de embedding
  - Changelogs → cada entrada de versión como chunk natural
- Content hash por sección.

#### 2.3 — Extracción de proposiciones atómicas (`src/indexing/proposition_extractor.py`)

- Implementar pipeline de extracción con LLM (Claude Haiku default, configurable):
  - Input: sección de documentación completa + contexto del documento
  - Output: lista de proposiciones autocontenidas con scope
- Diseñar el prompt de extracción: la proposición debe ser independiente, sin referencias a "lo anterior" o "como vimos", incluir nombres concretos de funciones/librerías, y especificar el scope (api_usage, config, migration, troubleshooting).
- **Prompt versioning**: hashear el prompt como parte del `content_hash` de las proposiciones. Si cambia el prompt, se dispara re-generación automática de todas las tarjetas. Esto permite iterar sobre el prompt sin correr re-indexing manual.
- Implementar batching con concurrencia controlada para eficiencia.
- Logging de costos via logger.

#### 2.4 — Generación de embeddings (`src/indexing/embeddings.py`)

- Implementar wrapper sobre `voyageai.Client` que:
  - Genera embeddings con `voyage-code-3` para código, usando `input_type="document"` para indexing.
  - Genera embeddings con `voyage-3.5` para documentación y proposiciones, usando `input_type="document"` para indexing.
  - Para queries de búsqueda (en Fase 3), usa `input_type="query"` — **Voyage AI diferencia entre document y query embeddings, esto es crítico para la calidad del retrieval**.
  - Maneja batching automático (max 128 items por request).
  - Retry con exponential backoff para rate limits.
  - Usa `voyageai.Client.count_tokens()` como tokenizer de referencia.
  - Loguea tokens consumidos via logger.
- Manejar textos que exceden 16K tokens (Voyage Code 3): truncar con warning.

#### 2.5 — Carga en Supabase (`src/indexing/ingestion.py`)

- Implementar flujo completo de ingestion:
  1. Insertar archivos fuente en `source_files` (código y markdown).
  2. Ejecutar chunker correspondiente (code o markdown).
  3. Calcular `content_hash`, verificar contra DB.
  4. Generar embeddings en batch.
  5. Insertar en `code_chunks` o `doc_chunks` con referencia a `file_id`.
- Lógica de re-indexing incremental (`src/indexing/hash_tracker.py`):
  - Comparar content_hash existente vs nuevo.
  - Solo re-procesar (chunk + embed + insert) secciones con hash cambiado.
  - Eliminar chunks cuyo archivo fuente ya no existe.
  - Para proposiciones: el hash incluye el hash del prompt de extracción, así un cambio de prompt dispara re-generación.

#### 2.6 — CLI de indexing

- Implementar `scripts/index_library.py`:
  - Uso: `make index LIB=<name> VER=<version>`
  - Flags: `--dry-run`, `--force` (re-indexa todo ignorando hashes), `--verbose`
- Implementar `scripts/reindex.py` para re-indexing incremental.
- Logging de métricas: chunks generados por tipo, tokens procesados, costo estimado de embeddings, tiempo total.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Threshold de merge en docs** | 100 tokens parece bajo para secciones cortas pero legítimas | **100 tokens como default configurable**. Agregar flag `--min-chunk-tokens` en el CLI. Evaluar después del primer indexing si produce demasiados merges |
| **Overlap: oraciones vs tokens** | Overlap por oraciones (2-3) vs por tokens (últimos N tokens) | **Oraciones (2-3)**. Más semánticamente coherente. Implementar sentence splitting básico (regex sobre `.`, `!`, `?` + heurísticas para abreviaturas) |
| **Prompt de proposiciones: un prompt para todo vs prompts por scope** | Un prompt genérico vs prompts especializados por tipo de contenido | **Un prompt genérico** para la primera iteración. Si el análisis de calidad muestra que ciertos tipos de contenido generan proposiciones pobres, especializar prompts |
| **Dual indexing de code blocks en docs** | Indexar code blocks de docs en ambos índices vs solo en docs | **Dual indexing**. Es barato (embeddings extras) y mejora recall para queries de código que aparecen en la documentación |
| **Modelo para proposiciones** | Claude Haiku vs GPT-4o-mini | **Claude Haiku** como default (consistencia con el stack Anthropic). Hacer el modelo configurable para comparar calidad si se desea |
| **Funciones >800 tokens** | Indexar enteras vs splitear | **Indexar enteras** si ≤ 16K tokens (límite de Voyage Code 3). Split solo si exceden ese límite, lo cual sería muy raro en una codebase típica |
| **Voyage AI `input_type`** | Ignorar vs usar `document`/`query` | **Siempre especificar**: `input_type="document"` para indexing, `input_type="query"` para búsqueda. Voyage AI optimiza los embeddings según el uso. Omitirlo degrada la calidad del retrieval |
| **Prompt versioning para proposiciones** | Re-generar manualmente vs automático por hash | **Automático**: el hash del prompt se incorpora al `content_hash` de las proposiciones. Cambio de prompt → re-generación automática |

### Entregables

- [ ] Pipeline de chunking de código funcional con tree-sitter (edge cases cubiertos)
- [ ] Pipeline de chunking de docs funcional con mistune
- [ ] Pipeline de extracción de proposiciones con LLM y prompt versioning
- [ ] Cliente de embeddings Voyage AI con batching, retry, y `input_type` correcto
- [ ] Carga completa del corpus en Supabase con FKs a `source_files`
- [ ] CLI de indexing con dry-run y re-indexing incremental
- [ ] Métricas del indexing: total de chunks por tipo, tokens procesados, distribución de tamaños de chunks

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Calidad de proposiciones depende fuertemente del prompt | Alto — proposiciones malas degradan el retrieval | Iterar sobre el prompt con 10-20 secciones antes de correr el indexing completo. Evaluar manualmente: ¿las proposiciones son realmente autocontenidas? ¿capturan la idea principal? Prompt versioning permite iterar sin fricciones |
| Tree-sitter puede no soportar todos los lenguajes del corpus | Medio — perder chunks de código de lenguajes no soportados | Verificar gramáticas disponibles antes de empezar. Para lenguajes sin soporte, usar fallback de chunking basado en `ast` de stdlib (Python) o regex |
| Tree-sitter no maneja edge cases Python (decorators, nested functions) | Medio — chunks incompletos o incorrectos | Tests exhaustivos con el corpus real. Fallback a `ast` de stdlib para los casos que tree-sitter no cubra |
| Rate limits de Voyage AI durante indexing | Bajo — pipeline falla a mitad | Batching con backoff exponencial. Re-indexing incremental con hashes permite retomar sin reprocesar todo |
| Costo de LLM para proposiciones puede ser mayor al estimado | Bajo — Haiku es económico y el volumen es acotado | Estimar costo antes de correr: (tokens de input por sección × número de secciones × precio por token). Para ~10M tokens de docs, Haiku cuesta ~$2.50 |

### Dependencias

- Fase 0 (Supabase configurado con las tres tablas)
- Fase 1 (corpus curado y manifiesto listo)

---

## Fase 3 — Search & Retrieval

### Objetivo

Implementar la capa de búsqueda híbrida compartida por los tres niveles: full-text search, vector search, fusión con RRF, deduplicación, y context builder con budget de tokens.

### Tareas

#### 3.1 — Full-Text Search (`src/retrieval/fulltext_search.py`)

- Implementar búsqueda FTS sobre `code_chunks` usando la columna `fts` generada, con `ts_rank_cd` para scoring.
- Implementar búsqueda FTS sobre `doc_chunks` usando la columna `fts` generada.
- Manejar queries técnicas: normalizar snake_case/camelCase en la query antes de buscar (reemplazar `_` por espacios, splitear camelCase).

#### 3.2 — Vector Search (`src/retrieval/vector_search.py`)

- Implementar query embedding con Voyage AI usando `input_type="query"` (crítico — Voyage diferencia document vs query embeddings).
- Buscar siempre con ambos modelos (`voyage-code-3` en `code_chunks`, `voyage-3.5` en `doc_chunks`) y dejar que RRF fusione. Más robusto que intentar clasificar la intención de la query.
- Implementar búsqueda por cosine similarity (`<=>`) en ambas tablas.
- Parámetros configurables: top-k por tabla, threshold mínimo de similarity.

#### 3.3 — Hybrid Search con RRF (`src/retrieval/hybrid_search.py`)

- Implementar Reciprocal Rank Fusion para combinar resultados de FTS y vector search.
- Formula RRF: `score(d) = Σ 1/(k + rank(d))` donde k es constante (default 60).
- Combinar rankings de los tres índices (código, docs, proposiciones) en un ranking unificado.
- Top-k diferenciado antes de RRF: top-20 para código, top-20 para docs, top-15 para proposiciones (configurable).
- Implementar como una sola query SQL via `rpc` en Supabase para minimizar round-trips de red.

#### 3.4 — Deduplicación

- Implementar reglas de deduplicación:
  - Si una proposición y su sección fuente están ambas en top-k → mantener la sección (más completa), descartar la proposición.
  - Si un chunk de función y el chunk de clase que la contiene están ambos en top-k → mantener la función (más granular), EXCEPTO si múltiples métodos de la misma clase aparecen → incluir la clase como overview.
  - Code blocks que aparecen en ambos índices (código y docs) → mantener el que rankea más alto.
- Loggear todos los chunks descartados por deduplicación para análisis post-hoc.

#### 3.5 — Context Builder (`src/retrieval/context_builder.py`)

Dos modos de operación:

- **Modo Nivel 1** (budget controlado por `max_tokens` del cliente):
  1. Deduplicar según las reglas de 3.4.
  2. Incluir todos los resultados en forma compacta (tipo, identificador, signature, descripción de una línea, file_id, start_line, end_line).
  3. Si queda budget, expandir resultados de mayor score con contenido completo (source code, docstring, sección de docs).
  4. **Proposiciones**: devolver `original_section` (el texto fuente completo), no la tarjeta sintética. La proposición sirvió para el retrieval pero el contexto al LLM debe ser la sección original más rica.
  5. Parar cuando el budget se alcanza.

- **Modo Nivel 2/3** (budget interno ~8K tokens):
  - Más agresivo: expande todos los resultados de alto score con contenido completo.
  - Misma deduplicación.
  - Mismo principio de devolver `original_section` para proposiciones.

- Implementar conteo de tokens con tiktoken como proxy rápido para el budget en runtime (más preciso que 4 chars/token, más rápido que llamar a Voyage AI por cada conteo).

#### 3.6 — Testing aislado del módulo de búsqueda

- Crear un set de 15-20 queries de prueba que cubran distintos tipos:
  - Exactas: "función X", "import Y"
  - Conceptuales: "cómo manejar errores", "best practice para retry"
  - Cross-module: "diferencia entre X e Y"
  - Migration: "cambios entre v1 y v2"
- Ejecutar cada query y validar manualmente que los resultados son relevantes.
- Medir: latencia de búsqueda, distribución de resultados por índice, efecto de RRF vs single-index search.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Modelo de embedding para queries** | Siempre ambos modelos vs heurística de detección | **Siempre ambos modelos, buscar en todos los índices, dejar que RRF fusione**. Es más robusto que intentar clasificar la intención de la query. El costo extra de un embedding adicional por query es despreciable |
| **`input_type` para queries** | `"query"` vs omitir | **Siempre `input_type="query"`**. Voyage AI optimiza los embeddings para el caso de uso. Omitirlo degrada recall |
| **K de RRF** | 60 (estándar) vs tunear por tipo de índice | **60 como default**. Agregar weights opcionales por índice si el análisis muestra sesgo: ej. `w_code=1.0, w_docs=1.0, w_propositions=0.8` |
| **Top-k por tabla antes de RRF** | Mismo k para todas vs diferenciado | **Diferenciado**: top-20 para código, top-20 para docs, top-15 para proposiciones (hay más overlap con docs). Configurable |
| **Tokenizer para budget en runtime** | tiktoken vs estimación de 4 chars/token vs Voyage AI | **tiktoken** para conteos en el context builder (rápido y suficientemente preciso). `voyageai.Client.count_tokens()` solo para estimaciones del corpus en indexing |
| **Proposiciones en el contexto al LLM** | Devolver la tarjeta sintética vs el `original_section` | **`original_section`**. La proposición sirvió para mejorar el recall en retrieval, pero el contexto al LLM debe ser la sección original más rica. La tarjeta puede ser demasiado comprimida |
| **Hybrid search: múltiples queries vs una sola SQL** | Round-trips separados vs `rpc` unificado | **Una sola query SQL via `rpc`** que ejecuta FTS + vector search + RRF en Supabase. Minimiza latencia de red |

### Entregables

- [ ] Módulo de full-text search funcional sobre ambas tablas
- [ ] Módulo de vector search con `input_type="query"` y dual-model embedding
- [ ] Implementación de RRF para fusión multi-índice en una sola query SQL
- [ ] Lógica de deduplicación implementada y testeada (con logging de descartados)
- [ ] Context builder con dos modos (Nivel 1 y Nivel 2/3) y lógica de llenado progresivo
- [ ] Set de queries de prueba con resultados validados manualmente
- [ ] Benchmarks de latencia de búsqueda (target: <500ms end-to-end)

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| RRF puede favorecer un índice sobre otros si el scoring es desbalanceado | Medio — resultados sesgados hacia código o docs | Analizar distribución de resultados por índice en las queries de prueba. Ajustar weights de RRF si hay sesgo claro |
| Full-text search en Supabase puede tener latencia alta con vectores pesados en la misma tabla | Bajo — el volumen es pequeño | Si FTS es lento, separar en una tabla dedicada de FTS sin la columna vector. En este volumen no debería ser necesario |
| Deduplicación puede ser demasiado agresiva y eliminar contexto útil | Medio — chunks descartados que eran informativos | Loggear todos los chunks descartados. Hacer las reglas configurables |
| Scores de similarity no comparables entre modelos de embedding | Medio — merge sesgado | RRF opera sobre rankings, no scores — resuelve la incomparabilidad. Documentar explícitamente |
| Conteo de tokens inconsistente entre tiktoken y Voyage AI | Bajo — budgets no exactos | tiktoken es suficiente para budgets aproximados. La diferencia es marginal para el context builder |

### Dependencias

- Fase 2 (datos indexados en Supabase para poder buscar)

---

## Fase 4 — Nivel 1: Retrieval Puro (Context7-style)

### Objetivo

Exponer las herramientas MCP del Nivel 1 (`resolve_library`, `search_library`, `get_file_content`) via FastMCP, integrando el módulo de búsqueda de la Fase 3.

### Tareas

#### 4.1 — Implementación de `resolve_library` (`src/mcp/tools_shared.py`)

- Input: `library_name: str`, `query: str`
- Buscar en Supabase las librerías que matchean el nombre (fuzzy match para manejar typos).
- Devolver lista de candidatos con `library_id`, `name`, `version`.
- Si hay exactamente una coincidencia, devolver directamente. Si hay ambigüedad, devolver opciones.

#### 4.2 — Implementación de `search_library` (`src/mcp/tools_level1.py`)

- Input: `library_id: str`, `query: str`, `max_tokens: int` (opcional, default 5000)
- Integrar con el módulo de búsqueda de la Fase 3 (hybrid search → RRF → deduplicación → context builder en modo Nivel 1).
- Cada resultado del ranking incluye: tipo, symbol name / header path, signature / primera oración, source code / sección completa (según budget), metadata (file_id, start_line, end_line, filepath).
- El context builder aplica la prioridad de llenado según `max_tokens`.

#### 4.3 — Implementación de `get_file_content` (`src/mcp/tools_shared.py`)

- Input: `file_id: str`, `start_line: int` (opcional), `end_line: int` (opcional)
- Recuperar de la tabla `source_files` el contenido del archivo completo o del rango de líneas solicitado.
- Devolver contenido con números de línea para referencia.
- Manejar errores: file_id inválido, rango fuera de bounds.

#### 4.4 — Registro de herramientas en FastMCP (`src/mcp/server.py`)

- Registrar las tres herramientas como MCP tools con descripciones detalladas que incluyan "Use this when..." y "Don't use this for..." con ejemplos.
- El LLM del cliente necesita entender bien cuándo llamar a cada herramienta — las descripciones son cruciales para la calidad del Nivel 1.
- Implementar manejo de errores: query sin resultados, library no encontrada, file_id inválido.

#### 4.5 — Logging y métricas

- Loggear por cada invocación via `src/logger.py`: herramienta llamada, query, tokens consumidos en la respuesta, latencia, número de resultados.

#### 4.6 — Testing end-to-end con Claude Code

- Configurar el MCP server como provider en Claude Code.
- Ejecutar 10-15 queries reales y validar que el flujo completo funciona: resolve → search → (opcionalmente) get_file_content → Claude genera respuesta.
- Documentar cualquier problema de integración.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Descripciones de las herramientas MCP** | Descripciones minimalistas vs detalladas con ejemplos de uso | **Detalladas con ejemplos**. El LLM del cliente necesita entender bien cuándo llamar a cada herramienta. Incluir "Use this when..." y "Don't use this for..." |
| **Default de `max_tokens` en `search_library`** | 3000 vs 5000 vs 10000 | **5000**. Permite un balance entre overview compacto y detalle suficiente para responder sin fetch adicional |

### Entregables

- [ ] Tres herramientas MCP funcionales y registradas en FastMCP
- [ ] Servidor MCP Nivel 1 testeable desde Claude Code
- [ ] Logs estructurados de cada invocación con métricas
- [ ] Documentación de las herramientas (interfaz, parámetros, ejemplos de uso)

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| El LLM del cliente puede no usar las herramientas de forma óptima (ej: max_tokens muy bajo, no usar get_file_content cuando debería) | Medio — afecta la calidad de respuesta del Nivel 1 | Optimizar las descripciones de las herramientas con ejemplos. Analizar logs para identificar patrones de mal uso y ajustar descripciones |
| Latencia de Supabase (red) puede ser perceptible si hay múltiples round-trips | Bajo — el volumen de datos es pequeño | Hybrid search en una sola query SQL (FTS + vector en un solo `rpc` call) para minimizar round-trips |

### Dependencias

- Fase 3 (módulo de búsqueda funcional)

---

## Fase 5 — Nivel 2: Retrieval + Synthesis

### Objetivo

Agregar generación server-side para sintetizar respuestas antes de entregarlas al cliente, manteniendo la posibilidad de verificar fuentes.

### Tareas

#### 5.1 — Cliente LLM unificado (`src/synthesis/llm_client.py`)

- Implementar wrapper Anthropic/OpenAI con modelo configurable (default: mismo modelo que evaluará el Nivel 1 para comparación controlada).
- Temperature baja (0.1-0.3) para respuestas factuales.
- Logging de tokens consumidos y latencia.

#### 5.2 — Prompt engineering (`src/synthesis/prompts.py`)

- Diseñar system prompt que guíe al LLM de síntesis:
  - Responder basándose exclusivamente en los chunks proporcionados.
  - Citar funciones y secciones de documentación por nombre.
  - No inventar información que no esté en los chunks.
  - Si la información es insuficiente, decirlo explícitamente.
  - Incluir code snippets relevantes en la respuesta cuando aplique.
  - Si hay contradicciones entre docs y código, señalar la discrepancia.
- Iterar el prompt sobre un subset del golden set antes de correr evaluación completa.

#### 5.3 — Query Store (`src/synthesis/query_store.py`)

- Almacén en memoria (dict) con TTL configurable (default 30 min).
- Store/retrieve chunks por `query_id` (UUID).
- Lazy cleanup de entradas expiradas.

#### 5.4 — Implementación de `query_internal_knowledge` (`src/mcp/tools_level2.py`)

- Input: `question: str`, `library_id: str`, `context: str` (opcional)
- Flujo interno:
  1. Hybrid search (reutilizando Fase 3) con context builder en modo Nivel 2 (~8K tokens)
  2. Prompt assembly con chunks recuperados
  3. Llamada API a Anthropic/OpenAI
  4. Almacenar chunks en query store con `query_id`
  5. Retornar respuesta sintetizada + query_id
- Edge cases:
  - Query sin resultados relevantes → el LLM responde "no encontré información sobre..."
  - Contexto del usuario → incluirlo en el prompt como información adicional

#### 5.5 — Implementación de `get_retrieval_context` (`src/mcp/tools_level2.py`)

- Input: `query_id: str`
- Devolver los chunks almacenados en query store, en el mismo formato que `search_library` del Nivel 1.
- Manejar query_id expirado o inválido con error descriptivo.

#### 5.6 — Integración con FastMCP

- Registrar `query_internal_knowledge` y `get_retrieval_context` como herramientas MCP adicionales.
- Mantener las herramientas del Nivel 1 disponibles en el mismo servidor (el cliente puede elegir qué nivel usar).
- `resolve_library` y `get_file_content` se comparten entre niveles.

#### 5.7 — Testing end-to-end

- Ejecutar las mismas 10-15 queries del Nivel 1 con el Nivel 2.
- Comparar informalmente: ¿la síntesis mejora la experiencia?
- Verificar que `get_retrieval_context` funciona y devuelve chunks coherentes.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Budget interno del context builder** | 4000 vs 8000 vs 12000 tokens | **8000 como default configurable**. Más generoso que el Nivel 1 porque no le cuesta contexto al cliente. Suficiente para incluir 5-10 chunks detallados |
| **Modelo de síntesis** | Claude Sonnet vs Claude Haiku vs GPT-4o-mini | **Configurable, default Claude Sonnet** (o el modelo que se use para evaluar el Nivel 1). Para la evaluación comparativa, usar el mismo modelo es clave. Haiku es una alternativa económica para testing |
| **Cache: in-memory vs Redis** | Dict + TTL vs Redis con TTL nativo | **In-memory (dict + TTL)** para el experimento. El servidor es single-instance y el volumen de queries concurrentes es bajo. Si se productiviza, migrar a Redis |
| **Validación de "no inventar"** | Confiar en el prompt vs post-processing | **Confiar en el prompt** para la primera iteración. Agregar como métrica de evaluación: % de respuestas que inventan info no presente en los chunks (requiere verificación humana) |

### Entregables

- [ ] Cliente LLM unificado con modelo configurable
- [ ] Prompt de síntesis diseñado, documentado y testeado
- [ ] Query store con TTL
- [ ] `query_internal_knowledge` funcional con pipeline completo
- [ ] `get_retrieval_context` funcional
- [ ] Herramientas del Nivel 2 registradas en FastMCP junto a las del Nivel 1
- [ ] Logs de métricas: tokens consumidos por síntesis, latencia del LLM call, costo por query

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| El LLM de síntesis puede inventar información no presente en los chunks (hallucination) | Alto — respuestas incorrectas que el cliente confía | Prompt engineering agresivo + métrica de hallucination en la evaluación. Incluir en el prompt: "Si los chunks no contienen la información necesaria, responde que no la encontraste" |
| La síntesis puede perder detalles importantes que los chunks raw preservaban | Medio — trade-off calidad vs concisión | Medir específicamente: preguntas donde Nivel 1 acierta y Nivel 2 falla. Esto indicaría que la síntesis está perdiendo información |
| Latencia de la llamada API de síntesis puede ser alta | Bajo-Medio — afecta UX pero no calidad | Medir latencia. Si es inaceptable, considerar streaming. Para el experimento, latencia de 2-5s es aceptable |
| Comparación sesgada si se usan modelos distintos entre niveles | Alto — resultados no conclusivos | Evaluar Nivel 1 con el mismo modelo que sintetiza en Niveles 2/3 (hacer una llamada LLM con los chunks recuperados) |

### Dependencias

- Fase 4 (Nivel 1 funcional — se reutilizan `resolve_library` y `get_file_content`)
- API key de Anthropic/OpenAI con créditos suficientes para las llamadas de síntesis

---

## Fase 6 — Nivel 3: Agentic Retrieval

### Objetivo

Implementar el loop agentic donde un LLM decide dinámicamente qué buscar, cuántas veces, y en qué fuentes, antes de sintetizar la respuesta.

### Tareas

#### 6.1 — Definición de herramientas internas del agente

- Implementar las herramientas que el LLM del agente puede usar via tool use (function calling). Estas NO son MCP tools — son funciones internas del servidor:
  - `search_library` — reutiliza la lógica de búsqueda de la Fase 3
  - `get_file_content` — acceso a archivos fuente via `source_files`
  - `context7_docs` — consulta documentación de librerías externas via Context7 API REST
  - `aws_docs` — consulta documentación de servicios AWS (via MCP de AWS docs si disponible, o web fetch + parsing)
- Definir los schemas de tool use para cada herramienta (JSON schema compatible con Anthropic/OpenAI function calling).

#### 6.2 — Implementación del loop agentic (`src/synthesis/agent.py`)

- Implementar el ciclo iterativo:
  1. LLM recibe la pregunta + herramientas disponibles.
  2. LLM decide qué herramienta usar y con qué parámetros.
  3. Se ejecuta la herramienta y se devuelve el resultado al LLM.
  4. LLM evalúa: ¿tengo suficiente? Si no → paso 2. Si sí → sintetiza.
- Máximo de iteraciones: configurable (default 5).
- Si alcanza el límite, sintetiza con lo que tiene (no falla).
- Acumular todos los resultados de herramientas para el contexto final de síntesis.
- Logging por paso via logger: herramienta usada, query/parámetros, tokens del resultado, tiempo.

#### 6.3 — System prompt del agente

- Diseñar prompt que guíe el comportamiento:
  - Empezar con búsqueda interna amplia.
  - Refinar con queries distintas si los resultados son insuficientes.
  - Usar `get_file_content` para ver relaciones entre componentes.
  - Recurrir a fuentes externas solo cuando la pregunta involucra tecnologías fuera de la librería interna.
  - No repetir búsquedas con la misma query.
  - Parar cuando tiene suficiente contexto (evitar over-fetching).
- Incluir ejemplos (few-shot) de cómo debería razonar el agente.

#### 6.4 — Integración con fuentes externas

- **Context7**: cliente que consulta la API REST de Context7 directamente. Input: library name + topic. Output: chunks de documentación. Timeout: 5s.
- **AWS docs**: usar el MCP server de AWS docs si disponible, o web fetch de la documentación con parsing básico. Timeout: 5s.
- Ambas integraciones manejan errores gracefully (timeout, servicio no disponible → el agente continúa sin esa fuente).

#### 6.5 — Implementación de `query_internal_knowledge_agent` (`src/mcp/tools_level3.py`)

- Misma interfaz que `query_internal_knowledge` del Nivel 2.
- Input: `question`, `library_id`, `context` (opcional).
- Output: `answer`, `query_id`, `metadata` (herramientas usadas, iteraciones, tokens consumidos).
- Almacenar los chunks **internos** recopilados en query store para `get_retrieval_context` (misma lógica que Nivel 2). No incluye info de Context7/AWS docs.

#### 6.6 — Integración con FastMCP

- Registrar `query_internal_knowledge_agent` en FastMCP.
- `get_retrieval_context` y `get_file_content` compartidas con Nivel 2.
- Herramientas de los tres niveles disponibles en el mismo MCP server.

#### 6.7 — Testing end-to-end

- Ejecutar las mismas queries del golden set.
- Verificar que el agente no entra en loops infinitos, no repite queries, y para cuando tiene suficiente.
- Probar preguntas que requieren fuentes externas para validar que el agente las usa.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Modelo del agente** | Claude Sonnet vs Claude Opus vs GPT-4o | **Configurable, default Claude Sonnet**. Buen balance de costo y capacidad para tool use. Opus para queries muy complejas si el budget lo permite |
| **Máximo de iteraciones** | 3 vs 5 vs 10 | **5 como default**. La mayoría de queries deberían resolverse en 1-3 iteraciones. 5 da margen para queries complejas sin riesgo de costos excesivos |
| **Context7: qué API usar** | API REST vs MCP server de Context7 | **API REST** directamente. El agente ya está en el servidor, no necesita pasar por MCP |
| **AWS docs: approach** | MCP de AWS docs vs web fetch + parse | **MCP de AWS docs** si disponible como herramienta. Si no, web fetch con parsing básico |
| **Qué incluir en `get_retrieval_context`** | Solo chunks internos vs todo lo que el agente recopiló | **Solo chunks internos**. La info de Context7/AWS docs ya está incorporada en la síntesis pero no se almacena en el índice propio |
| **Caching de búsquedas en el loop** | Cache intra-loop vs buscar cada vez | **Cache intra-loop**: si el agente busca la misma query, devolver el resultado cacheado. Evita desperdicio si el agente repite por error |

### Entregables

- [ ] Herramientas internas del agente definidas con JSON schemas
- [ ] Loop agentic implementado con control de iteraciones y caching intra-loop
- [ ] System prompt del agente diseñado y testeado
- [ ] Integración con Context7 y AWS docs con timeouts y degradación graceful
- [ ] `query_internal_knowledge_agent` registrado en FastMCP
- [ ] Logging detallado de cada iteración del agente
- [ ] Herramientas de los tres niveles disponibles en el mismo MCP server

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| El agente puede hacer over-fetching: búsquedas innecesarias que no aportan | Medio — costo y latencia innecesarios | Medir over-fetching como métrica. Optimizar system prompt si es alto. Límite de 5 iteraciones como guardrail. Cache intra-loop evita búsquedas duplicadas |
| Context7 o AWS docs pueden tener downtime o latencia alta | Bajo — degradación graceful | Timeouts de 5s por herramienta externa. Si falla, el agente continúa con fuentes internas |
| Costos del agente pueden escalar rápido (N llamadas LLM por query) | Medio — imprevisible hasta medir | Estimar costo por query antes de correr evaluación completa. Con Sonnet a ~$3/M output tokens y 3 iteraciones promedio (~6K tokens): ~$0.02/query |
| El agente puede no beneficiar queries simples (overhead sin ganancia) | Bajo — el análisis comparativo revelará esto | Resultado válido del experimento. Documentar en qué tipos de preguntas el agente aporta y en cuáles no |

### Dependencias

- Fase 5 (Nivel 2 funcional — se reutiliza la estructura y se comparten herramientas)
- Acceso a la API de Context7
- Créditos suficientes de API para las llamadas del agente

---

## Fase 7 — Evaluación

### Objetivo

Ejecutar una evaluación rigurosa de los tres niveles sobre el mismo golden set, recopilando todas las métricas definidas en la spec. Separar retrieval recall (calidad del hybrid search crudo) de context recall (calidad de lo que efectivamente llega al LLM).

### Tareas

#### 7.1 — Construcción del golden set

- Crear 30-50 preguntas categorizadas:
  - **API usage** (10-15): "cómo uso la función X", "qué parámetros acepta Y"
  - **Conceptual** (5-8): "cuál es la best practice para Z", "cómo funciona el retry"
  - **Migración** (5-8): "qué cambió entre v1 y v2", "cómo migro de X a Y"
  - **Troubleshooting** (5-8): "error cuando hago Z", "por qué falla la conexión"
  - **Cross-module** (3-5): "cómo interactúan X e Y", "dependencias entre módulos"
  - **Exacta** (3-5): "qué retorna la función X con parámetro Y"
- Para cada pregunta definir:
  - `question`: la pregunta
  - `expected_answer`: respuesta de referencia
  - `expected_chunks`: chunks que deberían ser recuperados (para medir recall)
  - `category`: tipo de pregunta
  - `difficulty`: easy / medium / hard
- `scripts/generate_golden_set.py` para generación semi-automática con LLM + curación manual.

#### 7.2 — Framework de evaluación automatizada (`scripts/run_eval.py`)

- Implementar script que ejecuta cada pregunta contra los tres niveles.
- **Cachear query embeddings** durante evaluación para reutilizar entre los tres niveles (el embedding de la query es el mismo, solo cambia lo que se hace con los resultados).
- Recopilar automáticamente:

**Métricas base (todos los niveles):**
  - Tokens consumidos por el cliente (tamaño de la respuesta)
  - Tokens consumidos por el servidor (API calls de síntesis/agente)
  - Latencia end-to-end
  - Costo total (embeddings de query + LLM calls)

**Retrieval recall@k** (resultado crudo del hybrid search, SIN context builder):
  - Mide la calidad de la búsqueda pura.
  - Ejecutar hybrid search con k=5, 10, 20 y medir qué proporción de `expected_chunks` aparece.
  - Matching: exacto por symbol/header, parcial si el chunk contiene al esperado, por source para proposiciones.
  - Esta métrica es la misma para los tres niveles (el hybrid search es compartido).

**Context recall** (lo que efectivamente entra en el context builder):
  - Mide la calidad de lo que el LLM recibe como contexto.
  - Para Nivel 1: proporción de `expected_chunks` en la salida del context builder con `max_tokens` default.
  - Para Niveles 2/3: proporción de `expected_chunks` en el prompt de síntesis (~8K tokens).
  - Diferir de retrieval recall señala que el context builder (deduplicación, priorización, budget) está filtrando chunks relevantes.

**Calidad de respuesta:**
  - Para el Nivel 1: ejecutar una llamada adicional al LLM con los chunks recuperados usando el **mismo modelo** que los Niveles 2 y 3 usan para sintetizar (control de variable).
  - Para Niveles 2 y 3: usar la respuesta sintetizada directamente.
  - Evaluación humana: correcto / parcial / incorrecto.

#### 7.3 — Evaluación humana

- Evaluar cada respuesta (de los tres niveles) como: correcto / parcial / incorrecto.
- **Rúbrica**: correcto = responde la pregunta sin errores, parcial = responde pero le falta algo o tiene un detalle incorrecto, incorrecto = no responde o responde mal.
- Registrar observaciones cualitativas: ¿falta contexto? ¿hay hallucination? ¿la respuesta es demasiado larga/corta?
- **Evaluación a ciegas**: presentar las respuestas en orden random sin indicar el nivel. Evita sesgo.
- **LLM-as-judge como complemento**: usar Claude para evaluar correctitud vs expected_answer como señal adicional, pero la decisión final es humana.

#### 7.4 — Métricas específicas por nivel

- **Nivel 1**: frecuencia de file fetch, impacto de max_tokens (correr con 1000, 3000, 5000, 10000).
- **Nivel 2**: frecuencia de escape (cuántas veces se necesita `get_retrieval_context`), corrección post-síntesis (cuando el cliente pide chunks, ¿llega a la misma conclusión que la síntesis?), compresión (ratio tokens contexto interno vs tokens respuesta sintetizada).
- **Nivel 3**: iteraciones del agente, distribución de herramientas, over-fetching, valor de fuentes externas.

#### 7.5 — Evaluación transversal del indexing

- Ejecutar Nivel 1 dos veces: con y sin proposiciones atómicas.
- Comparar **retrieval recall@k** y **context recall** para medir el aporte de las tarjetas.
- Analizar por categoría de pregunta: ¿las proposiciones ayudan más en queries conceptuales?

#### 7.6 — Análisis y reporte (`eval/analysis.py`)

- Consolidar todas las métricas en tablas comparativas.
- Generar visualizaciones: barplots de calidad por nivel, scatter de costo vs calidad, distribución de latencia, retrieval recall vs context recall.
- Escribir conclusiones: cuándo conviene cada approach, cuánto impacta el indexing vs la arquitectura.
- Identificar patrones: tipos de preguntas donde cada nivel es más fuerte.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Tamaño del golden set** | 30 vs 50 vs 100 preguntas | **30-50**. Suficiente para observar tendencias. 100 aumentaría la rigurosidad pero también el costo de evaluación humana |
| **Evaluador humano** | Una persona vs múltiples (inter-rater agreement) | **Una persona** para la primera iteración. Si los resultados son ambiguos, agregar un segundo evaluador |
| **LLM-as-judge** | Solo evaluación humana vs humana + LLM judge | **Humana + LLM judge** como complemento. Decisión final siempre humana |
| **Blind evaluation** | Evaluador sabe qué nivel vs evaluación a ciegas | **A ciegas** para las métricas de calidad. Evita sesgo |
| **Caching de query embeddings** | Regenerar por cada nivel vs cachear y reutilizar | **Cachear**. El embedding de la query es el mismo para los tres niveles. Ahorra costo y asegura que la comparación es justa |

### Entregables

- [ ] Golden set de 30-50 preguntas con expected_answer y expected_chunks
- [ ] Framework de evaluación automatizada con caching de embeddings
- [ ] Resultados de retrieval recall@k (crudo) y context recall (post context builder) para cada nivel
- [ ] Resultados de evaluación humana para cada pregunta × cada nivel
- [ ] Tablas comparativas de métricas (base + por nivel + transversales)
- [ ] Reporte de análisis con conclusiones y visualizaciones
- [ ] Dataset raw de todas las métricas para referencia futura

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Golden set no representativo del uso real | Alto — conclusiones sesgadas | Derivar preguntas de queries reales que harían usuarios de las librerías. Cubrir todas las categorías definidas |
| Evaluación humana subjetiva ("parcial" es ambiguo) | Medio — inconsistencia en ratings | Rúbrica clara definida arriba. En caso de duda, marcar como parcial y anotar la observación |
| Costo de correr todos los niveles para 50 preguntas | Bajo-Medio — especialmente Nivel 3 | Estimar costo total antes de correr. Budget aproximado: $5-15 para el run completo |

### Dependencias

- Fases 4, 5 y 6 (los tres niveles funcionales)
- Tiempo para evaluación humana (estimar 2-3 horas para 50 preguntas × 3 niveles)

---

## Fase 8 — Blog Series y Librería

### Objetivo

Documentar los resultados del experimento como una serie de 4 blogs y empaquetar el código como una librería reutilizable.

### Tareas

#### 8.1 — Blog Parte 1: Retrieval Puro

- Introducir el problema y la motivación.
- Describir la arquitectura base: Supabase, pgvector, hybrid search.
- Detallar la estrategia de indexing: AST-aware chunking, proposiciones atómicas.
- Mostrar la implementación del MCP server como context provider.
- Incluir resultados del Nivel 1: retrieval recall, context recall, consumo de tokens, impacto del indexing.
- Código de ejemplo y diagramas.

#### 8.2 — Blog Parte 2: Retrieval + Synthesis

- Introducir la generación server-side.
- Mostrar cómo la síntesis reduce contexto y ruido.
- Comparar contra Nivel 1 con métricas concretas.
- Discutir prompt engineering para síntesis.
- Trade-offs de costo vs calidad.

#### 8.3 — Blog Parte 3: Agentic Retrieval

- Introducir el modelo agentic con tool use.
- Mostrar cómo el agente decide dinámicamente.
- Comparar latencia, costo y calidad contra niveles anteriores.
- Explorar los límites: cuándo mejora y cuándo agrega complejidad.

#### 8.4 — Blog Parte 4: Resultados y Producto Final

- Análisis comparativo profundo con todas las métricas.
- Conclusiones del experimento.
- Lecciones aprendidas.
- Presentación de la librería final.

#### 8.5 — Empaquetado de la librería

- Refactorizar el código en un paquete instalable.
- Documentar API pública.
- Incluir las mejores decisiones de diseño validadas durante el experimento.
- README con quickstart, configuración, y ejemplos.

### Decisiones de Diseño

| Decisión | Opciones | Resolución |
|----------|----------|------------|
| **Plataforma de publicación** | Blog personal vs Medium vs dev.to vs GitHub Pages | **Pendiente** — depende de la audiencia target y preferencia personal |
| **Idioma de los blogs** | Español vs inglés | **Pendiente** — inglés para mayor audiencia, español si el target es la comunidad local |
| **Licencia de la librería** | MIT vs Apache 2.0 vs propietaria | **Pendiente** — si es open source, MIT o Apache 2.0. Si es para uso interno únicamente, propietaria |

### Entregables

- [ ] 4 blog posts escritos y publicados
- [ ] Librería empaquetada con documentación
- [ ] Repositorio público (o interno) con código, datos de evaluación y resultados

### Riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Los resultados no muestran diferencias claras entre niveles | Bajo — es un resultado válido | "No hay diferencia significativa" es una conclusión útil. Documentar en qué condiciones cada nivel empata y en cuáles diverge |
| El código experimental no es suficientemente limpio para liberar como librería | Medio — requiere refactoring significativo | Planificar el refactoring como parte de esta fase, no como afterthought. Definir la API pública antes de empezar |

### Dependencias

- Fase 7 (resultados de evaluación completos)

---

## Resumen de Dependencias entre Fases

```
Fase 0 (Infra)
   │
   ▼
Fase 1 (Curación de contenido)
   │
   ▼
Fase 2 (Indexing Pipeline)
   │
   ▼
Fase 3 (Search & Retrieval)
   │
   ▼
Fase 4 (Nivel 1 - Retrieval puro)
   │
   ▼
Fase 5 (Nivel 2 - Retrieval + Synthesis)
   │
   ▼
Fase 6 (Nivel 3 - Agentic Retrieval)
   │
   ▼
Fase 7 (Evaluación)
   │
   ▼
Fase 8 (Blogs + Librería)
```

Cada fase depende estrictamente de la anterior. No hay paralelismo.
