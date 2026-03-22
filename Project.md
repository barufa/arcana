# Internal Knowledge MCP Server — Proyecto Experimental

---

# 1. Visión del Proyecto

El proyecto consiste en construir un **MCP server experimental para conocimiento técnico de librerías internas**, cuyo objetivo es mejorar cómo asistentes como Claude Code utilizan APIs internas.

El sistema permitirá responder preguntas sobre:

- Uso de APIs internas
- Ejemplos de código
- Mejores prácticas
- Migraciones entre versiones

El proyecto tiene un propósito **educativo y experimental**, por lo que implementará **tres arquitecturas distintas** de MCP server para comparar enfoques.

Estas arquitecturas representan tres paradigmas actuales en sistemas RAG y agentes:

1. **Retrieval puro** — Context provider
2. **Retrieval + síntesis** — Context synthesizer
3. **Retrieval agentic** — LLM que usa herramientas

---

# 2. Infraestructura Compartida

Los tres niveles del experimento comparten la misma infraestructura base.

## 2.1 MCP Server

Implementado con Python y **FastMCP**.

FastMCP permite definir herramientas MCP y también ejecutar **Sampling**, donde una herramienta del servidor puede pedir generación al LLM del cliente mediante `ctx.sample()`.

Sin embargo, el soporte de Sampling en clientes MCP es todavía limitado. Como alternativa práctica para los niveles 2 y 3, el servidor realizará **llamadas directas a la API de Anthropic o OpenAI** para la generación. Conceptualmente es equivalente — el server recupera contexto, arma el prompt y pide generación — solo cambia quién ejecuta el LLM. El costo lo absorbe el servidor, lo cual es aceptable para un experimento.

Cuando el soporte de Sampling madure en clientes como Claude Code, se podrá migrar a `ctx.sample()` sin cambios arquitectónicos.

## 2.2 Base de Datos

Se utilizará **Supabase**, que provee:

- PostgreSQL gestionado
- Full-Text Search
- Extensión **pgvector**

Esto permite implementar búsqueda lexical, semántica e híbrida en una sola base de datos.

## 2.3 Tipo de Información Indexada

El índice contendrá:

- Chunks de código (funciones, clases, métodos)
- Firmas de funciones y type hints
- Docstrings
- Ejemplos de uso
- Documentación técnica (markdown)
- Blogs y guías internas
- Changelogs
- Metadatos de librerías
- Proposiciones atómicas extraídas por LLM

El objetivo es construir un **knowledge index interno** que cubra tanto código como documentación.

---

# 3. Estrategia de Indexing

El sistema utiliza **tres capas de indexing** sobre el mismo contenido fuente, cada una optimizada para un tipo distinto de query.

## 3.1 Índice de Código

### Modelo de Embeddings

**Voyage Code 3** (Voyage AI / Anthropic).

Está específicamente entrenado para código y documentación técnica. Consistentemente rankea arriba en benchmarks de code retrieval. Soporta hasta 16K tokens de contexto.

Dimensión del vector: **1024**.
Precio: **$0.18 / 1M tokens** (primeros 200M tokens gratis).

### Chunking: AST-aware con Tree-sitter

Se utiliza **tree-sitter** como parser para extraer unidades semánticas del código. Tree-sitter es el parser que usan los IDEs, soporta múltiples lenguajes y produce ASTs con byte ranges exactos para extraer source.

El principio fundamental es que un chunk de código debe ser una **unidad semántica completa**: una función, una clase, un método. Nunca se corta una función a la mitad.

Se extraen tres niveles de granularidad:

- **Módulo**: el archivo entero como contexto amplio
- **Clase**: con todos sus métodos
- **Función/método individual**: la unidad más granular

Los tres se indexan y al momento de retrieval el score decide cuál es más relevante.

### Metadata por Chunk

Cada chunk de código se almacena con metadata enriquecida:

- **Signature completa** con type hints
- **Docstring** (si existe)
- **Imports relevantes** (solo los que usa esa función, no todos los del archivo)
- **Filepath y módulo** al que pertenece
- **Dependencias internas** (si la función llama a otras funciones de la librería)
- **Librería y versión**

El embedding se genera sobre la concatenación de: signature + docstring + body.

### Funciones largas (>800 tokens)

Si una función excede los 800 tokens, se tienen dos opciones:

1. Indexarla entera (Voyage Code 3 soporta hasta 16K tokens)
2. Splitear en signature+docstring como un chunk y el body como otro con referencia cruzada

## 3.2 Índice de Documentación

### Modelo de Embeddings

**Voyage 3.5** (`voyage-3.5`, API de Voyage AI).

Modelo generalista de alta calidad para texto en lenguaje natural. Captura mejor las queries conceptuales tipo "cómo migro de v2 a v3" o "cuál es la best practice para retry". Al usar el mismo proveedor y la misma dimensionalidad que Voyage Code 3, se simplifica la infraestructura: una sola API, un solo billing, y scores de similaridad directamente comparables entre índices.

Dimensión del vector: **1024**.
Precio: **$0.06 / 1M tokens** (primeros 200M tokens gratis).

### Chunking: Secciones Semánticas de Markdown

Cada sección delimitada por un header (`#`, `##`, `###`) es un chunk candidato. Se parsea el markdown como un árbol de headers y se aplica una lógica recursiva:

- Se empieza desde los niveles más profundos
- Si un nodo es muy chico (<100 tokens), se mergea con su padre
- Si un nodo es muy grande (>800 tokens), se deja como chunk individual

**Parser**: `mistune` o `markdown-it-py` para obtener el AST del markdown programáticamente.

### Header Path como Contexto

Cada chunk lleva el **breadcrumb completo de headers** como prefijo. Si una sección dice "Use `exponential_backoff=True` in the client constructor", el chunk se indexa como:

```
SprAI Client SDK > Configuration > Retry Policy > Use exponential_backoff=True in the client constructor
```

El embedding se genera sobre la concatenación de: header path + contenido de la sección.

Se almacenan ambos por separado en Supabase (header path como metadata, contenido como texto, embedding sobre la concatenación).

### Overlap entre Secciones

Las últimas 2-3 oraciones de la sección anterior se incluyen como prefijo al generar el embedding, para capturar continuidad semántica. Este overlap **no se incluye** en el texto que se devuelve al LLM — solo se usa para generar el embedding.

### Elementos Especiales

- **Code blocks dentro de docs**: se indexan tanto en el índice de documentación (con Voyage 3.5) como en el índice de código (con Voyage Code 3). Así una query conceptual los encuentra por la explicación, y una query de código los encuentra por el snippet.
- **Tablas**: se convierten a texto estructurado antes de generar el embedding. "El parámetro timeout acepta int, default 30, controla el tiempo máximo de espera" embedea mejor que el formato tabular markdown.
- **Changelogs**: cada entrada de versión es un chunk natural. Header path: `Changelog > v2.3.0 > Breaking Changes`.

## 3.3 Índice de Proposiciones (Tarjetas Atómicas)

### Concepto

Basado en la técnica de **proposition extraction** (formalizada en "Dense X Retrieval" de Chen et al.). Se utiliza un LLM para descomponer documentos en proposiciones atómicas autocontenidas — "tarjetas" que describen una idea concreta en lenguaje natural, sin referencias externas.

### Por qué funciona

El problema de embeddings sobre texto raw es que un párrafo puede contener varias ideas, y el embedding resultante es un promedio borroso. Con proposiciones atómicas, cada embedding representa una sola idea con alta precisión y el retrieval mejora significativamente.

Además, la tarjeta se genera con el LLM viendo el **documento completo**, lo que resuelve referencias implícitas. Un párrafo que dice "a diferencia del approach anterior, este método usa connection pooling" se convierte en una tarjeta como: "El método X de la librería Y usa connection pooling en vez de conexiones individuales para mejorar performance".

### Modelo de Embeddings

**Voyage 3.5** (`voyage-3.5`, mismo modelo que documentación, ya que las proposiciones son texto en lenguaje natural).

Dimensión del vector: **1024**.
Precio: **$0.06 / 1M tokens** (comparte la cuota de 200M tokens gratis con el índice de documentación).

### Estructura de cada Tarjeta

- **Texto**: la proposición autocontenida en lenguaje natural, sin referencias a "el párrafo anterior" o "como vimos antes". Completamente independiente.
- **Metadata**:
  - Source file
  - Sección (header path)
  - Rango de líneas del texto fuente
  - Texto original de la sección de donde se extrajo la idea
  - Librería y versión
- **Scope**: API usage, configuración, migración, troubleshooting, etc. Permite filtrar en retrieval si la query tiene intención clara.

### Generación

Se utiliza un LLM económico (Claude Haiku o GPT-4o-mini) para extraer las tarjetas. El volumen es acotado (librerías internas) y el indexing corre una vez o cuando cambia la documentación.

Cada sección del documento fuente se hashea. Cuando un documento cambia, solo se re-generan las tarjetas de las secciones cuyo hash cambió.

---

# 4. Estrategia de Búsqueda

La búsqueda es **híbrida**, combinando dos señales complementarias.

## 4.1 Full Text Search

Útil para:

- Símbolos y nombres de funciones
- Imports
- Tokens exactos
- Nombres de parámetros

## 4.2 Vector Search

Útil para:

- Preguntas en lenguaje natural
- Queries conceptuales
- Casos donde el usuario no conoce la API exacta

## 4.3 Retrieval Multi-índice

Al buscar, se consultan los tres índices (código, documentación, proposiciones) y se mergean los resultados.

Aunque ambos modelos (Voyage Code 3 y Voyage 3.5) producen vectores de la misma dimensionalidad (1024) y son del mismo proveedor, los scores de cosine similarity no son directamente comparables. Por esto se utiliza **Reciprocal Rank Fusion (RRF)** para combinar resultados, ya que RRF opera sobre rankings relativos y no sobre scores absolutos, eliminando el problema de calibración entre modelos.

Una query técnica como "retry configuration" matchea fuerte en código y docs. Una query conceptual como "cómo manejo errores transientes" matchea mejor en proposiciones.

---

# 5. Schema en Supabase

Se utilizan dos tablas separadas para mantener una separación clara entre código y documentación, aunque ambas usan vectores de la misma dimensionalidad (1024).

## 5.1 Tabla `code_chunks`

| Columna          | Tipo              | Descripción                          |
| ---------------- | ----------------- | ------------------------------------ |
| id               | uuid              | PK                                   |
| library          | text              | Nombre de la librería                |
| version          | text              | Versión                              |
| filepath         | text              | Ruta del archivo fuente              |
| module           | text              | Módulo Python                        |
| chunk_type       | text              | function / class / method / module   |
| symbol_name      | text              | Nombre de la función/clase           |
| signature        | text              | Firma completa con type hints        |
| docstring        | text              | Docstring extraído                   |
| source_code      | text              | Código fuente del chunk              |
| imports          | text[]            | Imports relevantes                   |
| dependencies     | text[]            | Funciones internas que llama         |
| start_line       | int               | Línea de inicio en el archivo        |
| end_line         | int               | Línea de fin en el archivo           |
| embedding        | vector(1024)      | Voyage Code 3                        |
| content_hash     | text              | Hash del contenido para re-indexing  |

## 5.2 Tabla `doc_chunks`

| Columna          | Tipo              | Descripción                          |
| ---------------- | ----------------- | ------------------------------------ |
| id               | uuid              | PK                                   |
| library          | text              | Nombre de la librería                |
| version          | text              | Versión                              |
| source_file      | text              | Archivo markdown fuente              |
| content_type     | text              | section / proposition / changelog    |
| header_path      | text              | Breadcrumb completo de headers       |
| scope            | text              | api_usage / config / migration / ... |
| content          | text              | Texto del chunk o proposición        |
| original_section | text              | Texto original (para proposiciones)  |
| start_line       | int               | Línea de inicio                      |
| end_line         | int               | Línea de fin                         |
| embedding        | vector(1024)      | Voyage 3.5                           |
| content_hash     | text              | Hash del contenido para re-indexing  |

Ambas tablas requieren índices **HNSW** en pgvector para búsqueda eficiente.

---

# 6. Arquitectura Experimental en Tres Niveles

## Nivel 1 — Retrieval Puro (Context7-style)

Inspirado en el modelo de Context7: recupera documentación actualizada y la inyecta en el prompt del LLM sin sintetizar. A diferencia de Context7 que entrega snippets de documentación en una sola llamada, este nivel implementa un patrón **browse + fetch** que le da al LLM del cliente control sobre qué información profundizar.

### Arquitectura

```
Client LLM
   │
   ▼
resolve_library(library_name, query)
   │
   ▼
search_library(library_id, query, max_tokens?)
   │
   ▼
Hybrid Retrieval (Supabase)
   ├── code_chunks (Voyage Code 3)
   ├── doc_chunks sections (Voyage 3.5)
   └── doc_chunks propositions (Voyage 3.5)
   │
   ▼
Ranking compacto de chunks relevantes
   │
   ▼
LLM decide si necesita más contexto
   │          │
   │ NO       │ SÍ
   │          ▼
   │   get_file_content(file_id, start_line?, end_line?)
   │          │
   ▼          ▼
LLM genera respuesta final (client-side)
```

El MCP server **no sintetiza la respuesta**. Solo entrega información.

El flujo típico es: `resolve_library` → `search_library` → responde. Opcionalmente, si el LLM necesita contexto más amplio que los chunks individuales (por ejemplo, ver el archivo completo de un módulo, entender imports o constantes definidas fuera del chunk, o explorar la relación entre componentes), llama a `get_file_content` para obtener el archivo fuente.

### Herramientas Expuestas

**`resolve_library`**

Resuelve un nombre de librería a un ID. Si hay ambigüedad, devuelve opciones.

- Input: `library_name: str`, `query: str`
- Output: lista de librerías candidatas con `library_id`, `name`, `version`

**`search_library`**

Ejecuta hybrid search (vector + full-text) en las tres capas de indexing y devuelve un ranking compacto de resultados. Cada resultado incluye suficiente detalle para que el LLM pueda responder la mayoría de las preguntas sin necesidad de llamadas adicionales.

- Input:
  - `library_id: str`
  - `query: str`
  - `max_tokens: int` (opcional, default 5000) — Controla el volumen de la respuesta. Con valores bajos (~1000) devuelve un ranking con signatures y descripciones de una línea. Con valores altos (~5000-10000) expande los resultados de mayor score con source code completo, docstrings y secciones de documentación.
- Output: ranking de chunks, donde cada item incluye:
  - Tipo: `function`, `class`, `method`, `doc`, `proposition`
  - Identificador: symbol name o header path
  - Contexto: signature completa (para código) o primera oración (para docs)
  - Detalle: source code, docstring, o sección completa (según budget de tokens)
  - Metadata: `file_id`, `start_line`, `end_line`, `filepath`

El context builder llena el budget de tokens con una prioridad de llenado: primero incluye todos los resultados en forma compacta (signature + descripción), y si queda budget expande los resultados de mayor score con contenido completo. Esto permite que un `max_tokens` bajo devuelva un overview rápido y uno alto devuelva documentación detallada, similar al parámetro `tokens` de Context7.

**`get_file_content`**

Devuelve el contenido de un archivo fuente completo, o un rango de líneas. Para cuando el LLM necesita contexto más amplio que el chunk individual: ver el módulo completo, entender imports, constantes, o la relación entre funciones en el mismo archivo.

- Input:
  - `file_id: str` (presente en la metadata de cada chunk del ranking)
  - `start_line: int` (opcional)
  - `end_line: int` (opcional)
- Output: contenido del archivo con números de línea

### Deduplicación

El context builder deduplica resultados redundantes antes de armar la respuesta. Si una proposición y la sección de documentación de la que fue extraída están ambas en el top-k, se incluye la sección (más completa) y se descarta la proposición. Si un chunk de función y el chunk de clase que la contiene están ambos en el ranking, se prioriza el más granular (la función) a menos que múltiples métodos de la misma clase aparezcan, en cuyo caso se incluye la clase como overview.

### Propósito

Medir:

- Cuánto contexto consume el LLM en total (tokens del ranking + tokens de file fetches)
- Qué tan bien responde con retrieval puro
- Con qué frecuencia el LLM necesita llamar a `get_file_content` (indicador de suficiencia del chunking)
- Eficiencia de retrieval: proporción de preguntas respondidas correctamente solo con el ranking vs las que requieren fetch adicional

---

## Nivel 2 — Retrieval + Synthesis

En este nivel el retrieval sigue siendo determinista (misma hybrid search que el Nivel 1), pero se introduce **generación server-side** para sintetizar la respuesta antes de entregarla al cliente. El cliente recibe una respuesta concisa en vez de chunks crudos.

La diferencia fundamental con el Nivel 1: el servidor procesa internamente el ruido, resuelve contradicciones entre documentación y código, descarta chunks irrelevantes, y entrega una respuesta que ya hizo el trabajo de interpretación. El cliente no necesita razonar sobre múltiples chunks ni decidir qué profundizar.

Sin embargo, el Nivel 2 no es una caja negra. El cliente puede verificar las fuentes que informaron la síntesis y explorar archivos completos si necesita más contexto. Esto habilita un patrón de **tres niveles de profundidad**:

1. **Respuesta rápida**: la síntesis resuelve la mayoría de las preguntas.
2. **Verificar fuentes**: el cliente inspecciona los chunks que informaron la respuesta.
3. **Explorar a fondo**: el cliente accede al archivo completo para contexto amplio.

Implementado via **llamada directa a la API de Anthropic o OpenAI** (reemplazando `ctx.sample()` hasta que Sampling tenga soporte amplio en clientes MCP). El modelo de síntesis es configurable; por default se usa el mismo modelo que evaluará el Nivel 1 para mantener la comparación controlada.

### Arquitectura

```
Client
   │
   ▼
resolve_library(library_name, query)
   │
   ▼
query_internal_knowledge(question, library_id, context?)
   │
   ▼
Hybrid Retrieval (Supabase)
   ├── code_chunks (Voyage Code 3)
   ├── doc_chunks sections (Voyage 3.5)
   └── doc_chunks propositions (Voyage 3.5)
   │
   ▼
Context Builder (optimizado para síntesis, budget interno ~8K tokens)
   │
   ▼
LLM server-side (API call Anthropic / OpenAI)
   │
   ▼
Respuesta sintetizada + query_id → Client
   │
   ▼
LLM decide si necesita más contexto
   │          │
   │ NO       │ SÍ
   │          ▼
   │   get_retrieval_context(query_id)
   │          │
   │          ▼
   │   LLM revisa chunks fuente
   │          │
   │          ├── Suficiente → responde
   │          │
   │          └── Necesita más → get_file_content(file_id)
   │          │
   ▼          ▼
Respuesta final (client-side)
```

### Herramientas Expuestas

**`resolve_library`** — Igual que en el Nivel 1, compartida entre niveles.

**`query_internal_knowledge`**

Ejecuta el pipeline completo de retrieval + síntesis y devuelve una respuesta en lenguaje natural.

- Input:
  - `question: str` — la pregunta del usuario
  - `library_id: str` — obtenido de `resolve_library`
  - `context: str` (opcional) — contexto adicional del usuario (ej: "estoy migrando de v1 a v2", "uso Python 3.11")
- Flujo interno: question → hybrid search → context builder → prompt assembly → API call → respuesta
- Output:
  - `answer: str` — respuesta sintetizada en lenguaje natural, con referencias a funciones o secciones de documentación cuando es relevante
  - `query_id: str` — identificador para recuperar los chunks que informaron la síntesis

El context builder del Nivel 2 es más agresivo que el del Nivel 1: puede incluir más contexto porque no le cuesta ventana al cliente. El budget interno es configurable (default 8000 tokens de contexto al LLM de síntesis).

Si el retrieval no encuentra resultados relevantes, el LLM de síntesis responde explícitamente que no hay información suficiente en vez de inventar (a chequear como validar esto).

El servidor retiene los chunks asociados al `query_id` en memoria con un TTL corto (configurable, default 30 minutos) para que el cliente pueda consultarlos si lo necesita.

**`get_retrieval_context`**

Devuelve los chunks que fueron usados para generar una respuesta sintetizada. Permite al cliente verificar fuentes, detectar información faltante, o profundizar en chunks específicos.

- Input:
  - `query_id: str` — obtenido de `query_internal_knowledge`
- Output: ranking de chunks en el mismo formato que `search_library` del Nivel 1 (tipo, identificador, signature/header path, contenido, file_id, líneas). Esto permite que el LLM del cliente los interprete de la misma forma que interpretaría resultados del Nivel 1.

**`get_file_content`** — Igual que en el Nivel 1, compartida entre niveles.

### Propósito

Medir comparado con el Nivel 1:

- Reducción de tokens que recibe el cliente (respuesta sintetizada vs chunks crudos)
- Costo server-side (tokens consumidos por la llamada API de síntesis)
- Calidad de respuesta: ¿la síntesis mejora, empeora, o es equivalente a que el LLM del cliente interprete los chunks?
- Latencia adicional introducida por la llamada de generación
- Casos donde la síntesis pierde información que los chunks crudos preservaban
- **Frecuencia de escape**: con qué frecuencia el LLM del cliente necesita llamar a `get_retrieval_context` para verificar o complementar la síntesis. Si es baja, la síntesis funciona como diseñada. Si es alta, el Nivel 2 no aporta suficiente valor sobre el Nivel 1.
- **Corrección post-síntesis**: cuando el cliente sí pide los chunks, ¿llega a la misma conclusión que la síntesis, o la corrige? Esto mide la confiabilidad del LLM de síntesis.

---

## Nivel 3 — Agentic Retrieval

En este nivel la interfaz hacia el cliente es idéntica al Nivel 2 (mismas herramientas expuestas, mismo formato de respuesta), pero el proceso interno cambia fundamentalmente: en vez de un retrieval determinista seguido de una síntesis, un **LLM agente** decide dinámicamente qué buscar, cuántas veces, y en qué orden antes de sintetizar la respuesta.

La diferencia con el Nivel 2: el retrieval del Nivel 2 ejecuta una sola búsqueda híbrida y sintetiza lo que encuentra. El Nivel 3 puede hacer múltiples búsquedas, reformular queries, cruzar información entre búsquedas, y además consultar fuentes externas (Context7, AWS docs) cuando la librería interna depende de tecnologías externas. El agente razona sobre los resultados intermedios y decide si necesita más información antes de responder.

Implementado via **llamada a la API con tool use** (function calling), donde el LLM del servidor recibe herramientas de búsqueda y ejecuta un loop iterativo hasta tener suficiente contexto para responder.

### Arquitectura

```
Client
   │
   ▼
resolve_library(library_name, query)
   │
   ▼
query_internal_knowledge_agent(question, library_id, context?)
   │
   ▼
Agent LLM (API call con tool use)
   │
   ▼
Loop agentic (máx N iteraciones):
   │
   ├── search_library(library_id, query, max_tokens)
   │     → ranking de chunks internos
   │
   ├── get_file_content(file_id, start_line?, end_line?)
   │     → archivo fuente completo o parcial
   │
   ├── context7_docs(library_name, topic)
   │     → documentación de librerías externas
   │
   ├── aws_docs(query)
   │     → documentación de servicios AWS
   │
   │   El agente evalúa los resultados:
   │   ├── ¿Tengo suficiente contexto? → Sintetiza respuesta
   │   └── ¿Necesito más? → Reformula query, busca en otro índice,
   │                          consulta fuente externa, o pide otro archivo
   │
   ▼
Respuesta sintetizada + query_id → Client
   │
   ▼
(Mismo flujo de verificación que Nivel 2)
   ├── get_retrieval_context(query_id) → chunks internos usados
   └── get_file_content(file_id) → exploración bajo demanda
```

### Herramientas Expuestas al Cliente

La interfaz MCP es idéntica al Nivel 2:

**`resolve_library`** — Compartida entre niveles.

**`query_internal_knowledge_agent`**

Misma interfaz que `query_internal_knowledge` del Nivel 2, pero internamente el retrieval es agentic.

- Input:
  - `question: str`
  - `library_id: str`
  - `context: str` (opcional)
- Output:
  - `answer: str` — respuesta sintetizada
  - `query_id: str` — para recuperar chunks internos usados
  - `metadata: object` — información sobre el proceso agentic: herramientas usadas, número de iteraciones, tokens consumidos por el agente

**`get_retrieval_context`** — Compartida con el Nivel 2. Devuelve los chunks **internos** que el agente recopiló durante sus búsquedas. No incluye información de fuentes externas (Context7, AWS docs), ya que esos resultados no están almacenados en el índice propio. Sin embargo, la respuesta sintetizada sí los incorpora.

**`get_file_content`** — Compartida entre niveles.

### Herramientas Internas del Agente

Estas herramientas NO son MCP tools expuestas al cliente. Son funciones internas que el LLM del servidor puede llamar via tool use de la API durante el loop agentic.

**Herramientas de búsqueda interna** (reutilizan la lógica del Nivel 1):

- `search_library` — hybrid search en las tres capas de indexing, devuelve ranking de chunks
- `get_file_content` — acceso a archivos fuente completos

**Herramientas de búsqueda externa** (opcionales, para cuando la librería interna depende de tecnologías externas):

- `context7_docs` — consulta documentación de librerías open source via Context7
- `aws_docs` — consulta documentación de servicios AWS

El agente decide cuándo usar herramientas externas basándose en el contexto. Si un chunk interno referencia un servicio AWS (ej: "usa SQS para el queue de mensajes") y la pregunta es sobre configuración, el agente puede consultar la documentación de AWS para complementar la respuesta.

### Comportamiento del Agente

El system prompt del agente incluye guías sobre:

- Empezar con una búsqueda interna amplia y refinar si los resultados son insuficientes
- Usar `search_library` con queries distintas si la primera no devuelve resultados relevantes
- Consultar `get_file_content` cuando necesita ver relaciones entre componentes
- Recurrir a fuentes externas solo cuando la pregunta involucra tecnologías fuera de la librería interna
- No repetir búsquedas con la misma query
- Parar de buscar cuando tiene suficiente contexto (evitar over-fetching)

Límite máximo de iteraciones: configurable, default 5. Si el agente alcanza el límite, sintetiza con lo que tiene.

### Propósito

Medir comparado con los Niveles 1 y 2:

- **Calidad**: ¿el agente encuentra información que el retrieval estático del Nivel 2 no encontraba? Especialmente en preguntas cross-module o que involucran dependencias externas
- **Costo**: tokens totales del agente (suma de todas las iteraciones del loop) vs una sola búsqueda + síntesis del Nivel 2
- **Latencia**: overhead acumulado de múltiples búsquedas y llamadas LLM
- **Eficiencia del agente**: número promedio de iteraciones, distribución de herramientas usadas, proporción de búsquedas que realmente aportaron información nueva
- **Over-fetching**: con qué frecuencia el agente hace búsquedas innecesarias que no aportan a la respuesta final
- **Valor de fuentes externas**: cuántas preguntas se benefician de Context7/AWS docs vs cuántas se resuelven solo con el índice interno
- **Frecuencia de escape del cliente**: ¿el cliente necesita verificar fuentes con la misma frecuencia que en el Nivel 2, o la confía más porque el agente fue más exhaustivo?

---

# 7. Comparación entre los Tres Niveles

| Nivel | Arquitectura         | Control | Flexibilidad | Latencia | Costo Server |
| ----- | -------------------- | ------- | ------------ | -------- | ------------ |
| 1     | Retrieval puro       | Alto    | Bajo         | Baja     | Bajo         |
| 2     | Retrieval + síntesis | Alto    | Medio        | Media    | Medio        |
| 3     | Agentic retrieval    | Medio   | Alto         | Alta     | Alto         |

# 7. Comparación entre los Tres Niveles

| Dimensión                        | Nivel 1 — Retrieval puro       | Nivel 2 — Retrieval + Synthesis    | Nivel 3 — Agentic Retrieval          |
| -------------------------------- | ------------------------------ | ---------------------------------- | ------------------------------------ |
| Interfaz al cliente              | search + fetch                 | ask + verify                       | ask + verify (idéntica al Nivel 2)   |
| Quién interpreta los chunks      | LLM del cliente                | LLM del servidor                   | LLM del servidor (agente)            |
| Retrieval                        | Una búsqueda híbrida           | Una búsqueda híbrida               | Múltiples búsquedas iterativas       |
| Fuentes externas                 | No                             | No                                 | Sí (Context7, AWS docs)              |
| Tokens consumidos por el cliente | Alto (recibe chunks)           | Bajo (recibe respuesta)            | Bajo (recibe respuesta)              |
| Tokens consumidos por el servidor| Bajo (solo retrieval)          | Medio (retrieval + 1 LLM call)     | Alto (retrieval + N LLM calls)       |
| Latencia                         | Baja                           | Media                              | Alta                                 |
| Verificabilidad                  | Total (el cliente ve todo)     | Alta (get_retrieval_context)       | Parcial (solo chunks internos)       |
| Control del cliente              | Alto (decide qué profundizar)  | Medio (puede verificar fuentes)    | Medio (puede verificar fuentes)      |
| Control del servidor             | Bajo (solo entrega datos)      | Medio (sintetiza)                  | Alto (decide qué buscar y sintetiza) |

---

# 8. Métricas de Evaluación

Para que la comparación entre niveles sea rigurosa, se definen métricas organizadas en tres categorías: métricas base (aplican a los tres niveles), métricas por nivel (capturan las particularidades de cada arquitectura), y métricas transversales (comparan dimensiones del sistema de indexing).

### Métricas Base (todos los niveles)

- **Calidad de respuesta**: rating humano (correcto / parcial / incorrecto) sobre un golden set de 30-50 preguntas. Para que la comparación sea controlada, en el Nivel 1 se hace una llamada al LLM con los chunks recuperados usando el mismo modelo que los Niveles 2 y 3 usan para sintetizar.
- **Tokens consumidos por el cliente**: tamaño de la respuesta que recibe el LLM del cliente. Alto en el Nivel 1 (chunks crudos), bajo en los Niveles 2 y 3 (respuesta sintetizada).
- **Tokens consumidos por el servidor**: tokens usados en llamadas API server-side. Cero en el Nivel 1, medio en el Nivel 2, alto en el Nivel 3.
- **Latencia end-to-end**: tiempo desde la tool call inicial hasta la respuesta final al cliente.
- **Costo total**: costo combinado de embeddings (query) + LLM calls (síntesis), expresado en dólares por pregunta.
- **Precisión de retrieval**: recall@k (k=5, 10, 20) medido contra los `expected_chunks` del golden set.

### Métricas del Nivel 1

- **Eficiencia de retrieval**: proporción de preguntas respondidas correctamente solo con el ranking de `search_library` vs las que requieren `get_file_content`.
- **Frecuencia de file fetch**: cuántas veces por pregunta el LLM llama a `get_file_content`, indicador de suficiencia del chunking y la metadata.
- **Impacto de max_tokens**: cómo varía la calidad de respuesta al cambiar el budget de tokens (1000, 3000, 5000, 10000).

### Métricas del Nivel 2

- **Frecuencia de escape**: con qué frecuencia el LLM del cliente llama a `get_retrieval_context` para verificar o complementar la síntesis. Baja = la síntesis es suficiente.
- **Corrección post-síntesis**: cuando el cliente sí pide los chunks, ¿llega a la misma conclusión que la síntesis, o la corrige? Mide confiabilidad del LLM de síntesis.
- **Compresión**: ratio entre tokens del contexto interno (budget del context builder) y tokens de la respuesta sintetizada.

### Métricas del Nivel 3

- **Iteraciones del agente**: número promedio de tool calls por pregunta.
- **Distribución de herramientas**: qué herramientas usa el agente y con qué frecuencia (search_library, get_file_content, context7_docs, aws_docs).
- **Over-fetching**: proporción de búsquedas del agente que no aportaron información nueva a la respuesta final.
- **Valor de fuentes externas**: cuántas preguntas se benefician de Context7/AWS docs vs cuántas se resuelven solo con el índice interno.
- **Frecuencia de escape del cliente**: comparada contra el Nivel 2, ¿el cliente confía más en la respuesta del agente porque fue más exhaustivo?

### Métricas Transversales del Indexing

- **Proposiciones vs chunks raw**: se ejecuta la evaluación del Nivel 1 dos veces (con y sin proposiciones habilitadas) y se compara recall@k. Mide cuánto aporta la extracción de tarjetas atómicas.
- **Calidad por categoría de pregunta**: breakdown de todas las métricas por tipo de pregunta (API usage, conceptual, migración, troubleshooting, cross-module, exacta) para identificar en qué escenarios cada nivel es más fuerte.

---

# 9. Filosofía del Proyecto

El proyecto busca explorar tres modelos de interacción MCP:

- **Context provider** → Nivel 1 (Context7-style)
- **Context synthesizer** → Nivel 2 (RAG con síntesis)
- **Agentic retrieval** → Nivel 3 (LLM que usa herramientas)

El objetivo es entender:

- Cuándo el retrieval puro es suficiente
- Cuándo conviene sintetizar
- Cuándo un agente mejora el resultado
- Cuánto impacta la calidad del indexing (especialmente proposiciones) vs la sofisticación de la arquitectura

### Insight clave del diseño

Context7 demuestra que **retrieval simple puede funcionar muy bien si el índice es bueno**. Por eso el proyecto separa claramente:

- La **capa de datos** (Supabase + tres índices)
- La **capa MCP** (herramientas expuestas)
- La **capa de razonamiento** (retrieval puro / síntesis / agentic)

Esto permite experimentar con distintas arquitecturas sin cambiar el backend.

---

# 10. Objetivo Final y Serie de Blogs

El objetivo del proyecto **no es entregar un producto terminado**, sino realizar una experimentación rigurosa y documentar los resultados de forma pública.

La salida principal del proyecto es una **serie de 4 blogs** que guían al lector a través de la evolución del sistema, desde retrieval puro hasta comportamiento agentico, mostrando resultados concretos en cada etapa.

### Parte 1 — Retrieval Puro (Nivel 1)

Introduce el problema, la arquitectura base (Supabase, pgvector, hybrid search), la estrategia de indexing (AST-aware chunking, proposiciones atómicas), y la implementación del MCP server como context provider al estilo Context7. Muestra los primeros resultados: calidad de retrieval, consumo de tokens, y cómo impacta la calidad del índice en las respuestas del LLM.

### Parte 2 — Retrieval + Synthesis (Nivel 2)

Agrega generación server-side via API. Muestra cómo la síntesis reduce contexto y ruido, compara contra el Nivel 1 en métricas de calidad y eficiencia. Discute las decisiones de prompt engineering para el context builder y los tradeoffs de costo vs calidad.

### Parte 3 — Agentic Retrieval (Nivel 3)

Implementa el modelo agentic con tool use. Muestra cómo el LLM decide dinámicamente qué buscar y en qué orden. Compara latencia, costo y calidad contra los niveles anteriores. Explora los límites del approach: cuándo el agente mejora genuinamente y cuándo agrega complejidad innecesaria.

### Parte 4 — Resultados y Producto Final

Análisis comparativo profundo de los tres niveles con todas las métricas recopiladas. Discute las conclusiones del experimento: cuándo conviene cada approach, cuánto impacta el indexing vs la arquitectura, y lecciones aprendidas. Presenta un nuevo repositorio con una **librería lista para usar** en proyectos internos, incorporando las mejores decisiones de diseño validadas durante la experimentación.

---

# 11. Arquitectura General del Sistema

```
CloudCode / MCP Client
    │
    ▼
Internal Knowledge MCP Server (Python + FastMCP)
    │
    ├── Level 1: Retrieval puro
    │       └── Devuelve chunks al cliente
    │
    ├── Level 2: Retrieval + Synthesis
    │       └── API call → respuesta sintetizada
    │
    └── Level 3: Agentic Retrieval
            └── API call con tool use → búsqueda iterativa → respuesta
    │
    ▼
Supabase (PostgreSQL + pgvector)
    │
    ├── code_chunks
    │     ├── Full-Text Search
    │     └── Vector Search (Voyage Code 3, dim=1024)
    │
    └── doc_chunks
          ├── Full-Text Search
          └── Vector Search (Voyage 3.5, dim=1024)
              ├── Secciones de documentación
              └── Proposiciones atómicas (tarjetas)

Indexing Pipeline
    │
    ├── Código → tree-sitter → AST → chunks semánticos → Voyage Code 3 → code_chunks
    │
    ├── Markdown → mistune/markdown-it-py → secciones → Voyage 3.5 → doc_chunks
    │
    └── Markdown → LLM (Haiku/4o-mini) → proposiciones atómicas → Voyage 3.5 → doc_chunks
```

---

# 12. Pricing de Embeddings

Ambos modelos son de Voyage AI (ahora parte de MongoDB) y se acceden con la misma API y API key.

| Modelo | Uso | Precio / 1M tokens | Tokens gratis | Dimensiones |
| ------ | --- | ------------------- | ------------- | ----------- |
| `voyage-code-3` | Chunks de código | $0.18 | 200M | 1024 |
| `voyage-3.5` | Docs + proposiciones | $0.06 | 200M | 1024 |

### Estimación de costo para indexing inicial

Asumiendo ~50MB de texto total entre código y documentación (~15-20M tokens):

| Índice | Tokens estimados | Modelo | Costo |
| ------ | ---------------- | ------ | ----- |
| Código | ~10M | voyage-code-3 | $1.80 |
| Docs + proposiciones | ~10M | voyage-3.5 | $0.60 |
| **Total** | **~20M** | | **~$2.40** |

Este volumen entra dentro de los 200M tokens gratis por modelo, por lo que **la fase experimental no tendrá costo de embeddings**.

Para re-indexing incremental (cuando cambia documentación), el costo es marginal ya que solo se re-procesan las secciones cuyo hash cambió.

Voyage AI también ofrece un **Batch API** con 33% de descuento para procesamiento en lotes con ventana de 12 horas, útil para re-indexing completos.

---

# 13. Stack Tecnológico

| Componente             | Tecnología                          |
| ---------------------- | ----------------------------------- |
| MCP Server             | Python + FastMCP                    |
| Base de datos          | Supabase (PostgreSQL + pgvector)    |
| Embeddings de código   | Voyage Code 3 (API, dim=1024, $0.18/1M tokens) |
| Embeddings de docs     | Voyage 3.5 (API, dim=1024, $0.06/1M tokens) |
| Parser de código       | tree-sitter                         |
| Parser de markdown     | mistune o markdown-it-py            |
| Extracción de tarjetas | Claude Haiku o GPT-4o-mini          |
| API de generación      | Anthropic API o OpenAI API          |
| Retrieval fusion       | RRF (Reciprocal Rank Fusion)        |
