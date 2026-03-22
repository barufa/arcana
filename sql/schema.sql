-- Arcana: Initial schema for internal knowledge MCP server
-- Run this in the Supabase SQL Editor

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Table: source_files
-- Stores complete source files for get_file_content tool
-- =============================================================================
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

-- =============================================================================
-- Table: code_chunks
-- AST-aware code chunks with embeddings (Voyage Code 3, 1024 dims)
-- =============================================================================
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

-- =============================================================================
-- Table: doc_chunks
-- Documentation chunks and propositions with embeddings (Voyage 3.5, 1024 dims)
-- =============================================================================
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

-- =============================================================================
-- HNSW indexes for vector similarity search (cosine distance)
-- =============================================================================
CREATE INDEX code_chunks_embedding_idx ON code_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX doc_chunks_embedding_idx ON doc_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Full-text search: generated tsvector columns + GIN indexes
-- =============================================================================

-- FTS for code: simple dictionary (no stemming for Python symbols)
-- replace underscores so snake_case matches (get_user_by_id -> "get user by id")
ALTER TABLE code_chunks ADD COLUMN fts TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('simple',
            COALESCE(replace(symbol_name, '_', ' '), '') || ' ' ||
            COALESCE(signature, '') || ' ' ||
            COALESCE(docstring, '')
        )
    ) STORED;

CREATE INDEX code_chunks_fts_idx ON code_chunks USING gin(fts);

-- FTS for docs: english dictionary for natural language stemming
ALTER TABLE doc_chunks ADD COLUMN fts TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('english',
            COALESCE(header_path, '') || ' ' ||
            COALESCE(content, '')
        )
    ) STORED;

CREATE INDEX doc_chunks_fts_idx ON doc_chunks USING gin(fts);

-- =============================================================================
-- Auxiliary indexes for common filters
-- =============================================================================
CREATE INDEX code_chunks_library_idx ON code_chunks(library, version);
CREATE INDEX doc_chunks_library_idx ON doc_chunks(library, version);
CREATE INDEX doc_chunks_content_type_idx ON doc_chunks(content_type);
CREATE INDEX code_chunks_file_id_idx ON code_chunks(file_id);
CREATE INDEX doc_chunks_file_id_idx ON doc_chunks(file_id);
