CREATE TABLE ConfigSections (
    id INT IDENTITY PRIMARY KEY,
    name NVARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE ConfigEntries (
    id INT IDENTITY PRIMARY KEY,
    section_id INT NOT NULL,
    config_key NVARCHAR(100) NOT NULL,
    config_value NVARCHAR(MAX),
    FOREIGN KEY(section_id) REFERENCES ConfigSections(id),
    CONSTRAINT uq_config UNIQUE (section_id, config_key)
);

CREATE TABLE LLMChatLogs (
    id INT IDENTITY PRIMARY KEY,
    session_id NVARCHAR(100) NOT NULL,
    user_id NVARCHAR(200) NULL,
    request NVARCHAR(MAX) NOT NULL,
    response NVARCHAR(MAX) NOT NULL,
    model_name NVARCHAR(100) NULL,
    provider NVARCHAR(50) NULL,
    prompt_template NVARCHAR(100) NULL,
    retrieval_docs NVARCHAR(MAX) NULL,
    timestamp DATETIME DEFAULT GETDATE()
);

CREATE TABLE VectorVocabularies (
    id INT IDENTITY(1,1) PRIMARY KEY,
    collection_name VARCHAR(128) NOT NULL UNIQUE,
    vocab_json NVARCHAR(MAX) NOT NULL,
    updated_at DATETIME DEFAULT GETDATE()
);
