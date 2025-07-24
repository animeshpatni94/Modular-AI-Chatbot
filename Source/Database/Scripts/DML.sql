-- Insert all section names
INSERT INTO ConfigSections (name) VALUES
('DEFAULT'), ('SESSION'), ('OLLAMA'), ('AZUREAI'), ('OLLAMA_EMBED'),
('AZUREAI_EMBED'), ('QDRANT'), ('AZURESEARCH'), ('TEXT_SPLITTING'),
('GOOGLEAI'), ('SYSTEM_PROMPTS'), ('OIDC');

-- DEFAULT
DECLARE @section_id INT;
SELECT @section_id = id FROM ConfigSections WHERE name = 'DEFAULT';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'provider', 'azureai'),
(@section_id, 'embedding_provider', 'ollama'),
(@section_id, 'vectordb_provider', 'qdrant'),
(@section_id, 'folder_path', ''),
(@section_id, 'ingestion_collection_name', 'VRS'),
(@section_id, 'default_prompt', 'legal_prompt'),
(@section_id, 'search_population', '150');

-- SESSION
SELECT @section_id = id FROM ConfigSections WHERE name = 'SESSION';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'history_length', '6');

-- OLLAMA
SELECT @section_id = id FROM ConfigSections WHERE name = 'OLLAMA';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'model', 'llama3.2:latest'),
(@section_id, 'base_url', '');

-- AZUREAI
SELECT @section_id = id FROM ConfigSections WHERE name = 'AZUREAI';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'azure_deployment', 'gpt-4o'),
(@section_id, 'api_version', '2025-01-01-preview'),
(@section_id, 'azure_endpoint', ''),
(@section_id, 'openai_api_key', '');

-- OLLAMA_EMBED
SELECT @section_id = id FROM ConfigSections WHERE name = 'OLLAMA_EMBED';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'model', 'mxbai-embed-large:latest'),
(@section_id, 'base_url', '');

-- AZUREAI_EMBED
SELECT @section_id = id FROM ConfigSections WHERE name = 'AZUREAI_EMBED';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'azure_deployment', 'text-embedding-3-large'),
(@section_id, 'api_version', '2023-05-15'),
(@section_id, 'azure_endpoint', ''),
(@section_id, 'api_key', '');

-- QDRANT
SELECT @section_id = id FROM ConfigSections WHERE name = 'QDRANT';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'url', ''),
(@section_id, 'collection_name', 'legal');

-- AZURESEARCH
SELECT @section_id = id FROM ConfigSections WHERE name = 'AZURESEARCH';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'endpoint', 'https://<your-service>.search.windows.net'),
(@section_id, 'api_key', '<your-api-key>'),
(@section_id, 'index_name', 'my_vectors');

-- TEXT_SPLITTING
SELECT @section_id = id FROM ConfigSections WHERE name = 'TEXT_SPLITTING';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'chunk_size', '1000'),
(@section_id, 'overlap', '200');

-- GOOGLEAI
SELECT @section_id = id FROM ConfigSections WHERE name = 'GOOGLEAI';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'google_api_key', ''),
(@section_id, 'model_name', 'gemini-2.5-pro');

-- SYSTEM_PROMPTS
SELECT @section_id = id FROM ConfigSections WHERE name = 'SYSTEM_PROMPTS';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'vrs_prompt', '{"role": "system", "content": "You are a knowledgeable and helpful assistant specializing in the Virginia Retirement System (VRS). You must answer strictly and exclusively using ONLY the information provided in the CONTEXT section below. Do NOT use any outside knowledge, make assumptions, or add information not present in the context. CONTEXT: {context} INSTRUCTIONS: - ONLY use information from the above CONTEXT to answer the users question. - Do NOT use any external knowledge or make up information."}'),
(@section_id, 'legal_prompt', '{"role": "system", "content": "You are a knowledgeable and helpful assistant specializing in legal information and guidance. You must answer strictly and exclusively using ONLY the information provided in the CONTEXT section below. Do NOT use any outside knowledge, make assumptions, or add information not present in the context. You must not provide legal advice, but only summarize or clarify the information as presented in the context. Always maintain accuracy, neutrality, and compliance with the information provided. CONTEXT: {context} INSTRUCTIONS: - ONLY use information from the above CONTEXT to answer the users question. - Do NOT use any external knowledge or make up information. - Do NOT provide legal advice or opinions; only restate or clarify what is in the context."}'),
(@section_id, 'requirements_prompt', '{"role": "system", "content": "You are a knowledgeable and helpful technical assistant specializing in requirements analysis. You must answer strictly and exclusively using ONLY the information provided in the CONTEXT section below. Do NOT use any outside knowledge, make assumptions, or add information not present in the context. Your task is to determine whether the entered requirement meets, partially meets, or does not meet the criteria, specifications, or standards described in the context. Provide clear, accurate and neutral explanations based solely on the context provided. CONTEXT: {context} INSTRUCTIONS: - ONLY use information from the above CONTEXT to evaluate the users requirement. - Do NOT use any external knowledge or make up information. - Clearly state whether the requirement meets, partially meets, or does not meet the context criteria, and explain your reasoning using only the context."}'),
(@section_id, 'xsharp_prompt', '{"role": "system", "content": "You are a knowledgeable and helpful assistant specializing in the Xsharp programming language. You must answer strictly and exclusively using ONLY the information provided in the CONTEXT section below. Do NOT use any outside knowledge, make assumptions, or add information not present in the context. You must only provide responses in Xsharp syntax and format, without using any other programming language constructs or formats. Always maintain accuracy and compliance with the Xsharp language specifications provided in the context. CONTEXT: {context} INSTRUCTIONS: - ONLY use information from the above CONTEXT to answer the users question. - Do NOT use any external knowledge or make up information. - Provide responses exclusively in Xsharp syntax and format. - Do NOT use constructs from other programming languages unless explicitly mentioned in the context. - Focus solely on Xsharp language features, syntax, and capabilities as described in the context."}');

-- OIDC
SELECT @section_id = id FROM ConfigSections WHERE name = 'OIDC';
INSERT INTO ConfigEntries (section_id, config_key, config_value) VALUES
(@section_id, 'client_id', ''),
(@section_id, 'client_secret', ''),
(@section_id, 'discovery_url', ''),
(@section_id, 'scope', 'openid email profile'),
(@section_id, 'valid_api_keys', '');
