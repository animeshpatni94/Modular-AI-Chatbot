import importlib
from LLMProvider.llm_factory import LLMFactory
from EmbeddingProvider.embedding_factory import EmbeddingFactory
from VectorDbProvider.vectordb_factory import VectorDBFactory
from Rerank.bge_reranker import BgeReranker
from Database.db_config_loader import load_config_from_db

class ProviderManager:
    def __init__(self):
        self.config = load_config_from_db()
        self.llm_factory = LLMFactory()
        self.embedding_factory = EmbeddingFactory()
        self.reranker = BgeReranker(model_path="cross-encoder/ms-marco-MiniLM-L12-v2")
        self.llm_factory.register_provider('ollama', 'LLMProvider.ollama_layer')
        self.llm_factory.register_provider('azureai', 'LLMProvider.azureai_layer')
        self.llm_factory.register_provider('googleai', 'LLMProvider.googleai_layer')
        self.embedding_factory.register_provider('ollama', 'EmbeddingProvider.ollama_embedding_layer')
        self.embedding_factory.register_provider('azureai', 'EmbeddingProvider.azureai_embedding_layer')
        
        self.llm_provider = self._create_provider(
            factory=self.llm_factory,
            config_key='provider'
        )
        self.embedding_provider = self._create_provider(
            factory=self.embedding_factory,
            config_key='embedding_provider'
        )

        self.vectordb_factory = VectorDBFactory(self.embedding_provider)
        self.vectordb_factory.register_provider('qdrant', 'VectorDbProvider.qdrant_vectordb_layer')
        self.vectordb_factory.register_provider('azuresearch', 'VectorDbProvider.azuresearch_vectordb_layer')
        
        self.vectordb_provider = self._create_provider(
            config_key='vectordb_provider',
            factory=self.vectordb_factory
        )
        
    def _create_provider(self, factory, config_key):
        provider_name = self.config['DEFAULT'][config_key].lower()
        provider_module_name = factory.get_provider(provider_name)
        provider_module = importlib.import_module(provider_module_name)
        provider_class = getattr(provider_module, 'Provider')
        setattr(self, f"{config_key}_name", provider_name)
        if (config_key == "vectordb_provider"):
            return provider_class(factory.embedding_provider.embedding_model)
        return provider_class()

    def switch_provider(self, new_provider, provider_type = 'llm'):
        if provider_type == 'llm':
            self.llm_provider = self._create_provider(
                factory=self.llm_factory,
                config_key='provider'
            )
            self.llm_provider_name = new_provider
            self.config.set('DEFAULT', 'provider', new_provider)
        elif provider_type == 'embedding':
            self.embedding_provider = self._create_provider(
                factory=self.embedding_factory,
                config_key='embedding_provider'
            )
            self.embedding_provider_name = new_provider
            self.config.set('DEFAULT', 'embedding_provider', new_provider)
            self.vectordb_factory = VectorDBFactory(self.embedding_provider)
            self.vectordb_factory.register_provider('qdrant', 'qdrant_vectordb_layer')
            self.vectordb_factory.register_provider('azuresearch', 'azuresearch_vectordb_layer')
            self.vectordb_provider = self._create_provider(
                config_key='vectordb_provider',
                factory=self.vectordb_factory
            )
        elif provider_type == 'vectordb':
            self.vectordb_provider = self.vectordb_factory.get_provider(new_provider)
            self.vectordb_provider_name = new_provider
            self.config.set('DEFAULT', 'vectordb_provider', new_provider)
        else:
            raise ValueError(f"Invalid provider type: {provider_type}")

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        
        return new_provider
