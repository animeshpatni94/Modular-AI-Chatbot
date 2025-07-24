import importlib
from LLMProvider.llm_factory import LLMFactory
from EmbeddingProvider.embedding_factory import EmbeddingFactory
from VectorDbProvider.vectordb_factory import VectorDBFactory
from Rerank.bge_reranker import BgeReranker
from Database.db_config_loader import load_config_from_db
from keybert import KeyBERT


class ProviderManager:
    """
    Central manager for AI-related providers:
    - LLM providers
    - Embedding providers
    - Vector database providers
    - Reranker models

    It reads configs and dynamically loads & instantiates providers using factories
    and Python's importlib.
    """

    def __init__(self):
        # Load the global configuration once
        self.config = load_config_from_db()
        self.kw_model = KeyBERT()

        # Initialize factories for LLMs and embeddings
        self.llm_factory = LLMFactory()
        self.embedding_factory = EmbeddingFactory()

        # Register known LLM providers using module paths as strings
        self.llm_factory.register_provider('ollama', 'LLMProvider.ollama_layer')
        self.llm_factory.register_provider('azureai', 'LLMProvider.azureai_layer')
        self.llm_factory.register_provider('googleai', 'LLMProvider.googleai_layer')

        # Register known embedding providers similarly
        self.embedding_factory.register_provider('ollama', 'EmbeddingProvider.ollama_embedding_layer')
        self.embedding_factory.register_provider('azureai', 'EmbeddingProvider.azureai_embedding_layer')

        # Instantiate configured LLM and embedding providers dynamically
        self.llm_provider = self._create_provider(
            factory=self.llm_factory,
            config_key='provider'
        )
        self.embedding_provider = self._create_provider(
            factory=self.embedding_factory,
            config_key='embedding_provider'
        )

        # Initialize vector DB factory with the embedding provider because some vector DBs require it
        self.vectordb_factory = VectorDBFactory(self.embedding_provider)

        # Register vector DB providers by module path strings
        self.vectordb_factory.register_provider('qdrant', 'VectorDbProvider.qdrant_vectordb_layer')
        self.vectordb_factory.register_provider('azuresearch', 'VectorDbProvider.azuresearch_vectordb_layer')

        # Instantiate configured Vector DB provider
        self.vectordb_provider = self._create_provider(
            factory=self.vectordb_factory,
            config_key='vectordb_provider'
        )

        # Load and initialize a reranker model instance (hardcoded path here)
        self.reranker = BgeReranker(model_path="cross-encoder/ms-marco-MiniLM-L12-v2")

    def _create_provider(self, factory, config_key: str):
        """
        Utility method to instantiate a provider using:
         1. The config key to find the provider name in config file
         2. Using the factory to get the module path string
         3. Dynamically import the module and get the 'Provider' class
         4. Instantiate with required arguments

        Args:
            factory: factory instance managing registration and retrieval
            config_key: configuration key to find the provider name

        Returns:
            An instance of the requested provider class
        """
        # Read the configured provider name (case-insensitive)
        provider_name = self.config['DEFAULT'][config_key].lower()

        # Use factory to get the registered module path string for this provider
        provider_module_path = factory.get_provider(provider_name)

        # Dynamically import the module by string path
        module = importlib.import_module(provider_module_path)

        # 'Provider' is the expected class name inside each provider module
        provider_class = getattr(module, 'Provider')

        # Save the provider name to an attribute for reference
        setattr(self, f"{config_key}_name", provider_name)

        # Special case: vectordb providers require the embedding model on initialization
        if config_key == "vectordb_provider":
            # The embedding_provider was already instantiated and has attribute `embedding_model`
            # pass the key word model as well
            return provider_class(self.embedding_provider.embedding_model, self.kw_model)

        # For other providers (LLM, embedding), call constructor without arguments
        return provider_class()
