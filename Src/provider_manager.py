import configparser
import importlib
from llm_factory import LLMFactory
from embedding_factory import EmbeddingFactory

class ProviderManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.llm_factory = LLMFactory()
        self.embedding_factory = EmbeddingFactory()
        self.llm_factory.register_provider('ollama', 'ollama_layer')
        self.llm_factory.register_provider('azureai', 'azureai_layer')
        self.embedding_factory.register_provider('ollama', 'ollama_embedding_layer')
        self.embedding_factory.register_provider('azureai', 'azureai_embedding_layer')
        
        self.llm_provider = self._create_provider(
            factory=self.llm_factory,
            config_key='provider',
            fallback='ollama'
        )
        self.embedding_provider = self._create_provider(
            factory=self.embedding_factory,
            config_key='embedding_provider',
            fallback='ollama'
        )

    def _create_provider(self, factory, config_key, fallback):
        provider_name = self.config.get(
            'DEFAULT', 
            config_key, 
            fallback=fallback
        ).lower()
        
        provider_module_name = factory.get_provider(provider_name)
        provider_module = importlib.import_module(provider_module_name)
        provider_class = getattr(provider_module, 'Provider')
        
        setattr(self, f"{config_key}_name", provider_name)
        
        return provider_class()

    def switch_provider(self, new_provider, provider_type = 'llm'):
        factory = self.llm_factory if provider_type == 'llm' else self.embedding_factory
        config_key = 'provider' if provider_type == 'llm' else 'embedding_provider'
        
        new_instance = self._create_provider(
            factory=factory,
            config_key=config_key,
            fallback=new_provider
        )
        
        self.config.set('DEFAULT', config_key, new_provider)
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        
        if provider_type == 'llm':
            self.llm_provider = new_instance
            self.llm_provider_name = new_provider
        else:
            self.embedding_provider = new_instance
            self.embedding_provider_name = new_provider
        
        return new_provider
