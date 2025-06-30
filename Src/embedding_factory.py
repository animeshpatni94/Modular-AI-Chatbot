class EmbeddingFactory:
    def __init__(self):
        self._providers = {}

    def register_provider(self, name, provider_module):
        self._providers[name] = provider_module

    def get_provider(self, name):
        provider = self._providers.get(name)
        if not provider:
            raise ValueError(f"Embedding Provider '{name}' not found")
        return provider