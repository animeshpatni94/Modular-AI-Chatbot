class VectorDBFactory:
    def __init__(self, embedding_provider):
        self._providers = {}
        self.embedding_provider = embedding_provider

    def register_provider(self, name, provider_module):
        self._providers[name.lower()] = provider_module

    def get_provider(self, name):
        provider = self._providers.get(name.lower())
        if not provider:
            raise ValueError(f"VectorDB provider '{name}' not found")
        return provider