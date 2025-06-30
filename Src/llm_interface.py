class LLMInterface:
    def __init__(self, factory, provider_name):
        self.factory = factory
        self.provider_name = provider_name
        self.provider = factory.get_provider(provider_name)

    def stream_chat_response(self, messages):
        return self.provider.stream_chat_response(messages)

    def get_complete_response(self, messages):
        return self.provider.get_complete_response(messages)

    def switch_provider(self, new_provider_name):
        self.provider_name = new_provider_name
        self.provider = self.factory.get_provider(new_provider_name)
