import ollama_layer
import azureai_layer
from llm_factory import LLMFactory
from llm_interface import LLMInterface

# Register providers
factory = LLMFactory()
factory.register_provider('ollama', ollama_layer)
factory.register_provider('azure', azureai_layer)

# Create interface with default provider
llm = LLMInterface(factory, 'ollama')

# Example usage
test_messages = [{"role": "user", "content": "Hello! Tell me a joke."}]

# Streaming response
print("Streaming from Ollama:")
for chunk in llm.stream_chat_response(test_messages):
    print(chunk, end="", flush=True)
print("\n")

# Switch to Azure
llm.switch_provider('azure')
print("Streaming from Azure:")
for chunk in llm.stream_chat_response(test_messages):
    print(chunk, end="", flush=True)
print("\n")

# Complete response
print("Complete response from Azure:")
print(llm.get_complete_response(test_messages))
