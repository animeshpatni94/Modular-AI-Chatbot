from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
azure_deployment = config.get('AZUREAI', 'azure_deployment')
api_version = config.get('AZUREAI', 'api_version')
azure_endpoint = config.get('AZUREAI', 'azure_endpoint')
openai_api_key = config.get('AZUREAI', 'openai_api_key')

chat_llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,         
    api_version=api_version,        
    azure_endpoint=azure_endpoint,
    openai_api_key=openai_api_key
)

def stream_chat_response(messages):
    try:
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        for chunk in chat_llm.stream(langchain_messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
    except Exception as e:
        yield f"Error: {str(e)}"

def get_complete_response(messages):
    try:
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        response = chat_llm.invoke(langchain_messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Test block
if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Hello! Tell me a joke."}
    ]
    
    print("Testing with AzureChatOpenAI:")
    print("-" * 40)
    
    try:
        # Test streaming
        print("Streaming response:")
        full_response = ""
        for chunk in stream_chat_response(test_messages):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n" + "-" * 40)
        
        # Test complete response
        print("Complete response:")
        complete_response = get_complete_response(test_messages)
        print(complete_response)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")