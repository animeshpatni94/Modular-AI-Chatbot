from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
model = config.get('OLLAMA', 'model')
base_url = config.get('OLLAMA', 'base_url')

chat_llm = ChatOllama(model=model, base_url=base_url)

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
        {"role": "user", "content": "Hello! Tell me a joke about computers."}
    ]
    
    print("Testing with langchain_ollama:")
    print("-" * 60)
    
    print("Testing STREAMING response:")
    print("-" * 60)
    try:
        full_response = ""
        print("AI: ", end="", flush=True)
        

        for chunk in stream_chat_response(test_messages):
            print(chunk, end="", flush=True) 
            full_response += chunk
        
        print("\n" + "-" * 60)
        print(f"Full response received ({len(full_response)} characters)")
    except Exception as e:
        print(f"\nStreaming error: {e}")
    

    print("\nTesting COMPLETE response:")
    print("-" * 60)
    try:
        response = get_complete_response(test_messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 60)
