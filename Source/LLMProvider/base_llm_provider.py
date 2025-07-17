from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class BaseLLMProvider:
    def __init__(self, chat_llm):
        self.chat_llm = chat_llm

    def stream_chat_response(self, messages):
        try:
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            for chunk in self.chat_llm.stream(langchain_messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            yield f"Error: {str(e)}"

    def get_complete_response(self, messages):
        try:
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            response = self.chat_llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
