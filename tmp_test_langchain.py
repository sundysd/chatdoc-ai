from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
res = llm([SystemMessage(content='You are an assistant.'), HumanMessage(content='Say hi.')])
print(type(res))
print(res)
print('content attr', getattr(res, 'content', None))
print('generations', getattr(res, 'generations', None))