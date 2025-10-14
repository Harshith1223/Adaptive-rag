from langchain_core.output_parsers import StrOutputParser
from src.models.model import get_llm_model
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm_model()

system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.
Conversation History: {history}
Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()