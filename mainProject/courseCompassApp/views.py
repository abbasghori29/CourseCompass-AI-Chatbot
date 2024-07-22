import json
from django.shortcuts import render
from django.http import JsonResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ["Google_API_KEY"]='AIzaSyASa7wfJJ9elN-R837UEwUznB4wUnYm-b4'


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=100,
)
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("C:/Users/RTC/Desktop/clone/CourseCompass-AI-Chatbot/mainProject/courseCompassApp/faiss_index", google_embeddings,allow_dangerous_deserialization=True)


def getSimilar_documents(query):
    retriever=new_db.as_retriever()
    relevant_docs=retriever.get_relevant_documents(query)
    return relevant_docs
def generate_response(request):
    
    if request.method == 'POST':
        data=json.loads(request.body)
        print(data)
        query=data['query']
    
        retriever=new_db.as_retriever()
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(llm=llm ,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
        response=chain.invoke(query)
        return JsonResponse({"response": response['result']},status=200)
    
    return JsonResponse({'message':'error'},status=500)
    
    


def home(request):
    return render(request,'index.html')
    