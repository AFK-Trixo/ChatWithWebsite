#packages to install
#pip install streamlit langchain langchain-openai beautifulsoup 4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader #requered beautifulsoup lib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#variables in the .env file will be available with this function
load_dotenv()

#simulate a basic response


#gets the text from the website and stores it in documents
def get_vectorstore_fromurl(url):
    #getting the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    #split the document into chunks
    text_splitter=RecursiveCharacterTextSplitter()
    document_chunks =  text_splitter.split_documents(document)
    
    #create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store
    
    
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up order to get information relevant to the conversation")
        
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"), #this is the prompt for the AI to function
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    #
    stuff_documents_chain=create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
def get_response(user_input):
     #create a conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain=get_conversational_rag_chain(retriever_chain)
    
    response=conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history, 
            "input": user_query
    })
    
    return response['answer']
    
    
    
#configuration of the app
st.set_page_config(page_title="Chat with Website", page_icon="a")
st.title("Chat with Websites")



#implementing the sidebar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")
    
#if the user doesn't give any website url, then sends to prompt msg for the user to add one 
if website_url is None or website_url=="":
    st.info("Please enter a website URL")
  
else:
    #session state
    #object that does not change when you reread this program
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you")
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store=get_vectorstore_fromurl(website_url)
        
   
    
    
    #userinput
    user_query=st.chat_input("Type your message here")
    if user_query is not None and user_query != "":
        response=get_response(user_query)
       
        #append is useful for chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    
    #conversation code 
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
             with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
             with st.chat_message("Human"):
                st.write(message.content)