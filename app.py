import streamlit as st
import openai
import Utils
from streamlit_extras.add_vertical_space import add_vertical_space
import pinecone
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from dotenv import load_dotenv
load_dotenv(".env")

with st.sidebar:
    st.header("Tools used to build this chatbot")
    st.markdown("""
    - Streamlit
    - HuggingFace Transformer
    - OpenAI
    - LangChain
    - PineCone
    """)
    add_vertical_space(5)

st.title("ChatBot for Documents")
document = st.file_uploader("Upload the Document", type = "pdf")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can i help you?"] 
if "requests" not in st.session_state:
    st.session_state["requests"] = []   

openai.api_key = "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"
llm = ChatOpenAI(model_name=os.getenv("model_name"), openai_api_key=openai.api_key)     

if "buffer_memory" not in st.session_state:
    st.session_state["buffer_memory"] = ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template = "{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template,MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

def main():
    if document is not None:
        text = Utils.docLoader(document)
        chunk = Utils.docsplitter(text, chunk_size=1000,chunk_overlap=100 )
        vector = Utils.docembedding(chunk)
        with textcontainer:
            query = st.text_input("Query: ", key="input")
            if query:
                with st.spinner("reading..."):
                    conversation_string = Utils.get_conversation_string()
                    refined_query = Utils.query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    context = Utils.find_match(refined_query, vector) 
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response) 
        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i],key=str(i))
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
        
if __name__=="__main__":
    main()