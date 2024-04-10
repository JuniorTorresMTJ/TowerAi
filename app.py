import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import locale

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings )
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    
    st.set_page_config(page_title="TowerAi",
                       page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    


# Detecta o idioma preferido do sistema
    idioma_preferido = locale.getdefaultlocale()[0]

# Determina o idioma inicial para o radio com base no idioma do sistema
    idioma_inicial = "EN" if idioma_preferido.startswith('en') else "PT-BR"

# Opções de idioma para o usuário escolher
    opcoes_de_idioma = ['EN', 'PT-BR']
    

    
    with st.sidebar:
        
        st.subheader("🌐 :violet[*Idiomas*]")
        idioma_selecionado = st.radio(
        "Escolha o idioma do TowerAi:",
        options=opcoes_de_idioma,
        index=opcoes_de_idioma.index(idioma_inicial)
        )
        st.divider()
        if idioma_selecionado == 'EN':
            st.subheader("📝 :violet[*Documents*]")
            pdf_docs = st.file_uploader(
                    "Upload your PDFs here and click on  :violet['Process'].", accept_multiple_files=True, type="pdf")
            if st.button("Process", type="primary"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore, vectors = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
            st.divider()
            st.subheader("💡 :violet[*Guide*]")
            st.text("1️⃣ Do this")
            st.text("2️⃣ Do this")
            st.text("3️⃣ Do this")
            st.text("4️⃣ Do this")

            
        else:
            st.subheader("📝 :violet[*Documentos*]")
            pdf_docs = st.file_uploader(
                    "Carregue seus PDFs aqui e clique em  :violet['Processar'].", accept_multiple_files=True, type="pdf")
            if st.button("Processar", type="primary"):
                with st.spinner("Processando..."):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                st.success('This is a success message!', icon="✅")
            else: 
                st.warning('This is a warning', icon="⚠️")
            st.divider()
            st.subheader("💡 :violet[*Guia*]")
            st.text("1️⃣ Escolha o Idioma")
            st.text("2️⃣ Carregue o(s) PDF(s)")
            st.text("3️⃣ Clique em Processar")
            st.text("4️⃣ Faça Perguntas ao TowerAi")
            
    if idioma_selecionado == 'EN':        
        st.header("Welcome to :violet[*TowerAi*] 🤖")
        st.write("Your company's chat to ask questions about private documents! 📝")
        user_question = st.text_input("Ask questions about the documents you uploaded:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.header("Bem Vindo ao :violet[*TowerAi*] 🤖")
        st.write("O chat da sua empresa para tirar dúvidas sobre documentos particulares! 📝")
        user_question = st.text_input("Faça perguntas sobre os documentos que você carregou:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()