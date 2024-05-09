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
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatMaritalk
from ragas import evaluate
from ragas.metrics import  faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, answer_correctness
from datasets import Dataset 
import locale
#import evaluate
import pandas as pd



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
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    #gpt-3.5-turbo-0125
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings )
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI(temperature=0)
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125' , temperature=0)
    # llm = HuggingFaceHub(repo_id="NexaAIDev/Octopus-v2", model_kwargs={"temperature":0.0, "max_new_tokens":250})
#     llm = ChatMaritalk(
#     model="sabia-2-medium",  # Available models: sabia-2-small and sabia-2-medium
#     api_key="106360418574866720591$a22683aa2529c86b",  # Insert your API key here
#     temperature=0.1,
#     max_tokens=300,
# )

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
            
    data_samples = {
    'question': [
        "Qual é o documento que regula a situação, obrigações, deveres, direitos e prerrogativas dos membros das Forças Armadas do Brasil?",
        "Quem tem a autoridade suprema sobre as Forças Armadas brasileiras, segundo o Estatuto dos Militares?",
        "O que caracteriza a carreira militar, conforme definido no Estatuto dos Militares?"
    ],
    'answer': [
        "O documento que regula a situação, obrigações, deveres, direitos e prerrogativas dos membros das Forças Armadas do Brasil é a Lei no 6.880, de 9 de dezembro de 1980, conhecida como o Estatuto dos Militares. Este estatuto estabelece as bases legais sobre as quais se apoiam os direitos, os deveres e as obrigações dos militares em serviço ativo, na reserva ou reformados.",
        "De acordo com o Estatuto dos Militares, a autoridade suprema sobre as Forças Armadas brasileiras é o Presidente da República. Isso está estabelecido no Art. 3°, que afirma que as Forças Armadas são instituições nacionais, permanentes e regulares, organizadas com base na hierarquia e na disciplina, sob a autoridade suprema do Presidente da República e dentro dos limites da lei.",
        "De acordo com o Estatuto dos Militares, a carreira militar é caracterizada por atividade continuada e inteiramente devotada às finalidades precípuas das Forças Armadas, denominada atividade militar. A carreira militar é privativa do pessoal da ativa, inicia-se com o ingresso nas Forças Armadas e obedece às diversas sequências de graus hierárquicos. Além disso, são privativas de brasileiro nato as carreiras de oficial da Marinha, do Exército e da Aeronáutica."
    ],
    'contexts': [
        ["O Estatuto dos Militares é estabelecido pela Lei Nº 6.880, de 9 de Dezembro de 1980. Este documento é fundamental para a organização e o funcionamento das Forças Armadas brasileiras, constituídas pela Marinha, pelo Exército e pela Aeronáutica."],
        ["De acordo com o Estatuto dos Militares, as Forças Armadas do Brasil operam sob a autoridade suprema do Presidente da República, seguindo os princípios da hierarquia e da disciplina, com a missão de defender a Pátria e garantir a lei e a ordem."],
        ["Segundo o Estatuto dos Militares, a carreira é privativa do pessoal da ativa das Forças Armadas e inicia-se com o ingresso na Marinha, no Exército ou na Aeronáutica, obedecendo às diversas sequências de graus hierárquicos."]
    ],
    'ground_truth': [
    "O presente Estatuto regula a situação, obrigações, deveres, direitos e prerrogativas dos membros das Forças Armadas.",
    "As Forças Armadas, sob a autoridade suprema do Presidente da República, destinam-se a defender a Pátria e garantir os poderes constituídos, a lei e a ordem.",
    "A carreira militar é caracterizada por atividade continuada e inteiramente devotada às finalidades precípuas das Forças Armadas, denominada atividade militar."
    ]
    }
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset,metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, answer_correctness])
    score.to_pandas()
    st.table(data=score)

    # predictions = ["In Dungeons & Dragons, the metallic dragons come in brass, bronze, copper, gold, and silver varieties. Each has scales in hues matching their name - brass dragons have brassy scales, bronze have bronze scales, etc. The metallic dragons are generally more benign than the chromatic dragons in D&D lore."]


    # references =["""The five basic chromatic dragons (red, blue, green, black, and white) and metallic dragons (copper, brass, silver, gold, and bronze) appeared in the fifth edition Monster Manual (2014) in wyrmling, young, adult, and ancient. Gem dragons and other new-to-fifth-edition dragons appeared in Fizban's Treasury of Dragons (2021)"""]
    
    # rouge = evaluate.load('rouge')
    # results_rouge = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
    # df_rouge = pd.DataFrame.from_dict(results_rouge)
    # st.table(df_rouge)
    # bleu = evaluate.load('bleu')
    
    # results_bleu = bleu.compute(predictions=predictions, references=references,max_order=4)
    # df_results_bleu = pd.DataFrame.from_dict(results_bleu)
    # st.table(df_results_bleu)
#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
#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
    
if __name__ == '__main__':
    main()
    
