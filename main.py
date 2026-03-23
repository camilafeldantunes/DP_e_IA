from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") ##criando a conexão com a chave da api

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

embeddings = OpenAIEmbeddings()

arquivos = [
    "documentos/Política_Benefícios.pdf",
    "documentos/Política_Ponto.pdf",
    "documentos/Regras_Férias.pdf"
] ##trazendo os arquivos em pdfs

documentos = []
for arquivo in arquivos:
    docs = PyPDFLoader(arquivo).load()
    nome_doc = arquivo.split("/")[-1].replace(".pdf", "")
    for doc in docs:
        doc.metadata["fonte"] = nome_doc
    documentos.extend(docs)
## Aqui juntamos todos os pdfs em uma lista, os arquivos entram no for e viram uma 
##lista de páginas. Também em cada página colocamos um metadado que diz de qual fonte
## pdf ele veio. E por fim extendemos toda essa lista para documentos

pedacos = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 100
).split_documents(documentos) ##nesse trecho quebramos aquela lista de documentos em pedaços menores
##onde cada pedaço tem 1000 caracteres e ele tem uma redundância de 100, para não perder o contexto da frase
    
dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever() ##nessa parte utilizamos o FAISS para transformar os pedaços em índices vetoriais utilizando embeddings e 
## depois o as_retriver() realiza uma pesquisa e junta os 4 vetores que mais são semelhantes



prompt_DP = ChatPromptTemplate.from_messages(
    [
        ("system", """Você é um especialista em Departamento Pessoal extremamente educado e responsável. 
         Responda utilizando apenas o contexto fornecido.
         Se a resposta estiver claramente presente no contexto, responda de forma direta e objetiva.
         Se a resposta não estiver explícita, mas puder ser deduzida com base nas informações disponíveis, responda de forma cuidadosa, deixando claro que é uma inferência.
         Caso a informação não esteja presente nem possa ser inferida com segurança, responda educadamente que não é possível responder com base no conteúdo fornecido e oriente o usuário a entrar em contato com o time de Departamento Pessoal."""),
        ("placeholder", "{historico}"),
        ("human", "{query} \n\n Contexto: {contexto} \n\n Resposta: ")
    ]
)


cadeia = prompt_DP | modelo | StrOutputParser()

memoria = {}
sessao = "dp_ia"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]
    

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia, #define a cadeia que será executada, especificando o fluxo de processamento
    get_session_history=historico_por_sessao, #recupera o histórico da conversa atual para manter o contexto
    input_messages_key="query", #define a chave que identifica a entrada do usuário na mensagem
    history_messages_key="historico" #define a chave onde o histórico da conversa será armazenado e atualizado
)

def responder_human(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    respostas = cadeia_com_memoria.invoke(
        {
            "query": pergunta,
            "contexto": contexto
        },
        config={"session_id": sessao}
    )
    fontes = trechos[0].metadata.get('fonte', 'N/A')
    return respostas, fontes



print("CHAT DP")
while True:
    perguntas_usuario = input("Faça a sua pergunta para o ChatBot ou digite 0 para finalizar \n")
    if(perguntas_usuario == '0'):
        print("Adeus fofa!!")
        break
    respostas, fontes = responder_human(perguntas_usuario)
    print("Fonte: ", fontes)
    print(respostas)

    