from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
) ## Aqui juntamos todos os pdfs em uma lista, os arquivos entram no for e viram uma 
##lista de páginas e depois através da sum() eles se transformam em uma única lista

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
        ("system", """Você é um especialista em Departamento Pessoal extremamente educado e responsável. Responda as perguntas
         utilizando somente o conteúdo fornecido. Se você não souber a resposta
         responda assim: 'Me desculpe pelo inconveniente, mas não posso responder essa questão, por favor, entre em contato com o
         time de DP para que eles te expliquem da melhor maneira possível'"""),
        ("human", "{query} \n\n Contexto: {contexto} \n\n Resposta: ")
    ]
)


cadeia = prompt_DP | modelo | StrOutputParser()

def responder_human(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke(
        {
            "query": pergunta,
            "contexto": contexto
        }
    )

print(responder_human("Como funciona o pagamento das férias?"))