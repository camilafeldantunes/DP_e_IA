# DP e IA – Chatbot Inteligente para Departamento Pessoal
## Descrição
Sistema inteligente que utiliza IA + RAG (Retrieval-Augmented Generation) para responder dúvidas relacionadas ao Departamento Pessoal, como:
* ⏰ Controle de ponto
* 💰 Benefícios (VT, VA, VR, plano de saúde, etc.)
* 🌴 Férias
A aplicação lê documentos (PDFs) e fornece respostas contextualizadas com base nas políticas da empresa.

## Funcionalidades
* Upload e leitura de PDFs
* Indexação com FAISS
* Busca semântica de informações
* Chat com contexto (memória de conversa)
* Respostas baseadas em documentos reais
* Interface interativa (ex: Streamlit)

## 🛠️ Tecnologias Utilizadas
* Python
* LangChain
* OpenAI API
* FAISS
* Streamlit
* PyPDF

## 📂 Estrutura do Projeto

```
DP_e_IA/
│
├── documentos/            # PDFs usados no RAG
├── langchain/
|── main.py                # Aplicação principal
├── requirements.txt
├── .env                  # Chave da API (não subir!)
└── README.md
```

## Como executar o projeto

1. Clone o repositório
```
git clone https://github.com/seu-usuario/seu-repo.git
```

2. Criar ambiente virtual
```
python -m venv venv 
source venv/bin/activate  # Mac/Linux 
```

3. Instalar dependências
```
pip install -r requirements.txt
```

4. Configurar variável de ambiente

```
OPENAI_API_KEY=sua_chave_aqui
```

5. Rodar o projeto
