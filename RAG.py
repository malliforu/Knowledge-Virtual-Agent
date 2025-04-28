from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores import FAISS, Pinecone, Chroma
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv('api_key')
endpoint = os.getenv('endpoint')
api_version = os.getenv('api_version')
deployment = os.getenv('deployment')

azure_embedding_model = os.getenv('azure_embedding_model')

PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = os.getenv('INDEX_NAME')


class RAG:
    def __init__(self, path, 
                 chunking_type='fixed_text_split', 
                 chunk_size=1000, 
                 vectorstore_name='faiss', 
                 index_name= 'default', 
                 llm_model = None, 
                 embedding_model=None,
                 embedding_dim = 1536):
        self.path = path
        self.chunking_type = chunking_type
        self.chunk_size = chunk_size
        self.index_name = index_name
        self.vectorstore_name = vectorstore_name
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def load_documents(self, path):
        all_documents = []
        if 'http' in path: 
            web_loader = WebBaseLoader(web_path=path)
            documents = web_loader.load()
            all_documents.extend(documents)
        else:
            files = os.listdir(path)
            for file in files:
                if os.path.splitext[-1] == '.pdf':
                    pdf_loader = PyPDFLoader(file)
                    documents = pdf_loader.load()
                elif os.path.splitext[-1] == '.txt':
                    txt_loader = TextLoader(file)
                    documents = txt_loader.load()
                else:
                    print(f'File name: {file} is of unsupported type')
                all_documents.extend(documents)

        return all_documents

    def load_chunks(self, documents, chunking_type, chunk_size):
        chunk_overlap = int(chunk_size/5)
        if documents:
            if chunking_type =='fixed_text_split':
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap)
            elif chunking_type == 'recursive_text_split':
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif chunking_type == 'token_text_split':
                text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif chunking_type == 'sentence_transformer_text_split':
                text_splitter = SentenceTransformersTokenTextSplitter(model_name='all-MiniLM-L6-v2',chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            split_chunks = text_splitter.split_documents(documents)
            return split_chunks
        else:
            print("No documents found!")

    
    def vectorstore(self, documents, index_name = 'default'):
        index_dir = os.path.join("./indexes", self.vectorstore_name)
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        index_path = os.path.join(index_dir, index_name)

        if self.vectorstore_name == 'faiss':
            if os.path.exists(index_path) and os.listdir(index_path):
                vectorstore = FAISS.load_local(index_path, self.embedding_model, allow_dangerous_deserialization=True)
                if documents:
                    vectorstore.add_documents(documents)
                    vectorstore.save_local(index_path)
                    return vectorstore
            if documents:
                vectorstore = FAISS.from_documents(documents, self.embedding_model)
                vectorstore.save_local(index_path)
                return vectorstore

        elif self.vectorstore_name == 'pinecone':
            pc = PineconeClient(api_key = PINECONE_API_KEY, environment = PINECONE_ENVIRONMENT)
            indexes_names  = pc.list_indexes().names()
            if index_name in indexes_names:
                vectorstore = Pinecone.from_existing_index(index_name,
                                                            self.embedding_model,
                                                            client = pc)
                if documents:
                    vectorstore.add_documents(documents)
                    return vectorstore
            if documents:
                if index_name not in indexes_names:
                    pc.create_index(name=index_name,
                                          dimension=self.embedding_dim,
                                          metric='cosine',
                                          spec=ServerlessSpec(cloud='aws', region='us-east-1')
                                            )
                    vectorstore = Pinecone.from_existing_index(index_name,
                                                               self.embedding_model,
                                                               client = pc)
                    vectorstore.add_documents(documents)
                    return vectorstore
                
        elif self.vectorstore_name =='chroma':
            if os.path.exists(index_path) and os.listdir(index_path):
                vectorstore = Chroma(
                    persist_directory=index_path,
                    embedding_function=self.embedding_model,
                    collection_name=index_name
                )
                if documents:
                    vectorstore.add_documents(documents)
                    vectorstore.persist()
                    return vectorstore
            if documents:
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=index_path,
                    collection_name=index_name
                )
                vectorstore.persist()
                return vectorstore

    def ask_question(self, question):
        documents = self.load_documents(self.path)
        chunks = self.load_chunks(documents, chunking_type='fixed_text_split', chunk_size=self.chunk_size)
        vectorstore = self.vectorstore(documents = chunks)

        prompt ="""
You are an AI assistant, consider the context given below and answer the user question

Context:
{context}

Question: {question}

Answer:

"""

        PROMPT = PromptTemplate(template= prompt,
                                input_variables=['context', 'question'])
        
        qa_chain = RetrievalQA.from_chain_type(llm = self.llm_model,
                                               chain_type = 'stuff',
                                               retriever = vectorstore.as_retriever(k = 5),
                                               chain_type_kwargs = {'prompt': PROMPT})
        
        # top5=vectorstore.similarity_search(query=question, k=5)
        # top5_scores = vectorstore.similarity_search_with_relevance_scores(query=question,k=5)
        # top1 = vectorstore.similarity_search(query=question, k =1)
        # return top5, top5_scores, top1

        return qa_chain.invoke({'query':question})


llm_model = AzureChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=endpoint,
    openai_api_version=api_version,
    deployment_name=deployment,
    temperature=0.0)

embedding_model = AzureOpenAIEmbeddings(api_key= api_key,
                        api_version= api_version,
                        azure_endpoint= endpoint,
                        azure_deployment=azure_embedding_model)

rag = RAG(path='https://en.wikipedia.org/wiki/Bengaluru',
                chunking_type='fixed_text_split', 
                 chunk_size=1000, 
                 vectorstore_name='chroma', 
                 index_name= 'default', 
           llm_model=llm_model, embedding_model=embedding_model)

rag.ask_question("What is Bengaluru?")