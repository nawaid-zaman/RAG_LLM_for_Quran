#%%
import streamlit as st
import getpass
import os
import pandas as pd


API_KEY = "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"
os.environ["OPENAI_API_KEY"] = API_KEY

# splitter_type = 'recursive'
# store_type = 'chroma'
# retriever_type = 'similarity'

# demo = True
demo = False
max_context_column = 20


##%%
parameter_list = [
('recursive', 'chroma', 'similarity'),
('recursive', 'chroma', 'mmr_lambda_high'),
('recursive', 'chroma', 'mmr_lambda_low'),
('recursive', 'chroma', 'similarity_score_threshold'),

('recursive', 'faiss', 'similarity'),
('recursive', 'faiss', 'mmr_lambda_high'),
('recursive', 'faiss', 'mmr_lambda_low'),
('recursive', 'faiss', 'similarity_score_threshold'),

('character', 'chroma', 'similarity'),
('character', 'chroma', 'mmr_lambda_high'),
('character', 'chroma', 'mmr_lambda_low'),
('character', 'chroma', 'similarity_score_threshold'),

('character', 'faiss', 'similarity'),
('character', 'faiss', 'mmr_lambda_high'),
('character', 'faiss', 'mmr_lambda_low'),
('character', 'faiss', 'similarity_score_threshold'),

('sentence', 'chroma', 'similarity'),
('sentence', 'chroma', 'mmr_lambda_high'),
('sentence', 'chroma', 'mmr_lambda_low'),
('sentence', 'chroma', 'similarity_score_threshold'),

('sentence', 'faiss', 'similarity'),
('sentence', 'faiss', 'mmr_lambda_high'),
('sentence', 'faiss', 'mmr_lambda_low'),
('sentence', 'faiss', 'similarity_score_threshold')]

splitter_type, store_type, retriever_type = parameter_list[1]
print(splitter_type, store_type, retriever_type)

# #%%
# splitter_type = 'recursive'
# splitter_type = 'character'
# splitter_type = 'sentence'

# store_type = 'chroma'
# store_type = 'faiss'

# retriever_type = 'similarity'
# retriever_type = 'mmr_lambda_high'
# retriever_type = 'mmr_lambda_low'
# retriever_type = 'similarity_score_threshold'

##%% ############### UPDATING CHAT HISTORY WITH BASE MESSAGE HISTORY ##################

import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader 
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss


def extract_context(row):
    for x in range(0, len( row['ai_context'] ) ):
        row[f"ai_context_{x+1}"] = row['ai_context'][x].page_content
    return row

def extract_ai_context_len(row):
    row['ai_context_len'] = len(row['ai_context'] ) 
    return row


##%%
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

path_list = [
    "Source_data\quran_saheeh_international.txt",
    "Source_data\quran_mohsin_khan.txt"
]


##%%

# Extract text content from txt file
full_content = ""

for path in path_list:
        
    # Open and read the text file content
    with open(path, 'r', encoding='utf-8') as file:
        text_content = file.read()
        full_content += text_content
        full_content += '\n\n'

if demo == True:
    full_content = full_content[1500:3000]
    print('Demo content applied')

##%%
# Create a single Document object (like in LangChain)
txt_docs = [Document(page_content=full_content)]

docs = txt_docs
# docs = pdf_docs
# docs = docs + pdf_docs

##%% ######################## TEXT SPLITTER ######################## 
    
# splitter_type = 'recursive'
# splitter_type = 'character'
# splitter_type = 'sentence'

if splitter_type == 'recursive':
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200)
    print("recursive splitter")

# if splitter_type == 'recursive_doc':
#     splits = text_splitter.create_documents([text_content])

if splitter_type == 'character':
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,)
    print("character splitter")

# if splitter_type == 'semantic':
#     from langchain_experimental.text_splitter import SemanticChunker
#     from langchain_openai.embeddings import OpenAIEmbeddings

#     text_splitter = SemanticChunker(OpenAIEmbeddings())

if splitter_type == 'sentence':
    from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
    text_splitter = SentenceTransformersTokenTextSplitter()
    print("sentence splitter")


splits = text_splitter.split_documents(docs)
# splits

#%% ######################## VECTOR STORE ######################## 
# store_type = 'chroma'
# store_type = 'faiss'

if store_type == 'chroma': 
    # Create chroma vectorstore 

    if splitter_type == 'recursive':
        vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(),
                                        collection_name= 'quran_vector',
                                        persist_directory='chroma\\recursive_splitter'
                                        )
        print('Accessing chroma-recursive_splitter vectorstore')

    if splitter_type == 'character':
        vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(),
                                        collection_name= 'quran_vector',
                                        persist_directory='chroma\\character_splitter'
                                        )
        print('Accessing chroma-character_splitter vectorstore')

    if splitter_type == 'sentence':
        vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(),
                                        collection_name= 'quran_vector',
                                        persist_directory='chroma\\sentence_splitter'
                                        )
        print('Accessing chroma-sentence_splitter vectorstore')

#%%
if store_type == 'faiss': 
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    embeddings = OpenAIEmbeddings()
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello")))

    if splitter_type == 'recursive':
        vector_path = 'faiss\\recursive'

        try:
            # Load the vector store from disk
            vectorstore = FAISS.load_local(
                folder_path=vector_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True)
            print(f'Loaded Faiss vector store from {vector_path}')

        except:
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={})
            vectorstore.add_documents(documents=splits)

            # Save to disk
            vectorstore.save_local(vector_path)
            print(f'Saved Faiss vector store from {vector_path}')


    if splitter_type == 'character':
        vector_path = 'faiss\\character'

        try:
            # Load the vector store from disk
            vectorstore = FAISS.load_local(
                folder_path=vector_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True)
            print(f'Loaded Faiss vector store from {vector_path}')
            
        except:
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={})
            vectorstore.add_documents(documents=splits)

            # Save to disk
            vectorstore.save_local(vector_path)
            print(f'Saved Faiss vector store from {vector_path}')


    if splitter_type == 'sentence':
        vector_path = 'faiss\\sentence'

        try:
            # Load the vector store from disk
            vectorstore = FAISS.load_local(
                folder_path=vector_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True)
            print(f'Loaded Faiss vector store from {vector_path}')
        
        except:
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={})
            vectorstore.add_documents(documents=splits)
            print(f'Saved Faiss vector store from {vector_path}')

            # Save to disk
            vectorstore.save_local(vector_path)

 
# #%%
# # retriever_type = 'similarity'
# retriever_type = 'mmr_lambda_high'
# # retriever_type = 'mmr_lambda_low'
# # retriever_type = 'similarity_score_threshold'



# if retriever_type == 'similarity':
#     retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 8, "fetch_k": 20})
#     print("retriever_type == similarity")


# if retriever_type == 'mmr_lambda_high':
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.8})
#     print("retriever_type == mmr_lambda_high")

# if retriever_type == 'mmr_lambda_low':
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.2})
#     print("retriever_type == mmr_lambda_low")

# if retriever_type == 'similarity_score_threshold':
#     retriever = vectorstore.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={'score_threshold': 0.8})
#     print("retriever_type == similarity_score_threshold")

# #%% ######################## RETRIEVER ######################## 

# # if retriever_type == 'similarity'
# # Save retriever in session state
# # retriever = vectorstore.as_retriever()


# #%%

# ### Contextualize question ###
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# # Use three sentences maximum and keep the answer concise.\
# # Answer concisely until asked to elaborate.\

# ### Answer question ###
# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# Do not answer with your own memory. \
# Do not use out of the context knowledge to answer. \
# only use the context to answer the question. \
# if the question is not answered within the context, say I have not learnt this. \
# If you don't know the answer, just say that you don't know. \
# Elaborate the answer by making full use of context provided\
# Answer in atleast 5 sentences\

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



# #%%
# ## Statefully manage chat history ###
# store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # def get_session_history(session_id: str) -> BaseChatMessageHistory:
# #     if "chat_histories" not in st.session_state:
# #         st.session_state.chat_histories = {}

# #     if session_id not in st.session_state.chat_histories:
# #         st.session_state.chat_histories[session_id] = ChatMessageHistory()
    
# #     return st.session_state.chat_histories[session_id]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# #%%
# # ans  = conversational_rag_chain.invoke(
# #     {"input": "What is Task Decomposition?"},
# #     config={
# #         "configurable": {"session_id": "abc123"}
# #     },  # constructs a key "abc123" in `store`.
# # )["answer"]
# # print(ans)
# # print('\n \n ############################################################### \n \n')
# # #%%
# # ans = conversational_rag_chain.invoke(
# #     {"input": "What are common ways of doing it?"},
# #     config={"configurable": {"session_id": "abc123"}},
# # )["answer"]

# # print(ans)

# def get_ai_answer(prompt, return_context=False):
#     ai_response = conversational_rag_chain.invoke(
#         {"input": prompt},
#         config={"configurable": {"session_id": "abc123"}},
#     )
    

#     ai_input = ai_response['input']
#     ai_answer = ai_response['answer']
#     ai_context = ai_response['context']
#     ai_chat_history = ai_response['chat_history']

#     if return_context == True:
#         return ai_input, ai_answer, ai_context, ai_chat_history
#     else:
#         return ai_answer


# #%%

# # st.title("LLM with RAG FOR QURAN")

# # # Add a reset button to restart the session
# # if st.sidebar.button("Restart Session"):
# #     # Reset session state variables
# #     st.session_state.clear() 

# # # Initialize chat history
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # Display chat messages from history on app rerun
# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])



# # return_context = True
# # user_choice = st.sidebar.radio("Print Context", ["Yes", "No"])

# # if user_choice == 'Yes':
# #     return_context = True
# # else:
# #     return_context = False

# # React to user input
# # if prompt := st.chat_input("What is up?"):
# #     # Display user message in chat message container
# #     st.chat_message("user").markdown(prompt)
# #     # Add user message to chat history
# #     st.session_state.messages.append({"role": "user", "content": prompt})
    
# #     if return_context == True:
# #         ai_answer, ai_context = get_ai_answer(prompt, return_context=return_context)
# #     else:
# #         ai_answer = get_ai_answer(prompt)

# #     response = f"ChatGPT: {ai_answer}"
# #     # Display assistant response in chat message container
# #     with st.chat_message("assistant"):
# #         st.markdown(response)
# #     # Add assistant response to chat history
# #     if return_context == True:
# #         with st.chat_message("AI"):
# #             st.markdown(ai_context)
# #     st.session_state.messages.append({"role": "assistant", "content": response})



# # print('------------------------')
# # print('------------------------')
# # print(get_session_history("abc123"))
# # # print()

# # print('------------------------')
# # print('------------------------')
# # # print()
# # # print()
# # # print() 
# # # Get chat history from store
# # # for x in list(dict(store)['abc123'])[0][1]:
# # #   print(x)

# # #%%
# # # # TEST
# # # a = rag_chain.invoke({'input':'Who is Allah', 'chat_history': []})
# # # a

# question_list = [
#     "Who is Allah",
#     "Who is Muhammad",
#     "Who is Ibrahim",
#     "Tell me about Moses",
#     "Tell me about Isa",
#     "explain surah falaq",
#     "Narrate the story of Moses",
#     "How many surah are there in quran",
#     "which is the first 10 surah in quran",
#     "Narrate story of yajuj and Majuj",
#     "What is Tawbah",
#     "What are the name of Ibrahim's sons",
#     "What is the concept of Tawhid (oneness of God) in the Quran, and how does it emphasize the relationship between Allah and His creation",
#     "Can you explain the story of Prophet Yusuf (Joseph) in the Quran and the lessons it teaches about patience, forgiveness, and trust in Allah?",
#     "What are the rights and responsibilities between spouses as described in the Quran?",
#     "How does the Quran describe the purpose of human life and the concept of accountability in the Hereafter?",
#     "What does the Quran say about compassion and justice when dealing with others, even those who may oppose you?",
#     "Can you summarize the teachings of the Quran on charity (Zakat) and helping those in need?" ,
#     "What are the Quranic guidelines on the treatment of orphans and the vulnerable in society?",
#     "How does the Quran address the concept of free will and predestination (Qadar), and what balance does it strike between them?",
#     "What guidance does the Quran provide about dealing with hardships and challenges in life?",
#     "What are the signs of Allah’s creation mentioned in the Quran, and how do these encourage reflection and understanding?",
#     "Who are some of the main prophets mentioned in the Quran, and what is their role?",
#     "What does the Quran say about being kind to parents?",
#     "How does the Quran describe the creation of the world?",
#     "What story does the Quran tell about Prophet Adam and the beginning of humanity?",
#     "What guidance does the Quran give about being honest and truthful?",
#     "What does the Quran say about helping the poor and giving charity?",
#     "How does the Quran describe Heaven (Paradise) and Hell?",
#     "What are the main duties of a Muslim according to the Quran?",
#     "How does the Quran instruct people to treat others fairly?",
#     "What story does the Quran tell about Prophet Musa (Moses) and Pharaoh?",
#                  ]

# if demo == True:
#     question_list = [
#         "Who is Allah",
#         "Who is Muhammad",
#         "What story does the Quran tell about Prophet Musa (Moses) and Pharaoh?",
#                     ]
#     print('Demo list applied')
# # prompt = question_list[13]

# all_question_answer_list = []
# #%%
# for prompt in question_list:
#     ai_input, ai_answer, ai_context, ai_chat_history = get_ai_answer(prompt, return_context=True)
#     # ai_response_list = [ai_input, ai_answer, ai_context, ai_chat_history]
#     # ai_response_list

#     # Contextualise the question
#     # question = 'What is Tawbah'
#     chain = (contextualize_q_prompt | llm | StrOutputParser())
#     contextualize_question = chain.invoke({"input": prompt, "chat_history": tuple(store['abc123'])[0][1]})
#     contextualize_question

#     # # Get direct context
#     # direct_context = history_aware_retriever.invoke({'input': prompt})

#     result_list = [splitter_type, store_type, retriever_type, 
#                     ai_input, ai_answer, ai_chat_history,
#                     ai_context, 
#                     contextualize_question
#                     # ,direct_context
#                     ]

#     all_question_answer_list.append(result_list)
# #%%
# df = pd.DataFrame(all_question_answer_list)
# df.columns = ['splitter_type', 'store_type', 'retriever_type', 
#                     'ai_input', 'ai_answer', 'ai_chat_history',
#                     'ai_context', 'contextualize_question'
#                     # ,'direct_context'
#                     ]


# # %%


# # Extract number of context provided for each answer in column 'ai_context_len' 
# df = df.apply(extract_ai_context_len, axis=1)

# # Create context columns
# if max_context_column >= df['ai_context_len'].max():
#     # # Create columns
#     for x in range(1, max_context_column+1):
#         df[f"ai_context_{x}"] = None

# else:
#     max_context_column = df['ai_context_len'].max()
#     for x in range(1, max_context_column+1):
#         df[f"ai_context_{x}"] = None


# # Fill the context column with only the context text
# df = df.apply(extract_context, axis=1)

# # Change dtype to string for ease in saving
# df['ai_chat_history'] = df['ai_chat_history'].astype(str)

# # Remove ai_context column
# df.drop('ai_context', axis=1, inplace=True)

# # Save df as parquet
# df.to_parquet("eval_result\\" + splitter_type + '_' + store_type + '_' + retriever_type + ".parquet")
