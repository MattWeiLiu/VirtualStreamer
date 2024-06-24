from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import VertexAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.llms import VertexAI


class qa_chain(object):
    def __init__(self, config):
        self.config = config
    
    def get_qa_chain_memory(self, prior_memory=None):
        if self.config['memory_type'] == "summary":
            self.memory = ConversationSummaryMemory(llm=VertexAI(model_name="text-bison@002",
                                                                 project="aiops-338206",
                                                                 max_output_tokens=512,
                                                                 top_p=1,
                                                                 temperature=0.7),
                                                    memory_key='chat_history',
                                                    input_key='question',
                                                    return_messages=True)
            if prior_memory:
                self.memory.buffer = prior_memory
            
        elif self.config['memory_type'] == "bufferwindow":
            self.memory = ConversationBufferWindowMemory(k=10, memory_key='chat_history',
                                                         input_key='question', return_messages=True)
            if prior_memory:
                self.memory.chat_memory = prior_memory
        else:
            raise Exception("in qa chain type, it must hase memory")
        
        return self.memory
            
    def get_retriver_from_external(self, content):
        doc = [Document(page_content=content, metadata={"source": "local"})]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(doc)

        # vertexai embedding model: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings
        embeddings = VertexAIEmbeddings(model_name='textembedding-gecko-multilingual@001')
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        return retriever

    def get_qa_chain(self, llm, retriever, memory, prompt):
        self.chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                           retriever=retriever,
                                                           memory=memory,
                                                           chain_type="stuff",  # map_reduce 比較慢
                                                           rephrase_question=False,  # need to test
                                                           combine_docs_chain_kwargs={"prompt": prompt}
                                                           )
    
    def predict(self, input, current_time=None):
        if current_time:
            output = self.chain({"question": input, "current_time": current_time})
        else:
            output = self.chain({"question": input})
        return output['answer']
