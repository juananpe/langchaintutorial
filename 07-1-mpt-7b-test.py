from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from ctransformers.langchain import CTransformers
from langchain.embeddings import HuggingFaceInstructEmbeddings

llm = CTransformers(model='/tmp/mpt-7b-instruct.ggmlv3.q5_0.bin', 
                    model_type='mpt')

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                      model_kwargs={"device": "cpu"})

db = FAISS.load_local("faiss_index", instructor_embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever)