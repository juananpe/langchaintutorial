from langchain import LLMChain, OpenAI, PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    temperature=0
    )


prompt = PromptTemplate(input_variables=['question'], template='{question}')
llm_chain = LLMChain(prompt=prompt, llm=llm)
print("Answer: " + llm_chain.run("What is 13 raised to the .3432 power?"))
