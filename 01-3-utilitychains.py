from langchain import OpenAI
from langchain.chains import LLMBashChain
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    temperature=0
    )

llm_bash = LLMBashChain.from_llm(llm=llm, verbose=True)

# print(llm_bash.prompt.template)
# print(llm_bash.run("List the current directory then move up a level."))
print(llm_bash.run("Find in the current directory, not recursively, all py files that include 'util' in their names."))





# Advanced: inspecting the source code of a method
# import inspect
# print(inspect.getsource(llm_math._call))