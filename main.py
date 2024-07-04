from dotenv import load_dotenv
import os
import pandas as pd
# from llama_index.core.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
from prompt import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import uk_engine
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings


env_path = os.getenv('ENV_FILE', '/Users/mac/PycharmProjects/RAG/.env')
load_dotenv(dotenv_path=env_path)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY environment variable.")

print(f"Loaded API Key: {api_key}")

openai.api_key = api_key

OpenAIEmbedding.api_key = api_key

Settings.embed_model = OpenAIEmbedding(api_key=api_key)



population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("what is the population of canada")

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description="this gives information about the world population and demographics",
    ),),
    QueryEngineTool(query_engine=uk_engine, metadata=ToolMetadata(
        name="ireland_data",
        description="this gives detailed information about the United Kingdom",
    ),),
]


llm = OpenAI(model="gpt-3.5-turbo-16k")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context)

while (prompt := input("enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

