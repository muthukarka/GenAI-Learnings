from langchain_community.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from few_shots import few_shots
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

few_shots = [
    {'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'SQLResult': "Result of the SQL query",
     'Answer': "39"
     },
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "Result of the SQL query",
     'Answer': "14041"
     },
    {
        'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
        'SQLQuery': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
        'SQLResult': "Result of the SQL query",
        'Answer': "29496.50"
        },
    {
        'Question': "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?",
        'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        'SQLResult': "Result of the SQL query",
        'Answer': "29759"},
    {'Question': "How many white color Levi's shirt I have?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     'SQLResult': "Result of the SQL query",
     'Answer': "193"
     },
    {
        'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
        'SQLQuery': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
        'SQLResult': "Result of the SQL query",
        'Answer': "1040"
        }
]


def get_few_shot_db_chain():
    db_user = "root"
    db_password = "password"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    llm = ChatOpenAI(temperature=0.1, model="gpt-4-turbo")

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        few_shots,
        OpenAIEmbeddings(),
        Chroma,
        k=5,
        input_keys=["Question"],
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    system_prefix = """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run, then look at the results of the query and return the answer.
            Never query all columns from a table—query only the necessary ones. Wrap each column name in backticks (`). 
            Use `CURDATE()` for today's date if needed.

            Use this format:

            Question: {Question}
            SQLQuery: {SQLQuery}
            SQLResult: {SQLResult}
            Answer: {Answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        input_variables=["Question"],
        prefix=system_prefix,
        suffix="",
    )

    formatted_prompt = few_shot_prompt.format_prompt(
        Question="{input}",
        SQLQuery="",  # Placeholder to avoid KeyError
        SQLResult="",
        Answer=""
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", formatted_prompt.to_string()),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # print("Full Prompt Variables:", full_prompt.input_variables)
    # print("Example Selector Output:", example_selector.select_examples({"Question": "How many Levi’s T-shirts do I have?"}))

    # tools = [QuerySQLDataBaseTool(db=db)]

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prompt=full_prompt,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )

    return agent


if __name__ == "__main__":
    # Your main code logic here
    chain = get_few_shot_db_chain()
    response = chain.run("How many Levi tshirts are present in small size ?")
    print("answer from LLM " + response)