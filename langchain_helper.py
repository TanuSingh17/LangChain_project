# LangChain pipeline where:
# 1.prompt template formats the input
# 2. LLM generates AI response
# 3. parser converts output into clean string

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from secret_key import openapi_key
import os

os.environ["OPENAI_API_KEY"] = openapi_key

# Initialize LLM
# llm = OpenAI(temperature=0.7)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Output parser
parser = StrOutputParser()


def generate_restaurant_name_and_items(cuisine):

    # Prompt 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    # Chain 1
    name_chain = prompt_template_name | llm | parser

    # Generate restaurant name
    restaurant_name = name_chain.invoke({"cuisine": cuisine})

    # Prompt 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
    )

    # Chain 2
    food_chain = prompt_template_items | llm | parser

    # Generate menu items
    menu_items = food_chain.invoke(
        {"restaurant_name": restaurant_name}
    )

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))