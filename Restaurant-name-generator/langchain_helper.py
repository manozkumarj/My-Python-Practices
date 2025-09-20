from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from secret_key import groq_api_key
import os

os.environ["GROQ_API_KEY"] = groq_api_key

# Use a supported model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

def generate_restaurant_name_and_items(cuisine: str):
    prompt_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_name, output_key="restaurant_name")

    prompt_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
    )
    items_chain = LLMChain(llm=llm, prompt=prompt_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
        verbose=True
    )

    return chain.invoke({"cuisine": cuisine})

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
