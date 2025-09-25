import os
from dotenv import load_dotenv
from typing import Literal, Sequence, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import sympy as sp

# Load environment variables (set OPENAI_API_KEY in .env)
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the state with annotated messages for auto-append
class State(TypedDict):
    """Custom state for chat history."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Router prompt to classify the input
router_prompt = ChatPromptTemplate.from_template(
    "Classify the user input: {input}\n"
    "If it's a math problem (e.g., solve equation, calculate), respond with 'math'.\n"
    "Otherwise, respond with 'general'.\n"
    "Only output 'math' or 'general'."
)

# Condition function: Decides the path based on user input
def route_decision(state: State) -> Literal["chatbot", "math_solver"]:
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages and messages[-1].content else ""
    
    # Fallback to general if classification fails
    try:
        classification = llm.invoke(router_prompt.format(input=last_message)).content.strip().lower()
        return "math_solver" if classification == "math" else "chatbot"
    except Exception:
        return "chatbot"  # Default to chatbot on error

# Dummy router node: No-op, just for entry point
def router_node(state: State) -> dict:
    return {}

# Chatbot node: General response using LLM
def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": response}

# Math solver node: Uses SymPy to solve math problems
def math_solver(state: State) -> dict:
    last_message = state["messages"][-1].content
    
    # Simple parsing: Assume input like "solve x^2 - 4 = 0"
    try:
        if "solve" in last_message.lower():
            eq_str = last_message.lower().split("solve")[1].strip()
            eq = sp.sympify(eq_str)
            solution = sp.solve(eq)
            response_content = f"Solution: {solution}"
        else:
            response_content = "Please phrase as 'solve [equation]'."
    except Exception as e:
        response_content = f"Error solving: {str(e)}"
    
    return {"messages": AIMessage(content=response_content)}

# Build the graph with conditional nodes
def build_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("math_solver", math_solver)
    
    # Set initial node
    workflow.set_entry_point("router")
    
    # Conditional edge from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "chatbot": "chatbot",
            "math_solver": "math_solver"
        }
    )
    
    # From both branches to END
    workflow.add_edge("chatbot", END)
    workflow.add_edge("math_solver", END)
    
    return workflow.compile()

# Run the multi-turn chatbot
def run_multi_turn_chatbot(graph):
    state = {"messages": []}
    
    print("Chatbot ready! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Immutable update for messages
        new_messages = state["messages"] + [HumanMessage(content=user_input)]
        state = {"messages": new_messages}
        
        # Invoke the graph
        result = graph.invoke(state)
        
        # Update state
        state = result
        
        # Get and print the last AI response
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print("AI:", last_message.content)
        else:
            print("AI: Unexpected response.")

if __name__ == "__main__":
    graph = build_graph()
    run_multi_turn_chatbot(graph)