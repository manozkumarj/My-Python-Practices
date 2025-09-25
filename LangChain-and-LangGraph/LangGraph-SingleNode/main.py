import os
from dotenv import load_dotenv
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables (set OPENAI_API_KEY in .env)
load_dotenv()

# Define the state (conversation history)
class State(MessagesState):
    """Custom state extending MessagesState for chat history."""
    pass

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the chatbot node
def chatbot(state: State) -> State:
    """Chatbot node: Takes the current messages and appends an AI response."""
    messages = state["messages"]
    # Get the last user message as input
    user_input = messages[-1].content if messages else "Hello!"
    
    # Invoke the LLM to generate a response
    response = llm.invoke(messages)
    
    # Append the AI response to the state
    return {"messages": [response]}

# Build the graph
def build_graph():
    """Constructs the LangGraph workflow matching the diagram."""
    workflow = StateGraph(State)
    
    # Add the chatbot node
    workflow.add_node("chatbot", chatbot)
    
    # Define edges: start -> chatbot -> end
    workflow.add_edge("__start__", "chatbot")
    workflow.add_edge("chatbot", END)
    
    # Compile the graph
    return workflow.compile()

# Run the workflow
def run_chatbot(graph, user_input: str):
    """Executes the graph with user input."""
    # Initial state with user message
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Invoke the graph
    result = graph.invoke(initial_state)
    
    # Extract and print the final messages
    final_messages = result["messages"]
    print("User:", final_messages[-2].content)  # Last user message
    print("AI:", final_messages[-1].content)    # AI response
    return result

if __name__ == "__main__":
    # Build and run
    graph = build_graph()
    
    # Example user input (in a real app, this could be from CLI/input loop)
    user_prompt = input("Enter your message: ") or "Tell me a joke."
    
    result = run_chatbot(graph, user_prompt)
    
    # For multi-turn, you could loop and pass updated state, but keeping single-turn per the diagram
    print("\nWorkflow complete!")