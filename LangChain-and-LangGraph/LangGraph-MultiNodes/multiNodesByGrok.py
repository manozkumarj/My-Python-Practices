from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama  # Correct import
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import tool
from typing import TypedDict, List, Annotated, Optional
import operator
import os

# Set up local LLM (no OpenAI API key needed)
llm = ChatOllama(model="llama3", temperature=0.7)  # Assumes Ollama is running

# Set Tavily API key (only if using Tavily; see notes for skipping)
os.environ["TAVILY_API_KEY"] = "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"  # Replace with your Tavily key

# Custom Arxiv tool
@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for academic papers."""
    arxiv = ArxivAPIWrapper()
    return arxiv.run(query)

# Define tools
search_tool = DuckDuckGoSearchRun(name="web_search", description="Search the web for current information.")
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), name="wikipedia_search", description="Search Wikipedia for general knowledge.")
arxiv_tool = arxiv_search
python_repl_tool = PythonREPLTool(name="python_calculator", description="Execute Python code for math, calculations, or data processing.")

tools = [search_tool, wiki_tool, arxiv_tool, python_repl_tool]
llm_with_tools = llm.bind_tools(tools)

# Define the state
class GraphState(TypedDict):
    messages: Annotated[List[AIMessage], operator.add]
    query: str
    is_valid: Optional[bool]
    tool_results: Optional[str]
    answer: Optional[str]
    formatted_output: Optional[str]
    tool_usage_log: List[str]

# Node 1: Process Input
def process_input(state: GraphState) -> GraphState:
    query = state["query"].strip()
    messages = [HumanMessage(content=query)]
    return {
        "query": query,
        "messages": messages,
        "is_valid": None,
        "tool_results": None,
        "answer": None,
        "formatted_output": None,
        "tool_usage_log": []
    }

# Node 2: Validate Query
def validate_query(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Analyze the query: Does it need web search (current events), Wikipedia (general facts), Arxiv (research papers), or Python calc (math)? Respond with 'Direct' if answerable from knowledge, else 'Needs Tools: [tool1, tool2]'"),
        HumanMessage(content=state["query"])
    ])
    result = llm.invoke(prompt.messages).content
    needs_tools = "needs tools" in result.lower()
    is_valid = not needs_tools
    return {"is_valid": is_valid, "tool_usage_log": [result]}

# Node 3: Agent Decision
def agent_decision(state: GraphState) -> GraphState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "tool_usage_log": state["tool_usage_log"] + ["Agent decided on action"]}

# Node 4: Tool Execution
def tool_execution(state: GraphState) -> GraphState:
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls or []
    if not tool_calls:
        return {"tool_results": "No tools called."}
    
    tool_results = []
    used_tools = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_func = next((t for t in tools if t.name == tool_name), None)
        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
                used_tools.append(tool_name)
            except Exception as e:
                tool_results.append(ToolMessage(content=f"Error in {tool_name}: {str(e)}", tool_call_id=tool_call["id"]))
                used_tools.append(f"{tool_name} (error)")
        else:
            tool_results.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call["id"]))
    
    combined_results = "\n".join([msg.content for msg in tool_results])
    return {
        "messages": tool_results,
        "tool_results": combined_results,
        "tool_usage_log": state["tool_usage_log"] + [f"Executed: {', '.join(used_tools)}"]
    }

# Node 5: Generate Answer
def generate_answer(state: GraphState) -> GraphState:
    messages = state["messages"]
    tool_results = state.get("tool_results", "")
    log = state["tool_usage_log"]
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"Synthesize a comprehensive answer using these tool results: {tool_results}\nTool log: {log}"),
        *messages
    ])
    answer = llm.invoke(prompt.messages).content
    return {"answer": answer, "tool_usage_log": log + ["Answer generated"]}

# Node 6: Format Output
def format_output(state: GraphState) -> GraphState:
    answer = state["answer"]
    log = "\n".join(state["tool_usage_log"])
    formatted = f"**Query**: {state['query']}\n**Answer**: {answer}\n\n**Tool Usage Log**:\n{log}"
    return {"formatted_output": formatted}

# Conditional routing
def route_after_validation(state: GraphState) -> str:
    return "agent_decision" if not state["is_valid"] else "generate_answer"

def route_after_agent(state: GraphState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_execution"
    return "generate_answer"

def route_after_tools(state: GraphState) -> str:
    return "agent_decision"  # Loop back for potential refinement

# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("process_input", process_input)
workflow.add_node("validate_query", validate_query)
workflow.add_node("agent_decision", agent_decision)
workflow.add_node("tool_execution", tool_execution)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("format_output", format_output)

# Define edges
workflow.add_edge("process_input", "validate_query")
workflow.add_conditional_edges(
    "validate_query",
    route_after_validation,
    {"agent_decision": "agent_decision", "generate_answer": "generate_answer"}
)
workflow.add_conditional_edges(
    "agent_decision",
    route_after_agent,
    {"tool_execution": "tool_execution", "generate_answer": "generate_answer"}
)
workflow.add_conditional_edges("tool_execution", route_after_tools)
workflow.add_edge("generate_answer", "format_output")
workflow.add_edge("format_output", END)

# Set entry point
workflow.set_entry_point("process_input")

# Compile the graph
graph = workflow.compile()

# Run the graph
def run_multi_tool_agent(query: str) -> dict:
    initial_state = {"query": query}
    result = graph.invoke(initial_state)
    return result

# Example usage
if __name__ == "__main__":
    queries = [
        "What is the integral of x^2 from 0 to 1?",
        "Latest advancements in quantum computing",
        "How many people attended the 2024 Olympics opening ceremony? Estimate total cost per attendee."
    ]
    for query in queries:
        result = run_multi_tool_agent(query)
        print(result["formatted_output"])
        print("\n" + "="*50 + "\n")