from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel
import  os
import gradio as gr
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient

memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

from langchain_openai import ChatOpenAI
try:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key="AIzaSyD8EQItv3a2p9PNpdgf4nK7vLStWuzwAcg", convert_system_message_to_human=True, temperature=0.6)
except Exception as e:
    print("Error",e)
PLAN_PROMPT = """You are an experienced writer tasked with drafting a highly detailed and comprehensive outline for a research report. Your goal is to create an exceptionally structured outline for the user's chosen topic, ensuring it encourages an in-depth examination of recent trends, specific advancements, technical details, and future research directions.
**Outline Requirements:**
1. **Executive Summary:**
   - Summarize key findings, significance, and implications.
   - Highlight main conclusions and recommendations.
2. **Introduction:**
   - Provide background on the topic.
   - State research questions or hypotheses.
   - Define the  scope and objectives of the report.
3. **Recent Trends and Advancements:**
    -give resposne according to users topic
4. **Integration of Technologies:**
   - Discuss how these technologies intersect and enhance each other.
   - Provide examples of synergistic effects and combined applications.
   - Address integration challenges and potential solutions.
5. **Privacy and Security Challenges:**
   - Identify key privacy and security issues.
   - Discuss mitigation strategies and best practices.
   - Explore ethical considerations related to these technologies.
6. **Case Studies:**
   - Include detailed case studies with specific examples.
   - Analyze real-world applications and their outcomes.
   - Discuss lessons learned and best practices from each case study.
7. **Future Research Directions:**
   - Propose areas for further investigation and development.
   - Highlight emerging trends and technologies.
   - Discuss potential impact and applications of future research.
8. **Conclusions:**
   - Summarize main findings and insights.
   - Reflect on the implications for researchers and practitioners.
   - Provide actionable recommendations and future outlook.
9. **References:**
   - Include a comprehensive list of recent research papers, articles, and other relevant sources.
   - Ensure accurate and complete citations to support the analysis.
**Additional Instructions:**
- Break down complex topics into clear sub-sections.
- Ensure each section transitions smoothly to the next.
- Use visual aids where applicable (e.g., charts, graphs, tables).
- Address privacy and security systematically throughout the report.
"""


WRITER_PROMPT = """You are an essay assistant specializing in research reports. Produce a comprehensive research report based on the provided outline and user's requirements. Your report should cover recent trends, specific advancements, technical details, and future research directions in depth.
**Report Requirements:**
1. **Executive Summary:**
   - Summarize key findings, significance, and implications.
   - Ensure clarity and brevity while covering essential points.
2. **Introduction:**
   - Provide detailed background and context.
   - State and elaborate on research questions or hypotheses.
3. **Recent Trends and Advancements:**
   - Present detailed and up-to-date information on each technology.
   - Use concrete examples and recent data to support your analysis.
4. **Integration of Technologies:**
   - Discuss how technologies intersect and enhance each other.
   - Include specific examples of successful integration and its benefits.
5. **Privacy and Security Challenges:**
   - Thoroughly address potential privacy and security issues.
   - Provide a balanced view of potential solutions and ongoing research.
6. **Case Studies:**
   - Include detailed, well-researched case studies.
   - Provide concrete examples and analyze outcomes and lessons learned.
7. **Future Research Directions:**
   - Propose and justify areas for further investigation.
   - Discuss potential future applications and their significance.
8. **Conclusions:**
   - Summarize key insights and recommendations clearly.
   - Ensure the conclusions reflect the report’s findings and implications.
9. **References:**
   - Accurately cite recent research papers and relevant sources.
   - Ensure all references are credible and support the report’s content.
**Additional Instructions:**
- Maintain a clear and logical flow throughout the report.
- Use visual aids to present data effectively.
- Ensure smooth transitions between sections for readability.
- Provide a critical analysis of various perspectives.
{content}
"""


REFLECTION_PROMPT = """You are a seasoned educator evaluating an essay submission. Provide constructive feedback on the user's research report, focusing on depth, comprehensiveness, and clarity.
**Evaluation Criteria:**
1. **Coverage of Recent Trends and Advancements:**
   - Assess how well the report addresses recent trends and specific advancements.
   - Check for detailed and accurate information on each technology.
2. **Integration and Synergies:**
   - Evaluate how effectively the report discusses the integration of technologies and their synergies.
   - Look for examples and explanations of how these technologies enhance each other.
3. **Privacy and Security Challenges:**
   - Assess the depth of discussion on privacy and security challenges.
   - Evaluate the effectiveness of proposed solutions and mitigation strategies.
4. **Case Studies and Real-World Applications:**
   - Review the inclusion and analysis of case studies.
   - Check for relevance, detail, and real-world applicability of examples.
5. **Future Research Directions:**
   - Evaluate the clarity and feasibility of proposed future research directions.
   - Assess the potential impact and significance of these directions.
6. **References and Citations:**
   - Ensure that the report includes accurate and comprehensive citations.
   - Review the relevance and credibility of referenced sources.
**Recommendations:**
- Provide specific suggestions for improving depth and clarity.
- Highlight any gaps or areas needing further development.
- Suggest ways to enhance readability and logical flow.
"""

RESEARCH_PLAN_PROMPT = """As a researcher, your task is to generate effective search queries to gather comprehensive information for the upcoming research report. Provide up to 4 targeted queries to obtain the most relevant and recent data.
**Search Queries:**
1. **Latest Developments in [Technology]:** Focus on recent advancements and breakthroughs in the technology related to your report.
2. **Technical Details and Innovations:** Seek detailed technical information and innovations for the technology in question.
3. **Case Studies and Real-World Applications:** Find concrete examples and case studies that demonstrate the application of the technology.
4. **Integration and Privacy/Security Challenges:** Explore how different technologies integrate, their potential synergies, and address privacy/security challenges.
**Instructions:**
- Use credible sources and evaluate the relevance of the information.
- Ensure queries are specific to the technology and research focus.
- Gather information that supports a thorough and comprehensive analysis.
"""

RESEARCH_CRITIQUE_PROMPT = """You are tasked with gathering updated information to assist in revising the research report. Generate up to 3 focused search queries to find the most relevant and recent data.
**Search Queries:**
1. **Recent Trends in [Technology]:** Identify the latest trends and advancements in the technology relevant to the report.
2. **Detailed Technical Insights:** Obtain detailed technical information and recent innovations for a thorough understanding.
3. **Privacy and Security Challenges:** Find updated discussions on privacy and security challenges related to integrating the technologies.
**Instructions:**
- Ensure the queries target high-quality and credible sources.
- Focus on obtaining comprehensive and relevant data.
- Evaluate the information for its relevance and contribution to the report.
"""




class Queries(BaseModel):
    queries: List[str]



client = TavilyClient(api_key="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")
def plan_node(state: AgentState):
    print("reached planner node")
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]

    response = model.invoke(messages)

    return {"plan": response.content}

def research_plan_node(state: AgentState):
    print("reached research_plan_node")

    # Send messages to the model without structured output
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]

    try:
        response = model.invoke(messages)

        # Extract the text content from the response
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse the queries from the text content
        queries_list = parse_queries_from_content(response_text)
        print("parsed queries list", queries_list)

        # Perform search based on extracted queries
        content = state['content'] or []
        for q in queries_list:
            response = client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])

        return {"content": content}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"content": state['content']}

def parse_queries_from_content(content: str):
    # Split the content by newlines
    lines = content.split('\n')

    # Filter out empty lines and extract queries
    queries = [line.strip() for line in lines if line.strip()]

    # Remove numbering and quotes if needed
    queries = [q.split('. ', 1)[-1].strip('"') for q in queries]

    return queries
def generation_node(state: AgentState):
    print("reached generation_node")
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }
def reflection_node(state: AgentState):
    print("reached reflection_node")
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    print("reached research_critique_node")

    # Send messages to the model without structured output
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ]
    print("passed 1")

    try:
        response = model.invoke(messages)
        print("passed 2")

        # Extract content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)
        print("queries are", response_content)

        # Parse queries from the response content
        queries_list = parse_queries_from_text(response_content)
        print("queries_list", queries_list)

        # Perform search based on extracted queries
        content = state['content'] or []
        for q in queries_list:
            response = client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])

        return {"content": content}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"content": state['content']}

def parse_queries_from_text(text):
    print(text)
    # Assuming the queries are separated by new lines and potentially prefixed by numbers or bullet points
    lines = text.split('\n')

    # Clean up and filter out empty lines
    queries = [line.strip() for line in lines if line.strip()]

    # If needed, remove numbering or bullet points
    queries = [q.split('. ', 1)[-1].strip('"') for q in queries]

    return queries

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")


builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")


graph = builder.compile(checkpointer=memory)



# (Include all the existing code for AgentState, prompts, and node functions here)

# ... (keep all the existing code up to the graph compilation)

graph = builder.compile(checkpointer=memory)

def process_essay(task, max_revisions):
    thread = {"configurable": {"thread_id": "1"}}
    results = []
    for s in graph.stream({
        'task': task,
        "max_revisions": max_revisions,
        "revision_number": 1,
    }, thread):
        results.append(s)
    return results

def extract_agent_outputs(results):
    plan_output = ""
    research_output = ""
    generation_output = ""
    reflection_output = ""
    final_essay = ""

    for step in results:
        if 'planner' in step:
            plan_output += f"Plan:\n{step['planner']['plan']}\n\n"
        if 'research_plan' in step or 'research_critique' in step:
            research_key = 'research_plan' if 'research_plan' in step else 'research_critique'
            research_output += f"Research Queries:\n{step[research_key]['content']}\n\n"
        if 'generate' in step:
            generation_output += f"Draft (Revision {step['generate']['revision_number']}):\n{step['generate']['draft']}\n\n"
            final_essay = step['generate']['draft']  # Update final essay
        if 'reflect' in step:
            reflection_output += f"Critique:\n{step['reflect']['critique']}\n\n"

    return plan_output, research_output, generation_output, reflection_output, final_essay

graph = builder.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))

# Define functions for Gradio
def process_essay(task, max_revisions):
    thread = {"configurable": {"thread_id": "1"}}
    results = []
    for s in graph.stream({
        'task': task,
        "max_revisions": max_revisions,
        "revision_number": 1,
    }, thread):
        results.append(s)
    return results

def extract_agent_outputs(results):
    plan_output = ""
    research_output = ""
    generation_output = ""
    reflection_output = ""
    final_essay = ""

    for step in results:
        print("Processing step:", step)  # Debug line
        if 'planner' in step:
            plan_output += f"Plan:\n{step['planner']['plan']}\n\n"
        if 'research_plan' in step or 'research_critique' in step:
            research_key = 'research_plan' if 'research_plan' in step else 'research_critique'
            if step[research_key]:
                research_output += f"Research Queries:\n{step[research_key]['content']}\n\n"
            else:
                research_output += f"Research Queries:\nNo content found.\n\n"
        if 'generate' in step:
            generation_output += f"Draft (Revision {step['generate']['revision_number']}):\n{step['generate']['draft']}\n\n"
            final_essay = step['generate']['draft']  # Update final essay
        if 'reflect' in step:
            if step['reflect']:
                reflection_output += f"Critique:\n{step['reflect']['critique']}\n\n"
            else:
                reflection_output += f"Critique:\nNo critique found.\n\n"

    return plan_output, research_output, generation_output, reflection_output, final_essay

def gradio_interface(task, max_revisions):
    results = process_essay(task, max_revisions)
    plan, research, generation, reflection, final_essay = extract_agent_outputs(results)
    return final_essay, plan, research, generation, reflection

# Create and launch Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Report Topic", placeholder="Enter the report  topic or task here"),
        gr.Slider(minimum=1, maximum=5, step=1, label="Max Revisions", value=2)
    ],
    outputs=[
        gr.Textbox(label="Final Report"),
        gr.Textbox(label="Planning Output", lines=10),
        gr.Textbox(label="Research Output", lines=10),
        gr.Textbox(label="Generation Output", lines=10),
        gr.Textbox(label="Reflection Output", lines=10)
    ],
    title="Multi-Agent Report  Writing System",
    description="This system uses multiple AI agents to plan, research, write, and revise an research report  on the given topic."
)

iface.launch()
