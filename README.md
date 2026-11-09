## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
### NAME : MURALIDHARAN M
### REG.NO : 212223040120
### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/attachment?id=mu17VSX8q9&name=pdf",
    "https://openreview.net/attachment?id=9ydYaIe1Qj&name=pdf",
    "https://openreview.net/attachment?id=K83orsg2X9&name=pdf",
]

papers = [
    "28312_Reward_Redistribution_fo.pdf",
    "28435_The_Axiomatic_Value_of_R.pdf",
    "28445_The_Power_of_Power_Law_A.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "What is midtraining in the context of large language model training?, "
    "What is the typical relationship between the sizes of pretraining, midtraining, and target datasets?"
)
)
response = agent.query("Give me a summary of  midtraining")
print(str(response))

response = agent.query("Give me a summary of  natural language data follow")
print(str(response))

```

### OUTPUT:
<img width="1318" height="557" alt="image" src="https://github.com/user-attachments/assets/c0f2f820-da39-43cb-9d94-3035fc1d3034" />
<img width="1324" height="517" alt="image" src="https://github.com/user-attachments/assets/7523010e-ee51-43e2-87a0-1d21031c826f" />
<img width="1339" height="331" alt="Screenshot 2026-08-28 140529" src="https://github.com/user-attachments/assets/d49b44b5-757a-4586-b644-71118c2cb9bd" />



### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
