# vetting_prompts.py

VETTING_PROMPT_1_HF_DATASETS = """Review the following agent step and consider Hugging Face Datasets:


Step Tag: <Step Tag>

Prior Results Tag: <Prior Results Tag> (if applicable)

Thought Tag: <Thought Tag>

Action Tag: <Action Tag>


Based on the 'Thought' and 'Action' of this step, search Hugging Face Datasets for datasets that are relevant to the agent's task or the action it is taking. Identify datasets that could potentially:


Improve the agent's understanding of the task.


Provide data for the agent to use directly.


Be used to train or fine-tune models for similar tasks.


List any relevant datasets found, including their names and a brief explanation of why they are relevant and how they could be beneficial to the agent. If no relevant datasets are found, explicitly state that no relevant datasets were found on Hugging Face Datasets."""


VETTING_PROMPT_2_GITHUB_PROJECTS = """Review the following agent step and consider GitHub Projects:


Step Tag: <Step Tag>

Prior Results Tag: <Prior Results Tag> (if applicable)

Thought Tag: <Thought Tag>

Action Tag: <Action Tag>

Observation Tag: <Observation Tag>


Based on the 'Thought', 'Action', and 'Observation' of this step, search GitHub for projects that are similar to the agent's current task or the action it has taken. Look for projects that:


Implement similar functionalities or tools.


Solve related problems or tasks.


Offer alternative approaches or code implementations.


List any relevant GitHub projects found, including project names, repository links, and a brief explanation of how they are similar and potentially helpful to the agent. Consider aspects like code structure, algorithms used, or tool integrations. If no relevant projects are found, explicitly state that no relevant projects were found on GitHub."""


VETTING_PROMPT_3_WEB_SEARCH = """Review the following agent step and perform a Contextualized Web Search:


Step Tag: <Step Tag>

Prior Results Tag: <Prior Results Tag> (if applicable)

Thought Tag: <Thought Tag>

Action Tag: <Action Tag>

Observation Tag: <Observation Tag>


Perform a contextualized web search based on the agent's 'Thought', 'Action', and 'Observation'. The goal of this search is to:


Validate the information obtained in the 'Observation'.


Find more detailed or supporting information related to the 'Thought' or 'Action'.


Explore alternative perspectives, solutions, or data sources relevant to the agent's task.


Summarize the key findings from the web search that are relevant to the agent's step. Specifically, identify:


Any confirmations or contradictions of the 'Observation'.


New information that could enhance the agent's understanding or next steps.


Potential improvements or corrections to the agent's approach based on the search results."""


VETTING_PROMPT_4_SCHOLARLY_ARTICLES = """Review the following agent step and search for Scholarly Articles:


Step Tag: <Step Tag>

Prior Results Tag: <Prior Results Tag> (if applicable)

Thought Tag: <Thought Tag>

Action Tag: <Action Tag>

Observation Tag: <Observation Tag>


Search for scholarly articles related to the agent's task, the methods used in the 'Action', or the concepts discussed in the 'Thought' and 'Observation'. Focus on finding academic papers that provide:


Theoretical background and foundations for the agent's approach.


Empirical evidence supporting or challenging the agent's assumptions or actions.


Deeper insights and analyses related to the task or domain.


List any relevant scholarly articles found, including titles, authors (if easily available), and a brief explanation of their relevance to the agent's step. Explain what kind of insights or validation these articles provide. If no relevant articles are found, explicitly state that no relevant scholarly articles were found."""



CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Vetting Agent. Your task is to critically review the thinking process of a primary agent and provide feedback and identify relevant external resources to enhance its reasoning and actions. You will be provided with the primary agent's 'Thought', 'Action', 'Observation', 'Step Tag', and 'Prior Results Tag' (if available). Your goal is to perform specific vetting searches as instructed and provide insightful and actionable feedback.

<|eot_id|><|start_header_id|>user<|end_header_id|>

[Vetting Prompt - Choose one of the prompts below and populate with relevant tags]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[Vetting Agent's Response - Feedback, identified resources, and suggestions]

<|eot_id|>""" 