from jinja2 import Template
# from typing import Literal
from pydantic import BaseModel
from google import genai
from google.genai import types
# from google.genai import errors
import streamlit as st


# import json


class AgentResponseSchema(BaseModel):
    agent_agrees: str  # True or False
    current_answer: str


AGENT_A_SYS_INST = Template("""You are 'research_agent_a' and you must answer a question creatively.
You are an imaginative agent.
You like to express your ideas, even if there's a chance others won't like them.
You are working with 'research_agent_b' and must come to an agreement.
You are given a question, the current answer, and a history of all previously given answers.
Do you agree with the current answer?
If you agree with the current answer, set 'agent_agrees' to 'True' and return the current answer.
If you do not agree with the current answer, set 'agent_agrees' to 'False' and return a new answer.
""")

AGENT_B_SYS_INST = Template("""You are 'research_agent_b' and you must answer a question methodically.
You are a facts-based agent.
You need proof that an answer is correct.
You miss no detail left unchecked.
You are an expert editor and know how to take a good idea and make it even better.
You are quick to notice mistakes and always ready to question the validity of an answer.
You are working with 'research_agent_a' and must come to an agreement.
You are given a question, the current answer, and a history of all previously given answers.
Do you agree with the current answer?
If you agree with the current answer, set 'agent_agrees' to 'True' and return the current answer.
If you do not agree with the current answer, set 'agent_agrees' to 'False' and return a new answer.
""")

AGENT_A_CONFIG = types.GenerateContentConfig(safety_settings=None,
                                             system_instruction=AGENT_A_SYS_INST.render(),
                                             # tools=[set_agreement_agent_a, set_agreement_agent_b],
                                             max_output_tokens=2048,
                                             temperature=2.0,
                                             top_p=0.6,
                                             top_k=32,
                                             presence_penalty=0.0,
                                             frequency_penalty=0.0,
                                             response_mime_type='application/json',
                                             response_schema=AgentResponseSchema

                                             )

AGENT_B_CONFIG = types.GenerateContentConfig(safety_settings=None,
                                             system_instruction=AGENT_B_SYS_INST.render(),
                                             # tools=[set_agreement_agent_a, set_agreement_agent_b],
                                             max_output_tokens=2048,
                                             temperature=2.0,
                                             top_p=0.6,
                                             top_k=32,
                                             presence_penalty=0.0,
                                             frequency_penalty=0.0,
                                             response_mime_type='application/json',
                                             response_schema=AgentResponseSchema

                                             )

client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])

content = Template("""# Question

{{question}}

# Current Answer

{{current_answer}}

# Answer History

{{history}}
""")


def research_agent_a(question: str, current_answer: str, history: str) -> BaseModel:
    _response = client.models.generate_content(model='models/gemini-2.0-flash-lite-preview-02-05',
                                               contents=[content.render(question=question,
                                                                        current_answer=current_answer,
                                                                        history=history)],
                                               config=AGENT_A_CONFIG)

    _result = _response.parsed
    # _result.agreement_agent_a = 'True'
    # print("\n\n###SUMMARY###")
    # print(content.render(question=question,
    #                      current_answer=current_answer,
    #                      history=history))
    # print("\n\n####RESPONSE##")
    print(f"[Agent A]: {_result}")

    return _result


def research_agent_b(question: str, current_answer: str, history: str) -> BaseModel:
    _response = client.models.generate_content(model='gemini-2.0-flash-lite-preview-02-05',
                                               contents=[content.render(question=question,
                                                                        current_answer=current_answer,
                                                                        history=history)],
                                               config=AGENT_B_CONFIG)

    _result = _response.parsed
    # _result.agreement_agent_b = 'True'
    # print("\n\n###SUMMARY###")
    # print(content.render(question=question,
    #                      current_answer=current_answer,
    #                      history=history))
    # print("\n\n####RESPONSE##")
    print(f"[Agent B]: {_result}")

    return _result


CHAT_SYSTEM_INSTRUCTIONS = Template("""Generate a response to my questions with the help of your research agents.
You can not answer questions on your own, but can ask your agents to answer questions for you.
The current answer to the question is "This question is unanswered".
You have no opinion, you just report back the answer given by your agents.
When I ask you a question, you pass my question along to your agents for discussion.
You have two agents, 'research_agent_a' and 'research_agent_b'.
The two agents may have differing opinions.
Give my question to 'research_agent_b' agent first.
The two agents should work collaboratively at coming to an answer that they agree on.
If an agent disagrees with the current answer to the question, they should revise the answer to fit their beliefs.
Once an agent has revised an answer, they should return it to the other agent for their approval.
This back and forth should continue until an agent agrees with the current answer.
Check the agents agreement status at every turn.
Provide the agents with a history of previously given answers by summarizing the conversation so far.
To start, the history of previously given answers is set to "None so far, we're just getting started".
Update the history at every turn by summarizing the agent's answers.
If an agent agrees with the answer, then stop and return to me the final answer.

""")

# Configure our Gemini chat model
CHAT_MODEL_CONFIG = types.GenerateContentConfig(safety_settings=None,
                                                tools=[research_agent_a, research_agent_b],
                                                tool_config=types.ToolConfig(
                                                    function_calling_config=types.FunctionCallingConfig(
                                                        mode=types.FunctionCallingConfigMode("AUTO"))),
                                                system_instruction=CHAT_SYSTEM_INSTRUCTIONS.render(),
                                                max_output_tokens=2048,
                                                temperature=1.0,
                                                top_p=0.6,
                                                top_k=32,
                                                presence_penalty=0.3,
                                                frequency_penalty=0.3,
                                                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                                    disable=False,
                                                    maximum_remote_calls=15
                                                )
                                                )


my_prompt = """What is a good character name for a fantasy role playing game? I'm playing as a magical elf.
"""

chat = client.chats.create(model='models/gemini-2.0-flash-001', config=CHAT_MODEL_CONFIG)

while True:
    response = chat.send_message(my_prompt)
    print(f"[Supervisor]: {response.text}")
    my_prompt = input()
    if my_prompt.lower() in ('quit', 'q', 'exit'):
        break
