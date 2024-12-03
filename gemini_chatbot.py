import streamlit as st
import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import InvalidArgument, ResourceExhausted
import json
import os

SAFETY_SETTINGS = {HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

ASSISTANTS = "instructions/assistants.json"


class Assistants:

    def __init__(self, json_path):
        """

        :param json_path:
        """
        self.json_path = json_path
        self.json_data = None
        if not self.json_data:
            self.retrieve_json()

        return

    def retrieve_json(self):
        """
        Opens a json file containing definitions for AI assistants and
        organizes them into a standard Python dictionary.
        """

        valid_schema = False
        if os.path.exists(self.json_path):
            j = open(self.json_path)
            self.json_data = json.load(j)
            valid_schema = True
            if type(self.json_data) == dict:
                for key, val in self.json_data.items():
                    schema = {'icon': False, 'intro': False, 'instructions': False}
                    if type(val) == dict:
                        for s, _ in val.items():
                            if s in schema:
                                schema[s] = True
                    if False in schema.values():
                        valid_schema = False
        if not valid_schema:
            self.json_data = {'Default Gemini Assistant': {'icon': "♊️",
                                                           'intro': ("I'm your default Gemini Assistant, "
                                                                     "how can I help you?"),
                                                           'instructions': None}}

        return

    def list_assistants(self):
        """
        Retrieves a list of assistant names as defined in json
        :return: list of assistant names
        """

        if self.json_data:
            return list(self.json_data.keys())

        return

    def get_instructions(self, key):
        """
        Opens the instructions .txt file for a given assistant
        :param key: name of assistant to retrieve instructions for
        :return: assistant instructions if exists else None
        """

        if self.json_data and key in self.json_data:
            if 'instructions' in self.json_data[key]:
                if os.path.exists(self.json_data[key]['instructions']):
                    return open(self.json_data[key]['instructions'], "r").read()

        return

    def page_title(self, key):
        """
        Constructions a string to be displayed as the page title using
        the assistant name and icon if one exists
        :param key: name of assistant
        :return: string page title
        """

        page_title = ""
        if self.json_data and key in self.json_data:
            if 'icon' in self.json_data[key]:
                page_title += self.json_data[key]['icon']
        if page_title:
            return page_title + " " + key

        return key

    def get_intro(self, key):
        """
        Retrieves the introduction text for a given assistant that is
        used as the first message in a new chat
        :param key: name of assistant
        :return: string introduction text
        """

        if self.json_data and key in self.json_data:
            if 'intro' in self.json_data[key]:
                return self.json_data[key]['intro']

        return


def api_config():
    """
    loads the Google genai api key and configures the service
    :return:
    """

    if 'GOOGLE_API_KEY' in st.secrets:
        genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
        try:
            genai_model_names = {}
            for m in genai.list_models():
                if ("Gemini" in m.display_name
                        and "1.0" not in m.display_name
                        and "Tuning" not in m.display_name
                        and ("002" in m.display_name or "001" in m.display_name)):
                    genai_model_names[m.display_name] = m
        except InvalidArgument:
            st.warning("Configuration failed. API key not valid. Please pass a valid API key.")
            st.stop()
        st.session_state['models'] = genai_model_names
        if not st.session_state.models:
            st.warning('Failed to retrieve any models')
            st.stop()
    else:
        st.warning('Configuration failed. Missing a an API key.')
        st.stop()


class Model:

    def __init__(self,
                 model_name="gemini-1.5-flash-002",
                 max_output_tokens=1024,
                 temperature=0.9,
                 top_p=0.95,
                 top_k=32,
                 presence_penalty=0,
                 frequency_penalty=0,
                 safety_settings=None,
                 instructions=None):
        """

        :param model_name:
        :param max_output_tokens:
        :param temperature:
        :param top_p:
        :param top_k:
        :param presence_penalty:
        :param frequency_penalty:
        :param safety_settings:
        :param instructions:
        """

        # api_config()
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.safety_settings = safety_settings
        self.instructions = instructions
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=self.model_config(),
                                           safety_settings=self.safety_settings,
                                           system_instruction=self.instructions)

    def model_config(self):
        """
        Create a configuration file for the model
        :return:
        """

        return genai.GenerationConfig(max_output_tokens=self.max_output_tokens,
                                      temperature=self.temperature,
                                      top_p=self.top_p,
                                      top_k=self.top_k,
                                      presence_penalty=self.presence_penalty,
                                      frequency_penalty=self.frequency_penalty
                                      )

    def start_chat(self):
        """
        Start a new chat session
        :return:
        """

        return self.model.start_chat()


# ==================
# ===== MODELS =====
# ==================

if 'models' not in st.session_state:
    api_config()

# ==================
# === ASSISTANTS ===
# ==================

if 'assistants' not in st.session_state:
    st.session_state['assistants'] = Assistants(ASSISTANTS)
    if not st.session_state.assistants.list_assistants():
        st.warning('Failed to retrieve any assistants')
        st.stop()

# ===============
# === SIDEBAR ===
# ===============

with (st.sidebar):
    # Select the Assistant to use, path to an instructions .txt file
    st.selectbox(label='Assistant name',
                 options=st.session_state.assistants.list_assistants(),
                 key="assistant_name"
                 )

    # Select the gemini model to use
    st.selectbox(label='Model variant',
                 # options=['gemini-1.5-flash-001',
                 #          'gemini-1.5-flash-latest',
                 #          'gemini-1.5-pro-001',
                 #          'gemini-1.5-pro-latest',
                 #          ],
                 options=st.session_state.models.keys(),
                 key="model_name"
                 )

    # Set the model's max output between 64 and 8192
    st.slider(label='Max output tokens',
              min_value=64,
              max_value=st.session_state.models[st.session_state.model_name].output_token_limit,
              value=2048,
              step=64,
              key='max_output_tokens'
              )

    # Set the model's temperature between 0 and 2
    st.slider(label='Temperature',
              min_value=0.0,
              max_value=st.session_state.models[st.session_state.model_name].max_temperature,
              value=st.session_state.models[st.session_state.model_name].temperature,
              step=0.05,
              key='temperature'
              )

    # Set the model's top p between 0 and 1
    st.slider(label='Top p',
              min_value=0.0,
              max_value=1.0,
              value=st.session_state.models[st.session_state.model_name].top_p,
              step=0.05,
              key='top_p'
              )

    # Set the model's top k between 1 and 40 or 64 depending on the model
    st.slider(label='Top k',
              min_value=1,
              max_value=st.session_state.models[st.session_state.model_name].top_k,
              value=st.session_state.models[st.session_state.model_name].top_k,
              step=1,
              key='top_k'
              )

    # A few models support penalty parameters, if a supported model is selected, show the penalty sliders
    if (st.session_state.models[st.session_state.model_name].name in
            ("models/gemini-1.5-pro-002", "models/gemini-1.5-flash-002", "models/gemini-1.5-flash-8b-001")):
        # Set the model's presence penalty between -2 and 2
        st.slider(label='Presence penalty',
                  min_value=-2.0,
                  max_value=1.99,
                  value=0.0,
                  step=0.01,
                  key='presence_penalty'
                  )

        # Set the model's frequency penalty between -2 and 2
        st.slider(label='Frequency penalty',
                  min_value=-2.0,
                  max_value=1.99,
                  value=0.0,
                  step=0.01,
                  key='frequency_penalty'
                  )

    # Start a new chat with selected sidebar options as parameters
    if new_chat := st.button(label="New Chat"):
        st.session_state['page_title'] = st.session_state.assistants.page_title(st.session_state.assistant_name)
        # remove all previous messages from the chat
        st.session_state.pop('messages', None)
        # retrieve system instructions for selected assistant
        assistant_instructions = st.session_state.assistants.get_instructions(st.session_state.assistant_name)
        # start a new chat with the selected parameters
        st.session_state.chat = Model(model_name=st.session_state.models[st.session_state.model_name].name,
                                      max_output_tokens=st.session_state.max_output_tokens,
                                      temperature=st.session_state.temperature,
                                      top_p=st.session_state.top_p,
                                      top_k=st.session_state.top_k,
                                      presence_penalty=(0 if 'presence_penalty' not in st.session_state
                                                        else st.session_state.presence_penalty),
                                      frequency_penalty=(0 if 'frequency_penalty' not in st.session_state
                                                         else st.session_state.frequency_penalty),
                                      safety_settings=SAFETY_SETTINGS,
                                      instructions=assistant_instructions
                                      ).start_chat()

        try:
            # send a message (hidden message) to the assistant asking for an introduction
            response = st.session_state.chat.send_message(("Give me a brief overview of what you can help me with, "
                                                           "provide a short description of what you have been "
                                                           "instructed to do. "
                                                           "List in bullet format a few of the most important "
                                                           "things that I can expect you to do for me."))
            introduction = response.text

        except ResourceExhausted:
            st.warning(("Resource has been exhausted (e.g. check quota). "
                        "Wait 1 minute and try sending your message again."
                        ))
            st.stop()
            # introduction = st.session_state.assistants.get_intro(st.session_state.assistant_name)
        except InvalidArgument as ia:
            st.warning(ia.message)
            st.stop()
        except StopCandidateException as sce:
            st.warning(sce)
            st.stop()
        # add messages to the session state
        st.session_state["messages"] = [{"role": "assistant", "content": introduction}]
        # st.rerun()

    # clear current chat
    if 'chat' in st.session_state:
        if clear_chat := st.button(label="Clear Chat"):
            st.session_state.pop('messages', None)
            st.session_state.pop('chat', None)
            st.rerun()

# ==================
# === APP INTRO ====
# ==================

if "chat" not in st.session_state:
    st.session_state['page_title'] = "Google Gemini Chat Assistants"
    # display the title on the page
    st.title(st.session_state.page_title)
    # introduce the application and provide instructions
    st.markdown("Use the left sidebar to get started with a new chat.\n"
                "* **Assistant name**: An AI agent/assistant/persona that has be given "
                "instructions to perform a specific task."
                "\n\n"
                "* **Model variants**: "
                "The Gemini API offers different models that are optimized for specific use cases. "
                "Here's a brief overview of Gemini variants that are available:\n"
                "   - Gemini 1.5 Flash: Fast and versatile performance across a diverse variety of tasks\n"
                "   - Gemini 1.5 Flash-8B: High volume and lower intelligence tasks\n"
                "   - Gemini 1.5 Pro: Complex reasoning tasks requiring more intelligence"
                "\n\n"
                "* **Max output tokens**: The maximum number of tokens to include in a response candidate."
                "\n\n"
                "* **Temperature**: Controls the randomness of the output. "
                "A higher value will produce responses that are more varied, "
                "while a value closer to 0.0 will typically result in less surprising responses from the model. "
                "This value specifies default to be used by the backend while making the call to the model."
                "\n\n"
                "* **Top p**: The maximum cumulative probability of tokens to consider when sampling. "
                "The model uses combined Top-k and Top-p (nucleus) sampling. "
                "Tokens are sorted based on their assigned probabilities "
                "so that only the most likely tokens are considered. "
                "Top-k sampling directly limits the maximum number of tokens to consider, "
                "while Nucleus sampling limits the number of tokens based on the cumulative probability."
                "\n\n"
                "* **Top k**: The maximum cumulative probability of tokens to consider when sampling. "
                "The model uses combined Top-k and Top-p (nucleus) sampling. "
                "Tokens are sorted based on their assigned probabilities "
                "so that only the most likely tokens are considered. "
                "Top-k sampling directly limits the maximum number of tokens to consider, "
                "while Nucleus sampling limits the number of tokens based on the cumulative probability."
                "\n\n"
                "* **Presence penalty**: Presence penalty applied to the next token's logprobs "
                "if the token has already been seen in the response. "
                "This penalty is binary on/off and not dependant on the number of times "
                "the token is used (after the first)."
                "\n\n"
                "* **Frequency penalty**: Frequency penalty applied to the next token's logprobs, "
                "multiplied by the number of times each token has been seen in the response so far. "
                "A positive penalty will discourage the use of tokens that have already been used, "
                "proportional to the number of times the token has been used: "
                "The more a token is used, the more difficult it is for the model to use that token again "
                "increasing the vocabulary of responses."
                )

# ==================
# === BEGIN CHAT ===
# ==================

elif "messages" in st.session_state:

    # display the title on the page
    st.title(st.session_state.page_title)
    # display each message in the chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # send a new message to the chat and get a response
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            response = st.session_state.chat.send_message(prompt)
            msg = response.text
        except ResourceExhausted:
            st.warning(("Resource has been exhausted (e.g. check quota). "
                        "Wait 1 minute and try sending your message again."
                        ))
            st.stop()
        except StopCandidateException as sce:
            st.warning(sce)
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
