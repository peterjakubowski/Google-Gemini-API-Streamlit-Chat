import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors
import json
import os
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv


# =====================
# === Constant Vars ===
# =====================

SAFETY_SETTINGS = [types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                       threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                   types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                       threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                   types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                       threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                   types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                       threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH)]

ASSISTANTS = "instructions/assistants.json"

TOOLS = {'Code Execution': [types.Tool(code_execution=types.ToolCodeExecution())],
         'Google Search': [types.Tool(google_search=types.GoogleSearch())]
         }

# ========================
# === Assistants Class ===
# ========================


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
            if isinstance(self.json_data, dict):
                for key, val in self.json_data.items():
                    schema = {'icon': False, 'intro': False, 'instructions': False}
                    if isinstance(val, dict):
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

    def get_doc_string(self):
        """
        Constructs a string with the names and descriptions of all available assistants.
        :return: string
        """

        doc_string = ("* **Assistant name**: An AI agent/assistant/persona that has been given "
                      "instructions to perform a specific task.\n")
        assistant_intros = []
        for assist in self.list_assistants():
            assistant_intros.append(f'    * *{assist}*: {self.get_intro(assist)}')

        return doc_string + "\n".join(assistant_intros)

# ========================
# === Helper Functions ===
# ========================


def api_config():
    """
    loads the Google genai api key and configures the service
    :return:
    """

    if st.secrets and 'GOOGLE_API_KEY' in st.secrets:
        # configure a Gemini Client with API key
        client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])
    elif load_dotenv():
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    else:
        st.warning('Configuration failed. Missing a an API key.')
        st.stop()

    try:
        genai_model_names = {}
        for m in client.models.list():
            if ("Gemini" in m.display_name
                    and "1.0" not in m.display_name
                    and "Tuning" not in m.display_name
                    and ("002" in m.display_name or "001" in m.display_name or "Exp" in m.display_name)):
                genai_model_names[m.display_name] = m

    except errors.ClientError as ce:
        st.warning(f"{ce.code} {ce.status}: {ce.message}")
        st.stop()

    st.session_state['client'] = client
    st.session_state['models'] = genai_model_names
    if not st.session_state.models:
        st.warning('Failed to retrieve any models')
        st.stop()


def process_message(_part: types.Part) -> None:
    """
    Add content parts to the chat and display them.

    :param _part: Message to add to the chat
    :return: None
    """

    assert 'messages' in st.session_state, "Cannot find messages in the session state."

    _content = []

    if _part.text is not None:
        _content.append(_part.text)
    if _part.executable_code is not None:
        _content.append(_part.executable_code.code)
    if _part.code_execution_result is not None:
        _content.append(_part.code_execution_result.output)
    if _part.inline_data is not None:
        img = Image.open(BytesIO(part.inline_data.data))
        _content.append(img)

    for c in _content:
        # add the message to the chat
        st.session_state.messages.append({"role": "assistant", "content": c})
        # display the message in the chat
        st.chat_message("assistant").write(c)

    return

# ===================
# === Model Class ===
# ===================


class Model:

    def __init__(self,
                 model_name="gemini-2.0-flash-001",
                 max_output_tokens=1024,
                 temperature=0.9,
                 top_p=0.95,
                 top_k=32,
                 presence_penalty=0,
                 frequency_penalty=0,
                 safety_settings=None,
                 instructions=None,
                 tools=None):
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
        :param tools:
        """

        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.safety_settings = safety_settings
        self.instructions = instructions
        self.tools = tools
        self.model = st.session_state['client'].chats.create(model=self.model_name,
                                                             config=self.model_config())

    def model_config(self):
        """
        Create a configuration file for the model
        :return:
        """
        return types.GenerateContentConfig(max_output_tokens=self.max_output_tokens,
                                           temperature=self.temperature,
                                           top_p=self.top_p,
                                           top_k=self.top_k,
                                           presence_penalty=self.presence_penalty,
                                           frequency_penalty=self.frequency_penalty,
                                           safety_settings=self.safety_settings,
                                           system_instruction=self.instructions,
                                           tools=TOOLS[self.tools] if self.tools else None,
                                           tool_config=types.ToolConfig(
                                               function_calling_config=types.FunctionCallingConfig(
                                                   mode=types.FunctionCallingConfigMode("AUTO"))),
                                           automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                               disable=False,
                                               maximum_remote_calls=10),
                                           response_modalities=["Text", "Image"]
                                           )


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
                 options=sorted(st.session_state.models.keys()),
                 key="model_name",
                 index=0
                 )

    # Select tools
    st.selectbox(label='Tools',
                 options=sorted(TOOLS.keys()),
                 key="tools",
                 index=None)

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
              max_value=2.0,
              # max_value=st.session_state.models[st.session_state.model_name].max_temperature,
              value=0.90,
              # value=st.session_state.models[st.session_state.model_name].temperature,
              step=0.05,
              key='temperature'
              )

    # Set the model's top p between 0 and 1
    st.slider(label='Top p',
              min_value=0.0,
              max_value=1.0,
              value=0.95,
              # value=st.session_state.models[st.session_state.model_name].top_p,
              step=0.05,
              key='top_p'
              )

    # Set the model's top k between 1 and 40 or 64 depending on the model
    st.slider(label='Top k',
              min_value=1,
              max_value=40,
              # max_value=st.session_state.models[st.session_state.model_name].top_k,
              value=40,
              # value=st.session_state.models[st.session_state.model_name].top_k,
              step=1,
              key='top_k'
              )

    # A few models support penalty parameters, if a supported model is selected, show the penalty sliders
    if (st.session_state.models[st.session_state.model_name].name not in
            ("models/gemini-1.5-pro-001", "models/gemini-1.5-flash-001")):
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
                                      instructions=assistant_instructions,
                                      tools=st.session_state.tools
                                      ).model

        # add messages to the session state
        introduction = st.session_state.assistants.get_intro(key=st.session_state.assistant_name)
        st.session_state["messages"] = [{"role": "assistant", "content": introduction}]

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
    st.markdown("Use the left sidebar to get started with a new chat.")
    st.divider()
    # list all the assistants with a short description
    st.markdown(st.session_state.assistants.get_doc_string())
    st.divider()
    # display the documentation for model parameters
    if os.path.exists(params_doc := "instructions/params_doc.md"):
        with open(params_doc) as doc:
            st.markdown(doc.read())

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
    if prompt := st.chat_input(accept_file="multiple", file_type=["jpg", "png"]):
        # add prompt text to messages and display it in the chat
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        st.chat_message("user").write(prompt.text)
        # start a list of messages to send to the chat model
        content = [prompt.text]
        # process files included in the prompt
        if prompt.files:
            # loop through the list of files
            for file in prompt.files:
                # check if the file is an image, that's all we'll accept right now
                if file.type in ('image/jpeg', 'image/png'):
                    # try to open the uploaded file as an image with Pillow
                    try:
                        image = Image.open(file)
                    except FileNotFoundError:
                        st.warning(f"Error: Image file not found {file.name}")
                    except Image.UnidentifiedImageError:
                        st.warning(f"Error: Cannot identify image file {file.name}")
                    except IOError:
                        st.warning(f"Error: An I/O error occurred when opening {file.name}")
                    except ValueError:
                        st.warning("Error: Invalid mode or file path.")
                    # add the open image to the chat, display it, and append it to our list of prompt content
                    else:
                        st.session_state.messages.append({"role": "user", "content": image})
                        st.chat_message("user").write(image)
                        content.append(image)

        # try to get a response from the chat model with your prompt content
        try:
            response = st.session_state.chat.send_message(message=content)

        except errors.ClientError as ce:
            st.warning(f'{ce.code} {ce.status}: {ce.message}')
            st.stop()

        except errors.APIError as ae:
            st.warning(ae.message)
            st.stop()

        # loop through all the parts in the response and display them in the chat
        for part in response.candidates[0].content.parts:
            try:
                process_message(part)
            except AssertionError as ae:
                st.warning(ae.__str__())
