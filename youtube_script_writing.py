from google import genai
from google.genai import types
from google.genai import errors
import streamlit as st
import json


response_schema = {
    "type": "OBJECT",
    "properties": {
        "hook": {"type": "OBJECT",
                 "properties": {"dialogue": {"type": "STRING"},
                                "visuals": {"type": "STRING"},
                                "notes": {"type": "STRING"}
                                },
                 "required": ["dialogue", "visuals", "notes"]
                 },
        "scenes_a": {"type": "ARRAY",
                     "items": {
                         "type": "OBJECT",
                         "properties": {"setup": {"type": "OBJECT",
                                                  "properties": {"dialogue": {"type": "STRING"},
                                                                 "visuals": {"type": "STRING"},
                                                                 "notes": {"type": "STRING"}
                                                                 },
                                                  "required": ["dialogue", "visuals", "notes"]
                                                  },
                                        "payoff": {"type": "OBJECT",
                                                   "properties": {"dialogue": {"type": "STRING"},
                                                                  "visuals": {"type": "STRING"},
                                                                  "notes": {"type": "STRING"}
                                                                  },
                                                   "required": ["dialogue", "visuals", "notes"]
                                                   }
                                        },
                         "required": ["setup", "payoff"]
                     }
                     },
        "mid_call_to_action": {"type": "OBJECT",
                               "properties": {"dialogue": {"type": "STRING"},
                                              "visuals": {"type": "STRING"},
                                              "notes": {"type": "STRING"}
                                              },
                               "required": ["dialogue", "visuals", "notes"]
                               },
        "scenes_b": {"type": "ARRAY",
                     "items": {
                         "type": "OBJECT",
                         "properties": {"setup": {"type": "OBJECT",
                                                  "properties": {"dialogue": {"type": "STRING"},
                                                                 "visuals": {"type": "STRING"},
                                                                 "notes": {"type": "STRING"}
                                                                 },
                                                  "required": ["dialogue", "visuals", "notes"]
                                                  },
                                        "payoff": {"type": "OBJECT",
                                                   "properties": {"dialogue": {"type": "STRING"},
                                                                  "visuals": {"type": "STRING"},
                                                                  "notes": {"type": "STRING"}
                                                                  },
                                                   "required": ["dialogue", "visuals", "notes"]
                                                   }
                                        },
                         "required": ["setup", "payoff"]
                     }
                     },
        "end_call_to_action": {"type": "OBJECT",
                               "properties": {"dialogue": {"type": "STRING"},
                                              "visuals": {"type": "STRING"},
                                              "notes": {"type": "STRING"}
                                              },
                               "required": ["dialogue", "visuals", "notes"]
                               },
    },
    "required": ["hook", "scenes_a", "mid_call_to_action", "scenes_b", "end_call_to_action"]
}

# SYS_INST = 'I give you a topic for a Youtube video, you write a script. Scenes a and b should have multiple scenes.'

SYS_INST = 'I give you a draft script for a Youtube video, you parse the script into your response schema. Preserve any formatting.'

client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])

model_config = types.GenerateContentConfig(safety_settings=None,
                                           system_instruction=SYS_INST,
                                           # max_output_tokens=2048,
                                           temperature=1.0,
                                           top_p=0.6,
                                           top_k=32,
                                           presence_penalty=0.3,
                                           frequency_penalty=0.3,
                                           response_mime_type='application/json',
                                           response_schema=response_schema
                                           )

st.title("🎥 Youtube Script Writing Assistant")

st.text_area(label="Script", key='my_youtube_script', placeholder='Write or paste your script here...')

if st.button(label='Submit', key='submit'):

    chat = client.chats.create(model='gemini-2.0-flash-exp', config=model_config)

    try:
        response = chat.send_message(st.session_state.my_youtube_script)
    except errors.ClientError as ce:
        print(ce.message)
    else:
        try:
            res_json = json.loads(response.text)
        except Exception as e:
            st.error(e)
            st.text(response.model_dump_json())
        else:
            scene_cnt = 0

            st.header("Hook", divider=True)
            # st.text(res_json['hook'])
            st.subheader('Dialogue: ')
            st.write(res_json['hook']['dialogue'])
            st.subheader('Visuals: ')
            st.write(res_json['hook']['visuals'])
            st.subheader('Notes: ')
            st.write(res_json['hook']['notes'])

            for scene in res_json['scenes_a']:
                scene_cnt += 1
                st.header(f'Scene {scene_cnt}', divider=True)

                st.subheader('Setup')

                st.markdown('### Dialogue: ')
                st.write(scene['setup']['dialogue'])

                st.markdown('### Visuals: ')
                st.write(scene['setup']['visuals'])

                st.markdown('### Notes: ')
                st.write(scene['setup']['notes'])

                st.subheader('Payoff')

                st.markdown('### Dialogue: ')
                st.write(scene['payoff']['dialogue'])

                st.markdown('### Visuals: ')
                st.write(scene['payoff']['visuals'])

                st.markdown('### Notes: ')
                st.write(scene['payoff']['notes'])

            st.header("Mid CTA", divider=True)
            # st.text(res_json['mid_call_to_action'])
            st.subheader('Dialogue: ')
            st.write(res_json['mid_call_to_action']['dialogue'])
            st.subheader('Visuals: ')
            st.write(res_json['mid_call_to_action']['visuals'])
            st.subheader('Notes: ')
            st.write(res_json['mid_call_to_action']['notes'])

            for scene in res_json['scenes_b']:
                scene_cnt += 1

                st.header(f'Scene {scene_cnt}', divider=True)

                st.subheader('## Setup')

                st.markdown('### Dialogue: ')
                st.write(scene['setup']['dialogue'])

                st.markdown('### Visuals: ')
                st.write(scene['setup']['visuals'])

                st.markdown('### Notes: ')
                st.write(scene['setup']['notes'])

                st.subheader('## Payoff')

                st.markdown('### Dialogue: ')
                st.write(scene['payoff']['dialogue'])

                st.markdown('### Visuals: ')
                st.write(scene['payoff']['visuals'])

                st.markdown('### Notes: ')
                st.write(scene['payoff']['notes'])

            st.header("End CTA", divider=True)
            # st.text(res_json['end_call_to_action'])
            st.subheader('Dialogue: ')
            st.write(res_json['end_call_to_action']['dialogue'])
            st.subheader('Visuals: ')
            st.write(res_json['end_call_to_action']['visuals'])
            st.subheader('Notes: ')
            st.write(res_json['end_call_to_action']['notes'])
