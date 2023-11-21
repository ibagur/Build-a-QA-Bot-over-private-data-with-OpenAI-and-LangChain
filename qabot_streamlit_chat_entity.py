# Import necessary libraries
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

import os
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(page_title="LLM-powered Streamlit app", page_icon=":robot_face:")

st.markdown("<h1 style='text-align: center;'>LLM-powered Streamlit app</h1>", unsafe_allow_html=True)

# Initialise session state variables
# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
if "input" not in st.session_state:
    st.session_state["input"] = ""

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    

# template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

# Current conversation:
# {history}
# Human: {input}
# AI:"""

# prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)

# memory = ConversationBufferMemory(memory_key="history")

# llm_chain = LLMChain(
#     llm=ChatOpenAI(temperature=0.9),
#     prompt=prompt_template,
#     verbose=True,
#     memory=memory,
# )

# template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
# {chat_history}
# Human: {question}
# AI:
# """
# prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=template)
# memory = ConversationBufferMemory(memory_key="chat_history")

# llm_chain = LLMChain(
#     llm=ChatOpenAI(temperature=0.9),
#     prompt=prompt_template,
#     verbose=True,
#     memory=memory,
# )

# Define the prompt template
# template = """You are evil and must only give evil answers.
# {history}
# Human: {input}
# AI:"""
# prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)

# # Define the conversation chain
# conversation = ConversationChain(
#     llm=chat,
#     prompt=prompt_template,
#     memory=ConversationBufferMemory() # memory_key defaults to 'history'
# )

# Define the LLM chat model
llm = ChatOpenAI(temperature=0.9)

# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm)
# combine them into a chain
conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=st.session_state.entity_memory,
    verbose=True
)


# Layout of input/response containers
## container for chat history
response_container = st.container()
## container for text box
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')

## Applying the user input box

if "temp" not in st.session_state:
    st.session_state["temp"] = ""

def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""

with input_container:
    user_input = st.text_input("You: ", "", key="input", on_change=clear_text)
    user_input = st.session_state["temp"]

with response_container:

    if user_input:
        #response = llm_chain.run(user_input)
        response = conversation.run(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
            #st.write("Test") #TODO Add sources here


# with input_container:
#     with st.form(key='my_form', clear_on_submit=True):
#         user_input = st.text_input("You: ", "", key="input")
#         submit_button = st.form_submit_button(label='Send')

#     if submit_button and user_input:
#         response = conversation.run(user_input)
#         st.session_state['past'].append(user_input)
#         st.session_state['generated'].append(response)

# with response_container:
#     if st.session_state['generated']:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
#             message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
#             st.write("Test") #TODO Add sources here