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

# Define the LLM chat model
chat = ChatOpenAI(temperature=0.9)

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

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
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Layout of input/response containers
## container for chat history
response_container = st.container()
## container for text box
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')


# Define function to start a new chat
def new_chat(response_container):
    with response_container:
        if st.session_state['generated']:
            """
            Clears session state and starts a new chat.
            """
            save = []
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
                #st.write("Test") #TODO Add sources here
                save.append("User:" + st.session_state["past"][i])
                save.append("Bot:" + st.session_state["generated"][i])        
            st.session_state["stored_session"].append(save)
            st.session_state["generated"] = []
            st.session_state["past"] = []
            st.session_state["input"] = ""
            st.session_state.entity_memory.entity_store = {}
            st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        with st.expander("Memory-Store", expanded=False):
            st.session_state.entity_memory.store
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        with st.expander("Bufffer-Store", expanded=False):
            st.session_state.entity_memory.buffer
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
st.title("🤖 Chat Bot with 🧠")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
#API_O = st.sidebar.text_input("API-KEY", type="password")
API_O = "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"

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat(response_container), type='primary')

# Let user clear the current conversation
clear_button = st.sidebar.button("Clear Conversation", key="clear")
# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []


## Applying the user input box
with input_container:
    # Get the user input
    user_input = st.text_input("You: ", st.session_state["input"], key="input")

with response_container:
    if user_input:
        output = Conversation.run(input=user_input)  
        st.session_state.past.append(user_input)  
        st.session_state.generated.append(output)

               






# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)


# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session


# # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
# st.sidebar.title("Sidebar")
# clear_button = st.sidebar.button("Clear Conversation", key="clear")

# # reset everything
# if clear_button:
#     st.session_state['generated'] = []
#     st.session_state['past'] = []


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


# # Define the conversation chain
# conversation = ConversationChain(
#     llm=chat,
#     memory=ConversationBufferMemory() # memory_key defaults to 'history'
# )

# # Layout of input/response containers
# ## container for chat history
# response_container = st.container()
# ## container for text box
# input_container = st.container()
# colored_header(label='', description='', color_name='blue-30')

# ## Applying the user input box

# with input_container:
#     user_input = st.text_input("You: ", "", key="input")

# with response_container:

#     if user_input:
#         response = llm_chain.run(user_input)
#         st.session_state['past'].append(user_input)
#         st.session_state['generated'].append(response)

#     if st.session_state['generated']:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
#             message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
#             st.write("Test") #TODO Add sources here


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