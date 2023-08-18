import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

st.title('🦜️🔗 Presentation Script Maker')
prompt = st.text_input('Plug in your prompt here') # text box for user input

# Prompt templates
title_template = PromptTemplate(
    input_variables =['topic'],
    template = 'Write me a presentation about {topic}'
)

script_template = PromptTemplate(
    input_variables =['title'],
    template = 'Write me a script based around this TITLE: {title}'
)

# Memory
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# LLM code
llm = OpenAI(temperature=0.9) # level of creativity
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)

# verbose is for logging
sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True, input_variables=['topic'], output_variables=['title', 'script']) # specifies the order in which to run our chain



# to use the template we will need a LLM chain to chain everything together



if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title']) # write back the text
    st.write(response['script']) # write back the text

    with st.expander('Message History'):
        st.info(memory.buffer)



