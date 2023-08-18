import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('🦜️🔗 Presentation Script Maker')
prompt = st.text_input('Plug in your prompt here') # text box for user input

# Prompt templates
title_template = PromptTemplate(
    input_variables =['topic'],
    template = 'Write me a presentation about {topic}'
)

# LLM code
llm = OpenAI(temperature=0.9) # level of creativity
title_chain = LLMChain(llm=llm, prompt=title_template)

# to use the template we will need a LLM chain to chain everything together



if prompt:
    response = title_chain.run(topic = prompt)
    st.write(response) # write back the text


