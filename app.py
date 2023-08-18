import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper # api calls to wikipedia

os.environ['OPENAI_API_KEY'] = apikey

st.title('🦜️🔗 Presentation Script Maker')
prompt = st.text_input('Plug in your prompt here') # text box for user input

# Prompt templates
title_template = PromptTemplate(
    input_variables =['topic'],
    template = 'Write me a presentation about {topic}'
)

script_template = PromptTemplate(
    input_variables =['title', 'wikipedia_research'],
    template = 'You are a Management Consultant at Accenture and your audience is at the C-level, give a brief presentation based around this TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}. Adopt a friendly casual tone during the presentation'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLM code
llm = OpenAI(temperature=0.9) # level of creativity
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
wiki = WikipediaAPIWrapper()
# verbose is for logging



# to use the template we will need a LLM chain to chain everything together



if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title= title, wikipedia_research=wiki_research)
    st.write(title) # write back the text
    st.write(script) # write back the text

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research History'):
        st.info(wiki_research)

    



