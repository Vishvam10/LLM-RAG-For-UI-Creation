# !pip install -q -U torch datasets transformers tensorflow langchain playwright html2text sentence_transformers faiss-cpu
# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7

import os
import torch
import time

from data_loader import get_vector_embeddings
from utils import Color

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

import warnings

warnings.filterwarnings('ignore')

# #################################################################
# Utils
# #################################################################

def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        msg = Color.apply(f'\n[PROFILER] : Function "{func.__name__}" took : {execution_time:.4f} seconds\n', 'cyan')
        print(msg)
        return result

    return wrapper


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param}'
    )


ml_model_name = './models/model_mistral_7b_instruct_v01'
tokenizer_model_name = './models/tokenizer_mistral_7b'
rag_name = 'bootsrap_rag'

# #################################################################
# Init Tokenizer and LLM
# #################################################################

@profile
def init_tokenizer():

    print('\n[DEBUG] : Loading tokenizer ...\n')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return tokenizer

@profile
def init_llm():

    # With low_cpu_mem_usage=False, the time taken is ~75s, with low_cpu_mem_usage=True, the time taken is ~7s

    print('\n[DEBUG] : Loading model ...\n')
    model = AutoModelForCausalLM.from_pretrained(ml_model_name, low_cpu_mem_usage=True)

    print(print_trainable_parameters(model))

    return model

@profile
def init_pipeline(model, tokenizer):

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        repetition_penalty=1.0,
        return_full_text=True,
        max_new_tokens=1000,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return mistral_llm


# #################################################################
# RAG Setup
# #################################################################

@profile
def init_rag(rag_name):

    print('\n[DEBUG] : Loading RAG ...\n')
    db = get_vector_embeddings(article_name=rag_name)
    retriever = db.as_retriever()
    print(retriever)

    return retriever


#################################################################
# Prompt Template Setup
#################################################################

@profile
def init_prompt_template():

    print('\n[DEBUG] : Setting prompt template ...\n')

    prompt_template = '''
    ### [INST] Instruction: You are a UI generating assistant. You'll carefully analyse the input prompt and then create the UI using React and CSS3. Use CSS classes wherever you can and define them in a styles tag. Infer things like background colors, shadows, borders, etc from the nature of the UI component. Follow the input prompt thoroughly. Print only the code that is asked in the input prompt and nothing else (no explanation or comments or things like 'here you go', 'here's the code that you asked for', etc)

    {context}

    ### QUESTION:
    {question} [/INST]
    '''

    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=prompt_template,
    )

    return prompt

@profile
def init_rag_chain(llm_pipeline, retriever, prompt):

    # Create llm chain
    llm_chain = LLMChain(llm=llm_pipeline, prompt=prompt)

    rag_chain = ({'context': retriever, 'question': RunnablePassthrough()} | llm_chain)

    return rag_chain

@profile
def test_rag_chain(rag_chain, question):

    result = rag_chain.invoke(question)
    try:
        f = open('output.txt', 'w')
        f.write(str(result['text']))
        f.close()
    except:
        pass
    finally:
        print(result['text'])


if __name__ == '__main__':

    tokenizer = init_tokenizer()
    model = init_llm()
    llm_pipeline = init_pipeline(model, tokenizer)
    rag = init_rag(rag_name)

    prompt_template = init_prompt_template()
    rag_chain = init_rag_chain(llm_pipeline, rag, prompt_template)

    question= '''
    Generate a button
    '''

    test_rag_chain(rag_chain, question)

