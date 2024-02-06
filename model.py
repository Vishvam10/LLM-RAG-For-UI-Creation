from transformers import AutoModelForCausalLM

model_name ='mistralai/Mistral-7B-Instruct-v0.1'
model_path = './models/model_mistral_7b_instruct_v01'

llm = AutoModelForCausalLM.from_pretrained(model_name)

llm.save_pretrained(model_path)