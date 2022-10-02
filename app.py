import streamlit as st
import numpy as np
import pandas as pd
import cohere
from cohere.classify import Example

COHERE_KEY = open("cohere-key.txt", "r").readlines(0)[0][:-1]

co = cohere.Client(f'{COHERE_KEY}')
# using this script as an example: https://github.com/lablab-ai/streamlit-cohere-boilerplate/blob/main/myapp.py

if 'output' not in st.session_state:
    st.session_state['output'] = 'Output:'

# function to accept a prompt and classify it?
df = pd.read_csv("lexica_prompts_df.csv")

MODEL_NAME = "da1454fc-2118-4174-b26a-5173f4fc79a6-ft"
def preprocess_df(df):
    for k in df.keyword.unique():
        df["prompts_without_keywords"] = df.prompt.apply(lambda x: x.replace(k, ""))
    return df


def classify_prompts(df, prompts):
    exs = [Example(p, l) for p, l in zip(df.prompts_without_keywords, df.keyword)]
    # given a prompt or a list of ones, classifies them, and returns values
    print(prompts)
    response = co.classify(inputs = prompts, examples=exs)
    return response


def create_variations(prompt):
    # given a prompt and a keyword, creates variations of the prompts using the keyword
    initial_prompt = ""
    generated = co.generate(model = MODEL_NAME, prompt = prompt, num_generations = 5,
                            temperature = 0.9, max_tokens = 50)
    return generated


def create_prompt(input_prompt, keyword):
    initial_prompt = f"Rephrase this prompt with more details, in a list:\n {input_prompt} \n"
    end_prompt = "More details:"
    final_prompts = initial_prompt + end_prompt
    return final_prompts

@st.cache
def make_and_grade_variations(df, input_prompt, num_options = 5):
    # try to guess intent of the prompt being made
    #implement more generations
    if len(input_prompt) == 0:
        return
    keyword = classify_prompts(df, [input_prompt]).classifications[0].prediction
    generated = create_variations(create_prompt(input_prompt, keyword))
    list_of_gens = []
    for x in range(0, num_options):
        # will create num_options * 5 options
        generated = create_variations(create_prompt(input_prompt, keyword))
        gens_texts = [x.text for x in generated.generations]
        list_of_gens = gens_texts + list_of_gens

    #list_of_gens = [x.text for x in generated.generations]
    labels = classify_prompts(df, list_of_gens)
    # find prompts with highest probabilty
    print(labels.classifications)
    probs = []
    for output in labels.classifications:
        for l in output.confidence:
            if l.label == keyword:
                probs.append(l.confidence)
    return_dict = {"prompts": list_of_gens, "probs": probs}
    return_df = pd.DataFrame(return_dict)
    return_df.sort_values(by = ["probs"], ascending=False).reset_index(drop=True)
    #st.write(return_df)
    st.session_state.output = list(return_df.prompts)[:10]


KNOWN_ART_STYLES = {
    "cottage core": ":mushroom:",
    "cyberpunk": ":sunglasses:",
    "kawaii": ":heart:",
    "photorealistic": ":camera:",
    "water colors": ":rainbow:"
}


st.title(':art: Perfect Prompt!')
st.subheader('Get your prompt perfect before image generation, and save time!')

df = preprocess_df(df)
input = st.text_area('Enter your prospective prompts  here', height=100, value = "")
if input != "":
    st.header("We think you are trying to make art in the following style...")
    predicted_class = classify_prompts(df, [input]).classifications[0].prediction
    st.subheader(predicted_class + KNOWN_ART_STYLES[predicted_class])
    button_click = st.button('Generate Variations', on_click = make_and_grade_variations(df, input_prompt = input))
st.write(st.session_state.output)



# create a text box and button for submitting initial prompt

# create a text area to dump variations/save them

# create a suggestion tab?

#use replicate to serve stable diffusion models
