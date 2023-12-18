import os
import streamlit as st
import pandas as pd
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import pyjokes
from googletrans import Translator

# Function to get nutrient information from a food name
def get_nutrient_info(food_name):
    # Replace 'YOUR_API_KEY' with your actual USDA FoodData Central API key
    api_key = 'AJkI7imE8qiJN6F2a6F3kpdIlzogNsmCXDflzLTx'
    base_url = 'https://api.nal.usda.gov/fdc/v1/foods/search'

    # Make a request to the API to search for the food
    params = {
        'api_key': api_key,
        'query': food_name,
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        # Parse the response to get the first food item
        food_item = response.json()['foods'][0]

        # Extract relevant nutrient information
        nutrient_info = {'Food': food_item['description']}

        # Check if 'foodNutrients' key is present
        if 'foodNutrients' in food_item:
            # Extract nutrient information based on the available keys
            for nutrient in food_item['foodNutrients']:
                nutrient_name = nutrient.get('nutrientName', '')
                nutrient_amount = nutrient.get('amount', '')
                nutrient_info[nutrient_name] = nutrient_amount

        return nutrient_info
    else:
        # Handle API request failure
        return {'error': 'Unable to fetch nutrient information'}

# Function to get a joke
def get_joke():
    return pyjokes.get_joke()

# Function for language translation using Google Translate
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Set Streamlit app theme
st.markdown("""
    <style>
        body {
            background-color: #FFD2D5;  /* Light Rose background color */
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #3E4E50;  /* Dark Gray text color */
        }
        .st-bw {
            color: #FF7E8D;  /* Rose color for emphasized text */
        }
        .stButton > button {
            background-color: #FF7E8D;  /* Rose color for buttons */
            color: #FFFFFF;  /* White text color for buttons */
        }
        .stTextInput > div > div > div > input {
            background-color: #FFF;  /* White color for text input */
            color: #3E4E50;  /* Dark Gray text color for text input */
        }
        .stTextInput > div > div > div > label {
            color: #3E4E50;  /* Dark Gray text color for labels */
        }
        .stTextInput > div > div > div > div > svg {
            fill: #3E4E50;  /* Dark Gray fill color for icons */
        }
        .nutrient-box {
            background-color: #FFD2D5;  /* Light Rose color for nutrient box */
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .chat-box {
            background-color: #FFF;  /* White color for chat box */
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .user-message {
            background-color: #DCF8C6;  /* Light Green color for user messages */
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 5px;
        }
        .bot-message {
            background-color: #D1C4E9;  /* Light Purple color for bot messages */
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 5px;
        }
        .prompt-button {
            background-color: #FF7E8D;  /* Rose color for prompt buttons */
            color: #FFFFFF;  /* White text color for prompt buttons */
            margin-right: 10px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and navigation bar
st.title('Lucy-AI✨')

# Navigation bar
nav_selection = st.sidebar.radio('Navigate to', ['Home', 'Emotional Support', 'Medical Terms', 'Mental Health Advice', 'Period Tracking', 'Movie Suggestion', 'Nutrient Information', 'Mood Tracker', 'Language Translation'])

if nav_selection == 'Home':
    st.write('Welcome to Lucy-AI! Choose a feature from the navigation bar.')

# Features
elif nav_selection == 'Emotional Support':
    # Feature implementation for Emotional Support
    st.header('Emotional Support')
    prompt1 = st.text_input('Ask me anything (Emotional Support)', help='Enter your question here')
    template1 = PromptTemplate(
        input_variables=['topic'],
        template='Give sympathy and listen to the user and provide reassurance have a friendly convo {topic}'
    )
    if st.button('Submit'):
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            frequency_penalty=0.5,
            max_tokens=500,
            streaming=True,
        )
        chain = LLMChain(llm=llm, prompt=template1, verbose=True)
        if prompt1:
            response = chain.run(topic=prompt1)
            st.markdown('<div class="chat-box"><div class="user-message">User: ' + prompt1 + '</div><div class="bot-message">Lucy-AI: ' + response + '</div></div>', unsafe_allow_html=True)

elif nav_selection == 'Medical Terms':
    # Feature implementation for Medical Terms
    st.header('Medical Terms')
    prompt2 = st.text_input('Enter medical term (Medical Terms)', help='Type the medical term you want to understand')
    template2 = PromptTemplate(
        input_variables=['prompt'],
        template='Convert this medical language to layman language {prompt}'
    )
    if st.button('Submit'):
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            frequency_penalty=0.5,
            max_tokens=500,
            streaming=True,
        )
        chain = LLMChain(llm=llm, prompt=template2, verbose=True)
        if prompt2:
            response = chain.run(prompt=prompt2)
            st.markdown('<div class="chat-box"><div class="user-message">User: ' + prompt2 + '</div><div class="bot-message">Lucy-AI: ' + response + '</div></div>', unsafe_allow_html=True)

elif nav_selection == 'Mental Health Advice':
    # Feature implementation for Mental Health Advice
    st.header('Mental Health Advice')
    prompt3 = st.text_input('Enter mental health topic (Mental Health Advice)', help='Type the mental health topic')
    template3 = PromptTemplate(
        input_variables=['topic'],
        template='Provide mental health advice for feeling {topic}'
    )
    if st.button('Submit'):
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            frequency_penalty=0.5,
            max_tokens=500,
            streaming=True,
        )
        chain = LLMChain(llm=llm, prompt=template3, verbose=True)
        if prompt3:
            response = chain.run(topic=prompt3)
            st.markdown('<div class="chat-box"><div class="user-message">User: ' + prompt3 + '</div><div class="bot-message">Lucy-AI: ' + response + '</div></div>', unsafe_allow_html=True)

elif nav_selection == 'Period Tracking':
    # Feature implementation for Period Tracking
    st.header('Period Tracking')
    last_period_date = st.date_input('Enter the date of your last period')
    cycle_length = st.number_input('Enter your cycle length (in days)', min_value=1, max_value=50, value=28)
    if st.button('Submit'):
        if last_period_date:
            next_period_date = last_period_date + pd.DateOffset(days=cycle_length)
            st.markdown('<div class="nutrient-box">Your estimated next period date is: ' + next_period_date.strftime('%Y-%m-%d') + '</div>', unsafe_allow_html=True)

elif nav_selection == 'Movie Suggestion':
    # Feature implementation for Movie Suggestion
    st.header('Movie Suggestion')
    language = st.selectbox('Select Language', ['English', 'Spanish', 'French'])
    genre = st.selectbox('Select Genre', ['Action', 'Drama', 'Comedy'])
    if st.button('Get Movie Suggestion'):
        movie_prompt = f"Please suggest a {genre.lower()} movie in {language.lower()} language."
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            frequency_penalty=0.5,
            max_tokens=500,
            streaming=True,
        )
        chain = LLMChain(llm=llm, prompt=movie_prompt, verbose=True)
        response = chain.run(topic='')
        st.markdown('<div class="chat-box"><div class="user-message">User: ' + movie_prompt + '</div><div class="bot-message">Lucy-AI: ' + response + '</div></div>', unsafe_allow_html=True)

elif nav_selection == 'Nutrient Information':
    # Feature implementation for Nutrient Information
    st.header('Nutrient Information')
    food_name_nutrient = st.text_input('Enter a food name for nutrient information')
    if st.button('Get Nutrient Information'):
        nutrient_info = get_nutrient_info(food_name_nutrient)
        if 'error' in nutrient_info:
            st.markdown('<div class="nutrient-box">' + nutrient_info['error'] + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nutrient-box">Nutrient Information:<br>' + ', '.join(f"{key}: {value}" for key, value in nutrient_info.items()) + '</div>', unsafe_allow_html=True)

elif nav_selection == 'Mood Tracker':
    # Feature implementation for Mood Tracker
    st.header('Mood Tracker')
    mood_rating = st.slider('Rate your mood today (1-10)', min_value=1, max_value=10, value=5)
    mood_notes = st.text_area('Additional notes about your mood')
    # Joke Generator based on mood
    if mood_rating <= 3:
        st.warning("Looks like you're not feeling great. Let me tell you a joke to cheer you up!")
        joke = get_joke()
        st.markdown('<div class="chat-box"><div class="bot-message">Lucy-AI: ' + joke + '</div></div>', unsafe_allow_html=True)

elif nav_selection == 'Language Translation':
    # Feature implementation for Language Translation
    st.header('Language Translation')
    text_to_translate = st.text_area('Enter text to translate')
    target_language = st.selectbox('Select target language', ['en', 'es', 'fr', 'de', 'ja'])
    if st.button('Translate'):
        translation_result = translate_text(text_to_translate, target_language)
        st.markdown('<div class="nutrient-box">Translated Text:<br>' + translation_result + '</div>', unsafe_allow_html=True)
