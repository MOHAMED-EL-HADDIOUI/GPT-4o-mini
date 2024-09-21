import streamlit as st
from duckduckgo_search import DDGS
from collections import deque
import asyncio
import random
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

# Asynchronous function to get LLM response
async def get_llm_response_async(prompt, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(DDGS().chat, prompt, model=model)
            return response.split()
        except Exception as e:
            if attempt < max_retries - 1:
                logging.error(f"Error occurred: {e}. Retrying in {2 ** attempt} seconds...")
                await asyncio.sleep(2 ** attempt + random.random())
            else:
                logging.error(f"Max retries reached. Error: {e}")
                return [f"<error>Unable to get response from {model} after {max_retries} attempts.</error>"]

# Asynchronous generator to process messages
async def process_message_async(message, history, prompts):
    conversation_history = deque(maxlen=5)
    for h in history:
        conversation_history.append(f"User: {h[0]}\nEcho-Refraction: {h[1]}")

    context = "\n".join(conversation_history)
    full_response = []

    # Analyze user query
    gpt4o_prompt = f"{prompts['analysis']}\n\nConversation history:\n{context}\n\nUser query: {message}\n\nPlease analyze this query and respond accordingly."
    gpt4o_response = await get_llm_response_async(gpt4o_prompt, "gpt-4o-mini")
    full_response.append(
        "### Analysis:\n" + " ".join(gpt4o_response).replace("<analyzing>", "").replace("</analyzing>", ""))

    if "<error>" in " ".join(gpt4o_response):
        return full_response

    # Rethink the response
    llama_prompt = f"{prompts['rethinking']}\n\nConversation history:\n{context}\n\nOriginal user query: {message}\n\nInitial response: {' '.join(gpt4o_response)}\n\nPlease review and suggest improvements."
    llama_response = await get_llm_response_async(llama_prompt, "gpt-4o-mini")
    full_response.append(
        "\n### Rethinking:\n" + " ".join(llama_response).replace("<rethinking>", "").replace("</rethinking>", ""))

    if "<error>" in " ".join(llama_response):
        return full_response

    # Finalize the response
    if "done" not in " ".join(llama_response).lower():
        final_prompt = f"{prompts['refinement']}\n\nConversation history:\n{context}\n\nOriginal user query: {message}\n\nInitial response: {' '.join(gpt4o_response)}\n\nSuggestion: {' '.join(llama_response)}\n\nPlease provide a final response."
        final_response = await get_llm_response_async(final_prompt, "gpt-4o-mini")
        full_response.append(
            "\n### Final Response:\n" + " ".join(final_response).replace("<output>", "").replace("</output>", ""))
    else:
        full_response.append(
            "\n### Final Response: The initial response is satisfactory and no further refinement is needed.")

    return full_response

# Function to handle user input
def respond_async(message, history, prompts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_message_async(message, history, prompts))

# Prompts definitions
prompts = {
    "analysis": """You are Echo-Refraction, an AI assistant tasked with analyzing user queries. Your role is to:
    1. Carefully examine the user's input for clarity, completeness, and potential ambiguities.
    2. Identify if the query needs refinement or additional information.
    3. If refinement is needed, suggest specific improvements or ask clarifying questions.
    4. If the query is clear, respond with "Query is clear and ready for processing."
    5. Provide a brief explanation of your analysis in all cases.
    Enclose your response in <analyzing> tags.""",

    "rethinking": """You are Echo-Refraction, an advanced AI model responsible for critically evaluating and improving responses. Your task is to:
    1. Carefully review the original user query and the initial response.
    2. Analyze the response for accuracy, relevance, completeness, and potential improvements.
    3. Consider perspectives or approaches that might enhance the response.
    4. If you identify areas for improvement:
        a. Clearly explain what aspects need refinement and why.
        b. Provide specific suggestions for how the response could be enhanced.
        c. If necessary, propose additional information or context that could be included.
    5. If the initial response is satisfactory and you have no suggestions for improvement, respond with "Done."
    Enclose your response in <rethinking> tags.""",

    "refinement": """You are Echo-Refraction, an AI assistant tasked with providing a final, refined response to the user. Your role is to:
    1. Review the original user query, your initial response, and the suggestions provided.
    2. Consider the feedback and suggestions for improvement.
    3. Integrate the suggested improvements into your response, ensuring that:
        a. The information is accurate and up-to-date.
        b. The response is comprehensive and addresses all aspects of the user's query.
        c. The language is clear, concise, and appropriate for the user's level of understanding.
    4. If you disagree with any suggestions, provide a brief explanation of why you chose not to incorporate them.
    5. Deliver a final response that represents the best possible answer to the user's query.
    Enclose your response in <output> tags."""
}

# Streamlit app layout with enhanced design
st.set_page_config(page_title="Chat with Echo-Refraction", layout="wide")
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f6;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #1e3a8a;
            font-family: 'Arial', sans-serif;
        }
        .analysis-title { color: #0056b3; font-weight: bold; font-size: 1.2em; }
        .rethinking-title { color: #e67e22; font-weight: bold; font-size: 1.2em; }
        .final-response-title { color: #28a745; font-weight: bold; font-size: 1.2em; }
        .response-box { 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .analysis-box { background-color: #e6f2ff; }
        .rethinking-box { background-color: #fff4e6; }
        .final-response-box { background-color: #e6ffe6; }
        .response-time { font-weight: bold; color: #ff6600; margin-bottom: 10px; }
        .progress-bar { 
            width: 100%; 
            background-color: #ddd; 
            border-radius: 5px; 
            overflow: hidden;
        }
        .progress-bar-filled { 
            height: 30px; 
            background-color: #4caf50; 
            width: 0; 
            text-align: center; 
            line-height: 30px; 
            color: white; 
            transition: width 0.5s ease-in-out;
        }
        .stTextInput input {
            border-radius: 20px;
            border: 2px solid #1e3a8a;
            padding: 10px 15px;
        }
        .stButton>button {
            border-radius: 20px;
            background-color: #1e3a8a;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            transition: all 0.3s ease;
            width: 40%;
            margin-top: 10px;
        }
        .stButton>button:hover {
            background-color: #2c5282;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .conversation-history {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .conversation-history h3 {
            color: #1e3a8a;
            margin-bottom: 15px;
        }
        .user-message, .assistant-message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e6f2ff;
            text-align: right;
        }
        .assistant-message {
            background-color: #f0f0f0;
        }
        .input-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .processing {
            color: #ff6600;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Chat with GPT-4o mini")
st.subheader("Engage with GPT-4o mini, an AI assistant that analyzes, rethinks, and refines responses.")

# Input box for user query and submit button
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
user_input = st.text_input("Enter your query:")
submit_button = st.button("Submit")
st.markdown("</div>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []  # Initialize history in session state
if 'processing' not in st.session_state:
    st.session_state.processing = False  # Flag to indicate if a query is being processed

# Display conversation history
if st.session_state.history:
    st.markdown("<div class='conversation-history'>", unsafe_allow_html=True)
    st.markdown("### Conversation History:")
    for i, h in enumerate(st.session_state.history):
        st.markdown(f"<div class='user-message'>{h[0]}</div>", unsafe_allow_html=True)
        if i == len(st.session_state.history) - 1 and st.session_state.processing:
            st.markdown("<div class='assistant-message processing'>Processing your request...</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{h[1]}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if submit_button:
    if user_input:
        st.session_state.processing = True
        start_time = time.time()  # Start timing
        st.session_state.history.append((user_input, ""))  # Add user input to history

        # Progress bar simulation for response time
        progress_bar = st.empty()
        progress_bar.markdown("<div class='progress-bar'><div id='progress-bar-filled' class='progress-bar-filled'></div></div>",
                    unsafe_allow_html=True)

        responses = respond_async(user_input, st.session_state.history, prompts)
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        st.markdown(f"<div class='response-time'>Response Time: {elapsed_time:.2f} seconds</div>",
                    unsafe_allow_html=True)

        final_response = ""
        for response in responses:
            title = ""
            box_class = ""
            if "### Analysis:" in response:
                title = "| Analysis |"
                box_class = "analysis-box"
            elif "### Rethinking:" in response:
                title = "| Rethinking |"
                box_class = "rethinking-box"
            elif "### Final Response:" in response:
                title = "| Final Response |"
                box_class = "final-response-box"
                final_response = response.replace("### Final Response:", "").strip()

            st.markdown(f"<div class='{box_class} response-box'>{title}<br>{response}</div>", unsafe_allow_html=True)

        # Update history with the latest response
        st.session_state.history[-1] = (user_input, final_response)  # Store only the final response
        st.session_state.processing = False

        # JavaScript to simulate progress bar based on elapsed time
        st.markdown(f"""
            <script>
                let progressBar = document.getElementById("progress-bar-filled");
                let width = 1;
                let interval = setInterval(frame, {elapsed_time * 10});
                function frame() {{
                    if (width >= 100) {{
                        clearInterval(interval);
                    }} else {{
                        width++;
                        progressBar.style.width = width + '%';
                    }}
                }}
            </script>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please enter a query.")