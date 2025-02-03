import streamlit as st
import os

# Function to display the main page with cards
def main_page():
    """
    Displays the main page with a large title and 6 interactive cards for navigation.
    """
    st.markdown("""<h1 style='text-align: center; font-size: 100px; color: #2C3E50;'>LOBO</h1>""", unsafe_allow_html=True)
    st.write("### Welcome to LOBO! Click on a card to explore.")

    # Create 6 large buttons (cards) in 2 rows with 3 columns each
    col1, col2 = st.columns(2, gap='large')  # Adjust column width to move left buttons more left

    button_style = """
        <style>
            div.stButton > button {
                width: 100%;
                height: 200px;
                font-size: 28px;
                font-weight: 500;
                border-radius: 10px;
                background-color: #ECF0F1; /* Light Gray */
                color: #2C3E50; /* Dark Grayish Blue */
                border: none;
                transition: all 0.3s ease;
            }
            div.stButton > button:hover {
                background-color: #BDC3C7; /* Slightly darker gray on hover */
            }
        </style>
    """

    st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAFAFA; /* Minimalist light background */
    }
    </style>
    """, unsafe_allow_html=True
    )
    st.markdown(button_style, unsafe_allow_html=True)

    with col1:
        if st.button("""**CHATBOT**\n\n An intelligent digital assistant designed to enhance user interaction with AI-driven responses. Engage in seamless, human-like conversations and explore its powerful capabilities.
        """, use_container_width=True):
            os.system("streamlit run app.py")
        if st.button("Card 2"):
            st.session_state.page = "Page 2"
        if st.button("Card 3"):
            st.session_state.page = "Page 3"

    with col2:
        if st.button("Card 4"):
            st.session_state.page = "Page 4"
        if st.button("Card 5"):
            st.session_state.page = "Page 5"
        if st.button("Card 6"):
            st.session_state.page = "Page 6"

# Function to display individual pages
def show_page(page_name):
    """
    Displays the content of the selected page with a back button.
    """
    st.markdown(f"""<h1 style='text-align: center; font-size: 40px; color: #2C3E50;'>{page_name}</h1>""", unsafe_allow_html=True)
    st.write(f"### This is {page_name}. You clicked on a card to get here!")
    
    if st.button("Back to Main Page"):
        st.session_state.page = "Main"

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Main"

# Render the appropriate page based on session state
if st.session_state.page == "Main":
    main_page()
else:
    show_page(st.session_state.page)