import streamlit as st
import langchain_helper

st.title("🍴 Restaurant Name Generator")

# Sidebar cuisine selector
cuisine = st.sidebar.selectbox(
    "Pick a Cuisine",
    ("Indian", "Italian", "Mexican", "Arabic", "American")
)

if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)

    # Display restaurant name
    st.header(response["restaurant_name"].strip())

    # Display menu items
    st.subheader("Menu Items")
    menu_items = [item.strip() for item in response["menu_items"].split(",")]
    for item in menu_items:
        st.write(f"- {item}")
