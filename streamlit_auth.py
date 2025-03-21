import streamlit as st

st.title("Webpage")

st.write("Testing authentication.")

if not st.experimental_user.is_logged_in:

    if st.button(label="Login with Google"):
        st.login()

else:
    st.write(f"Hello, {st.experimental_user.name}!")
    st.write(st.experimental_user)

if st.button(label="Logout"):
    st.logout()
