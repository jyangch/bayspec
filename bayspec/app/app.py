import streamlit as st
from bayspec.util.tools import init_session_state
from st_pages import add_page_title, get_nav_from_toml


st.set_page_config(layout="wide")

nav = get_nav_from_toml('.streamlit/pages.toml')

st.logo('.streamlit/logo.png')

pg = st.navigation(nav)

add_page_title(pg)

pg.run()

init_session_state()
