import streamlit as st
from bayspec import __readme__


css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

with open(__readme__) as file:
    readme = file.read()
    
readme.replace('*BAYSPEC*', ':rainbow[*BAYSPEC*]')

st.markdown(readme)

st.markdown(
    """
    ## ⭐ Star the project on Github <iframe src="https://ghbtns.com/github-btn.html?user=jyangch&repo=bayspec&type=star&count=true" width="150" height="20" title="GitHub"></iframe> 
    """, 
    unsafe_allow_html=True)