import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.image("images/logo-no-background.png", width=300)

with col2:
    st.title("Music Solution 2000")
    info = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    In vitae elementum mi. Mauris sit amet tincidunt risus, vitae semper lorem.
    Ut et nisl dictum, sodales arcu quis, ultricies ex. Suspendisse hendrerit, neque vitae luctus bibendum, urna quam molestie velit, ac tincidunt metus mi ut massa. 
    Nam ac interdum neque. Sed feugiat velit velit, ut mattis nisi facilisis nec. 
    Nunc accumsan euismod diam. Curabitur commodo ex lobortis feugiat posuere. 
    Vivamus et turpis sed lorem tristique pharetra non sed nisl. Phasellus vitae pretium massa. Aliquam erat volutpat. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    """
    st.write(info)