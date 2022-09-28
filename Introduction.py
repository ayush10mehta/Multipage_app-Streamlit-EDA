import streamlit as st
from st_functions import st_button, load_css



# st.set_page_config(layout="wide")

st.set_page_config(
page_icon = "➡️",
layout="wide",)


st.title("Lending Club Case Study")
#st.sidebar.success("Select a page above.")

st.sidebar.header("Ayush Mehta")
st.sidebar.info("Data Science | Data Analytics | Tableau developer | Machine Learning")

icon_size = 20

with st.sidebar:
    load_css()
    st_button('linkedin', 'https://www.linkedin.com/in/ayush-mehta-09a3a91b8/', 'Follow me on LinkedIn', icon_size)
    st_button('medium', 'https://ayush10mehta.github.io/Projects/', 'Read my Projects', icon_size)
    st_button('tableau', 'https://public.tableau.com/app/profile/ayush.mehta', 'Review my Tableau Public', icon_size)
    st_button('github', 'https://github.com/ayush10mehta?tab=repositories', 'Github', icon_size)


"""
A consumer finance company that specialises in providing urban customers with various forms of loans. When a company receives a loan application, it must decide whether or not to approve the loan based on the applicant's profile. The bank's decision is related with two sorts of risks:
\nIf the applicant is likely to repay the loan, the company will lose business if the loan is not approved.
Approving the loan may result in a financial loss for the company if the applicant is unlikely to repay the loan, i.e. if he or she is likely to default.
\nThe data pertains to previous loan applicants and whether or not they 'defaulted.' The goal is to find patterns that can be used to make decisions.
"""
st.header('Business Problem')
"""
This firm is the world's largest online loan marketplace, allowing personal loans, commercial loans, and medical procedure funding. Through a quick internet interface, borrowers can readily acquire cheaper interest rate loans.
\nLending to 'risky' applicants, like most other lending organisations, is the most common source of financial loss (called credit loss). The amount of money lost by the lender when a borrower refuses to pay or flees with the money owed is referred to as credit loss. In other words, defaulting borrowers do the most financial harm to lenders. The 'defaulters' are the consumers who have been labelled as 'charged-off.'
\nIf these problematic loan applicants can be identified, the size of the loan can be reduced, reducing the amount of credit loss.
\nThe goal of this case study is to identify such applications using EDA.
\nIn other words, the organisation needs to know the reasons (or driver variables) that cause loan default, i.e. the variables that are significant predictors of default. This knowledge can be used to the company's portfolio and risk assessment.
\nTo gain a better understanding of the field, you should perform some independent study on risk analytics.
"""
from PIL import Image

st.header("Process flow diagram")
#background = st.image('/Users/ayushmehta/Downloads/Upgrad/Case Study EDA/Untitled Diagram.drawio.png')

background = Image.open("/Users/ayushmehta/Downloads/Upgrad/Case Study EDA/Untitled Diagram.drawio.png")
col1, col2, col3 = st.columns([1, 0.2, 0.2])
col1.image(background, use_column_width=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
#import warnings
#warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------
#from st_aggrid import AgGrid
loan = pd.read_csv("/Users/ayushmehta/Downloads/Upgrad/Case Study EDA/loan.csv")
#AgGrid(loan)


