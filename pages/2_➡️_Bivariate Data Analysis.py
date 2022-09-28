import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
#import warnings
#warnings.filterwarnings('ignore')
import streamlit as st



st.set_page_config(
page_title = "Biivariate Data Analysis",
layout="wide",)


loan = pd.read_csv("loan.csv")
loan.drop(['collections_12_mths_ex_med', 'mths_since_last_major_derog', 'tax_liens', 'tot_hi_cred_lim', 
           'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'annual_inc_joint', 'dti_joint', 
           'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 
           'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 
           'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 
           'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 
           'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 
           'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 
           'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 
           'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 
           'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 
           'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75'], axis=1, inplace=True)

loan.drop(['application_type', 'policy_code', 'initial_list_status', 'pymnt_plan'], axis=1, inplace=True)

loan.drop(['mths_since_last_delinq', 'mths_since_last_record', 'out_prncp', 'out_prncp_inv', 
           'collection_recovery_fee', 'total_rec_late_fee', 'recoveries', 'next_pymnt_d', 'pub_rec_bankruptcies',
          'delinq_2yrs'], axis=1, inplace=True)

loan.drop(['dti', 'id', 'member_id', 'emp_title', 'desc', 'url', 'title', 'zip_code'], axis=1, inplace=True)
loan.drop(['funded_amnt'], axis=1, inplace=True)

loan.drop(["total_pymnt", "total_rec_prncp", "total_rec_int", "total_pymnt", "total_rec_int", 
           "total_rec_prncp", "last_pymnt_amnt" ], axis = 1, inplace = True)
loan = loan.iloc[:,:15]

loan.rename(columns = {'loan_amnt':'loan_amount', 'funded_amnt_inv':'funded_amount_investment',
                              'emp_length':'employment_length', 'int_rate':'interest_rate', 'annual_inc':'annual_income'
                      , 'issue_d':'issue_date', 'addr_state':'address_state'}, inplace = True)
loan["employment_length"].value_counts()
loan.fillna({'employment_length': 'Not Mentioned'}, inplace=True)
# Split int_rate on "%"
loan[['int_rate-split-0-whu1', 'int_rate-split-1-whu1']] = loan['interest_rate'].str.split('%', -1, expand=True)
loan = loan[loan.columns[:7].tolist() + ['int_rate-split-0-whu1', 'int_rate-split-1-whu1'] + loan.columns[7:-2].tolist()]

# Deleted columns int_rate-split-1-whu1 & interest_rate
loan.drop(['int_rate-split-1-whu1','interest_rate'], axis=1, inplace=True)

# Changed int_rate-split-0-whu1 to dtype float
loan = loan.astype({"int_rate-split-0-whu1": float})

# Renamed columns interest_rate
loan.rename(columns={'int_rate-split-0-whu1': 'interest_rate'}, inplace=True)
#convert column with object to date time
loan['issue_date'] = pd.to_datetime(loan['issue_date'], errors = 'coerce')



st.header("Bivariate Analysis and Extended Segmented Analysis")

# --------------------------------------------------------------------------------------------------

st.subheader("Creating the count and default rate(derived metric) plot of all categorical values by taking 2 categorical in an account at a time.")
cat_variables1 = ["loan_status","term","grade","sub_grade","employment_length",
                 "home_ownership","verification_status","purpose","address_state"]

colSWWW1,colSWWW2,colSWWW3 = st.columns(3)
field1 = colSWWW1.selectbox('Pick 1st Categorical Variable', cat_variables1, key = 'count')
field2 = colSWWW2.selectbox('Pick 2nd Categorical Variable', cat_variables1, key = 'count1')
colFIGG1,colFIGG2= st.columns(2)
if field1 != field2:
    loan_charged_off = loan.query("loan_status == 'Charged Off'")
    b = loan_charged_off.filter([field2,field1])
    b['Count'] = 1
    group_1 = b.groupby([field2,field1]).sum()

    c = loan.filter([field2,field1])
    c['Count_Total'] = 1
    group_2 = c.groupby([field2,field1]).sum()
    df_cd = pd.merge(group_1, group_2, how='inner', left_on = [field2,field1], right_on = [field2,field1])


    df_cd['rate'] = (df_cd['Count']/df_cd['Count_Total'])*100
    df_cd.reset_index(level=0, inplace=True)
    df = pd.DataFrame(dict(col1=np.linspace(1, 10, 5), col2=np.linspace(1, 10, 5)))

    fig5 = px.histogram(x=df_cd.index, y =df_cd['rate'], color= df_cd[field2])
    fig5.update_layout(
        title="Default Rate", 
        xaxis = dict(
            rangeslider = dict(
                visible=True, 
                thickness=0.05
            )
        ), 
        yaxis = dict(

        ), 
        barmode='group', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True,
    xaxis_title=field1,
    yaxis_title="Default Rate"
    )
    with colFIGG1:
        st.plotly_chart(fig5, use_container_width=True)
# ----------------------------------------------------------
    fig6 = px.histogram(x=df_cd.index, y = df_cd['Count_Total'], color=df_cd[field2])
    fig6.update_layout(
        title="Total Count", 
        xaxis = dict(
            rangeslider = dict(
                visible=True, 
                thickness=0.05
            ),
        ), 
        yaxis = dict(

        ), 
        barmode='group', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True,
    xaxis_title=field1,
    yaxis_title="Total Count"
    )
    with colFIGG2:
        st.plotly_chart(fig6, use_container_width=True)
else:
    loan_charged_off = loan.query("loan_status == 'Charged Off'")
    
    b = loan_charged_off.filter([field1])
    b['Count'] = 1
    group_1 = b.groupby(field1).sum()
    
    c = loan.filter([field1])
    c['Total_count'] = 1
    group_2 = c.groupby(field1).sum()

    df_cd = pd.merge(group_1, group_2, how='inner', left_on = field1, right_on = field1)
    
    df_cd['Default_rate'] = (df_cd['Count']/df_cd['Total_count'])*100


    df = pd.DataFrame(dict(col1=np.linspace(1, 10, 5), col2=np.linspace(1, 10, 5)))

    fig5 = px.histogram(x=df_cd.index, y =df_cd['Default_rate'])
    fig5.update_layout(
        title="Default Rate", 
        xaxis = dict(
            rangeslider = dict(
                visible=True, 
                thickness=0.05
            )
        ), 
        yaxis = dict(

        ), 
        barmode='group', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True,
    xaxis_title=field1,
    yaxis_title="Default Rate"
    )
    with colFIGG1:
        st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.histogram(x=df_cd.index, y = df_cd['Total_count'])
    fig6.update_layout(
        title="Total Count", 
        xaxis = dict(
            rangeslider = dict(
                visible=True, 
                thickness=0.05
            ),
        ), 
        yaxis = dict(

        ), 
        barmode='group', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True,
    xaxis_title=field1,
    yaxis_title="Total Count"
    )
    with colFIGG2:
        st.plotly_chart(fig6, use_container_width=True)

# --------------------------------------------------------------------------------------------------

st.subheader("Scatter plot for all the numerical variables.")
continous_variable = ["loan_amount","annual_income","installment","interest_rate"]
colSWWW11,colSWWW22,colSWWW33 = st.columns(3)
field11 = colSWWW11.selectbox('Pick 1st Numerical Variable', continous_variable, key = 'count33')
field22 = colSWWW22.selectbox('Pick 2nd Numerical Variable', continous_variable, key = 'count44')

fig7 = px.scatter(loan, x=field11, y = field22)
fig7.update_layout(
    title='Scatter plot between Loan amount and Funded amount', 
    xaxis = dict(
        rangeslider = dict(
            visible=True, 
            thickness=0.05
            )
        ), 
        yaxis = dict(

        ), 
        barmode='stack', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True
    )

st.plotly_chart(fig7, use_container_width=True)

# --------------------------------------------------------------------------------------------------
st.subheader("Scatter plot for all the numerical variable and keeping one categorical column as a differentiator.")
colSwitch1,colSwitch2,colSwitch3 = st.columns(3)
field111 = colSwitch1.selectbox('Pick 1st Numerical Variable', continous_variable, key = 'count333')
field222 = colSwitch2.selectbox('Pick 2nd Numerical Variable', continous_variable, key = 'count444')
field333 = colSwitch3.selectbox('Pick Categorical Variable', cat_variables1, key = 'count555')

fig8 = px.scatter(loan, x=field111, y = field222,color = field333)
fig8.update_layout(
    title='Scatter plot between Loan amount and Funded amount', 
    xaxis = dict(
        rangeslider = dict(
            visible=True, 
            thickness=0.05
            )
        ), 
        yaxis = dict(

        ), 
        barmode='stack', 
        paper_bgcolor='#FFFFFF', 
        showlegend=True
    )

st.plotly_chart(fig8, use_container_width=True)

# --------------------------------------------------------------------------------------------------

"""
## Insights
### 1. Home Ownership & Purpose
### The default rate of all the purposes is equaly distributed for one who owns, mortgage and rent the home.
### In the case where the people had filled other in home ownership are likely to be defaulters but specifically for purposes like  education, home, major-purchase, medical, renewable enery & wedding are giving zero defaulters and maximum we are getting defaulter for the purpose moving i:e 100%.
### 2. Grade & Employment length
### A grade people are the highest who haven't mentioned there employment length.
### There are 100% defaulters who haven't mentioned there employment length.
### 3. Grade & Sub grade
### G & F grade people are the one with highest default rate. F comes in the list of highest default rate because of the F5 sub grade as it is having the highest default rate of 45% increasing the overall default rate in F grade.
### 4. Grade & Purpose
### Highest default rate is from the one who had taken loan for opening small businesses and mostly high grade people take loan for this purposes. Midiocar grade people take loan for purposes like education and house.
### 5. Interest rate & grade
### It has been seen that loan are provided at high interests to low grade people as compared to the high grade people.
### 6. Address state, Verification status & Terms
### In NE state no one has a verified status and this state has the highest rate of defaulters and all had taken loan for 36 terms.
### 7. Address state & Home ownership
### PA, MO, AR are major states where people had filled home ownership = other and from our analysis we know that the one who had filled home owneship as other are most likely to be defaulters.
### 8. Issue date & Grade
### The rate of defaulters for grade A has been decreased in a span from 2007 to 2011 and the number of A grade people taking loan has been increased.
### There is much more raise in defaulters for grade C & E as compared to the number of people taking loans.

# Recommendations
### 1. Interest rate increases as per the grade. May be this is one of the reason for the increase in the defaulters in the grade G as the interest rate is highest for this grade 20-25%. We need to reconsider the rate of interest according to some different factors not by the grades.
### 2. There should be through analysis of the people who are applying loans for opening small business. The background check is usually be done but we also need to do the detail analysis of the business plans which the one is asking loan for because from our analysis mostly defaulters belongs to this category and from our bivariate analysis we can clearly see that mostly high grade people apply for this type of loan so majoritly there backgroud is great but not there plans for the business which ultimatly leads to the loss.
### 3. It should be strict to mention there employment lenght as most of the defaulters are coming from the one who haven't mentioned there employment length and also a detail background check of there employbility as there can be cases where the show fake employment.
### 4. Either stop or be very carefull before taking any application from the state NE, it has the highest defaulters rate and all apply for only 36 months terms.

# Interesting facts
### 1. For sub grade A1 who has taken loan for 60 months terms are least likely to be defaulters with 0% defaulters.
### 2. For sub grade G3 who has taken loan for 36 months terms are most likely to be defaulters with 100% defaulters.
### 3. For grade G who has not mentioned there employment length are most likely to be defaulters with 100% defaulters.
"""
