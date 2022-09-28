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
page_title = "Univariate Data Analysis",
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



st.header("Univariate Analysis and Segmented Analysis")
st.subheader("Checking for the outliers")
"""Filter the columns for which we need to check the outliers."""

check_outliers = ["loan_amount","annual_income","installment","interest_rate"]
colSW1,colSW2,colSW3 = st.columns(3)
field = colSW1.selectbox('Pick one', check_outliers)
col1,col2 = st.columns(2)
# ------------------------------------------------------------------------------

fig = px.box(loan, x='loan_status', y = field, color='loan_status')
fig.update_layout(
    title='Box Plot of'+ " " + field, 
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
with col1:
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
fig = px.histogram(loan, x=field, color='loan_status')
fig.update_layout(
    title='Histogram of'+ " " + field, 
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
with col2:
    st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------------------------------------------

loan.annual_income.quantile([0.50,0.75,0.80,0.85,0.90, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99])
loan = loan[loan.annual_income <= 132000]


st.subheader("Creating count plot for all categorical variables.")
cat_variables = ["loan_status","term","grade","sub_grade","employment_length",
                 "home_ownership","verification_status","purpose","address_state"]
colSWW1,colSWW2,colSWW3 = st.columns(3)
field2 = colSWW1.selectbox('Pick one', cat_variables)
fig2 = px.histogram(loan, x=field2)
fig2.update_layout(
        title=field2 + " " + 'histogram', 
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
        showlegend=True
    )
st.plotly_chart(fig2, use_container_width=True)
# ------------------------------------------------------------------------------
st.subheader("Creating count plot of the numerical variables by creating the bins.")
colFig1,colFig2= st.columns(2)

loan['int_rate_groups'] = pd.cut(loan['interest_rate'], bins=5,precision =0,labels=['5%-10%','10%-15%','15%-20%','20%-25%','25%-30%'])
fig3 = px.histogram(loan[loan.loan_status == 'Charged Off'], x='int_rate_groups')
fig3.update_layout(
        title="Interest Rate", 
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
        showlegend=True
    )
with colFig1:
    st.plotly_chart(fig3, use_container_width=True)
# ----------------------------------------------------------------------------------------------------------
loan['loan_amnt_grps'] = pd.cut(loan['loan_amount'], bins=6,precision =0,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k'])
fig4 = px.histogram(loan[loan.loan_status == 'Charged Off'], x='loan_amnt_grps')
fig4.update_layout(
        title="Loan amount group", 
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
        showlegend=True
    )
with colFig2:
    st.plotly_chart(fig4, use_container_width=True)

"""
#### Rate of interest
#### 1. The most number of defaulters count are from one having interest rate between 15-20%.
#### 2. But the defulters are most likely to be the one having the interest rate between 20-23% as the average rate of defaulters is the highest for this range.
#### Loan Amount
#### 1. The most number of defaulters count are from one applying loan for amount range from 1-10k.
#### 2. But the defaulters are most likely to be the one applying loan for amount range from 26-28k the average defaulters rate is around 25%.

#### Comparison of categorical columns with the count of defaulters and the rate of defaulters.
#### 1. Count of defaulters can be high for any category as the number of loan taken from any category is high.
#### 2. Finding rate is needed as analysing only the count will not give us the correct insights.
#### 3. For every graph we had created a derived coulmn 'Default_rate' which give calculate the rate of defaulters.
"""
"""
####
"""
# ---------------------------------------------------------------------------------------------------------------------

st.subheader("Creating the count and default rate(derived metric) plot of all categorical values.")
cat_variables1 = ["loan_status","term","grade","sub_grade","employment_length",
                 "home_ownership","verification_status","purpose","address_state"]

colSWWW1,colSWWW2,colSWWW3 = st.columns(3)
field1 = colSWWW1.selectbox('Pick one', cat_variables1, key = 'count')

colFIGG1,colFIGG2= st.columns(2)

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

# ---------------------------------------------------------------------------------------------------------------------
"""
## Insights
### Max Defaulters Rate: The rate of defaulters.
### Max Count: The count of number of defaulters.
### 1. Terms
#### Max Count: The most number of defaulters count are from 36 months term.
#### Max Defaulters Rate:  But the defulters are most likely to be the one from 60 months term as the average rate of defaulters is the highest for this 22%.
### 2. Grade
#### Max Count: The most number of defaulters count are from grade B.
#### Max Defaulters Rate:  But the defulters are most likely to be the one from grade G and F as the average rate of defaulters is the highest for this 31%. The rate for F is less if we exclude sub grade F5 because of which the F grade is contributing 30%.
### 3. Sub Grade
#### Max Count: The most number of defaulters count are from Sub grade A4.
#### Max Defaulters Rate:  But the defulters are most likely to be the one from Sub grade F5 as the average rate of defaulters is the highest for this 45%.
### 4. Employment Length
#### Max Count: The most number of defaulters count are one having the length 10+ years.
#### Max Defaulters Rate:  But the defulters are most likely to be the one who haven't mentioned there employment length as the average rate of defaulters is the highest for this 21%.
### 5. Home Ownership
#### Max Count: The most number of defaulters count are from the one who lives on rent.
#### Max Defaulters Rate:  But the defulters are most likely to be the one who mentioned other in there home ownership column as the average rate of defaulters is the highest for this 18%.
### 6. Verification Status
#### Max Count: The most number of defaulters count are from the one who are not verified.
#### Max Defaulters Rate:  But the defulters are most likely to be the one from the one who are verified as the average rate of defaulters is the highest for this 16%. Cannot comment much on it.
### 7. Purpose
#### Max Count: The most number of defaulters count are the one who had taken loan for debt consolidation.
#### Max Defaulters Rate:  But the defulters are most likely to be the one who had taken loan for small businessess as the average rate of defaulters is the highest for this range 25%.
### 8. Address State
#### Max Count: The most number of defaulters count are from state CA.
#### Max Defaulters Rate:  But the defulters are most likely to be the one from state NE as the average rate of defaulters is the highest for this range 60%. This is a big number we have to take this into a serious consideration.
"""

# ---------------------------------------------------------------------------------------------------------------------

