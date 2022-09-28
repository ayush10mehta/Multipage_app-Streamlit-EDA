import streamlit as st

# st.set_page_config(layout="wide")

st.set_page_config(
page_title = "Data Cleaning",
page_icon = "🧹",
layout="wide",)


st.header("Data Cleaning")

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

#--------------------------------------------------------------------------------

loan = pd.read_csv("/Users/ayushmehta/Downloads/Upgrad/Case Study EDA/loan.csv")
st.subheader("A look at the data")
st.dataframe(loan.head())

st.subheader("Summary of the data")

st.dataframe(loan.describe())

"""
### From the above summary data we can observe that 
1. There are a total of 111 columns and 39717 rows.
2. There are so many null values in multiple columns.
"""

#--------------------------------------------------------------------------------

st.subheader("Checking for Null values.")
fig = px.imshow(loan)
st.plotly_chart(fig, use_container_width=True)

#--------------------------------------------------------------------------------

"""
#### 1. Dropping the columns having Null or 0 values.
"""
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
"""
#### 2. Dropping columns having single value.
##### The columns will be of no use if all the values will be same in the column.
"""
loan.drop(['application_type', 'policy_code', 'initial_list_status', 'pymnt_plan'], axis=1, inplace=True)

"""
#### 3. Dropping columns having null values or 0 more that 40-50%.
"""
loan.drop(['mths_since_last_delinq', 'mths_since_last_record', 'out_prncp', 'out_prncp_inv', 
           'collection_recovery_fee', 'total_rec_late_fee', 'recoveries', 'next_pymnt_d', 'pub_rec_bankruptcies',
          'delinq_2yrs'], axis=1, inplace=True)

"""
#### 4. Dropping columns having so many categorical values.
##### Columns like id, member id, etc. which can't help us in finding the insights.
"""
loan.drop(['dti', 'id', 'member_id', 'emp_title', 'desc', 'url', 'title', 'zip_code'], axis=1, inplace=True)
"""
#### 5. Dropping duplicate columns.
##### Deleted columns funded_amnt becasue it is same as loan amount
"""
fig = px.imshow(loan.corr())
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(loan, x='loan_amnt', y = 'funded_amnt')
fig.update_layout(
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

st.plotly_chart(fig, use_container_width=True)
"""
#### As we cas clearly see in the heatmap loan amount and funded amount are highly correlated with each other.
"""
loan.drop(['funded_amnt'], axis=1, inplace=True)

"""
#### 6. Dropping the columns which are of no use for the analysis.

Columns which are of no use as they are the columns used after the loan is approved.
We can ignore them, our objective is to know about factors which contirbuted for the loan default.
"""

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

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------









