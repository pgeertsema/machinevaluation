# Machine Valuation Project
# Copyright (C) Paul Geertsema, 2019-2021

# Loosely based on some of the ideas and approaches in
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3447683
# and
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740270


import streamlit as st
import lightgbm as lgb
import numpy as np
import pandas as pd

# Fama French 49 industries
# see https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html

ff49 = [
    '1-Agriculture',
    '2-Food Products',
    '3-Candy & Soda',
    '4-Beer & Liquor',
    '5-Tobacco Products',
    '6-Recreation',
    '7-Entertainment',
    '8-Printing and Publishing',
    '9-Consumer Goods',
    '10-Apparel',
    '11-Healthcare',
    '12-Medical Equipment',
    '13-Pharmaceutical Products',
    '14-Chemicals',
    '15-Rubber and Plastic Products',
    '16-Textiles',
    '17-Construction Materials',
    '18-Construction',
    '19-Steel Works Etc',
    '20-Fabricated Products',
    '21-Machinery',
    '22-Electrical Equipment',
    '23-Automobiles and Trucks',
    '24-Aircraft',
    '25-Shipbuilding',
    '26-Defense',
    '27-Precious Metals',
    '28-Non-Metallic and Industrial Metal Mining',
    '29-Coal',
    '30-Petroleum and Natural Gas',
    '31-Utilities',
    '32-Communication',
    '33-Personal Services',
    '34-Business Services',
    '35-Computers',
    '36-Computer Software',
    '37-Electronic Equipment',
    '38-Measuring and Control Equipment',
    '39-Business Supplies',
    '40-Shipping Containers',
    '41-Transportation',
    '42-Wholesale',
    '43-Retail',
    '44-Restaurants',
    '45-Banking',
    '46-Insurance',
    '47-Real Estate',
    '48-Trading',
    '49-Almost Nothing or Missing'
]

DATA = 'C:/data/MV/Work/'

st.set_page_config(page_title='', page_icon=":dollar:", layout='centered', initial_sidebar_state='expanded')
st.header('Machine Valuation')
st.write('Enterprise Valuations Based on Machine Learned Associations')
st.write('*Copyright (C) Paul Geertsema, 2019-2021*')
st.write('WARNING: Experimental Research Project in Beta. Do **NOT** use for investment decisions!')

# Background
st.sidebar.header("Valuation Inputs")
selected_ff49 = st.sidebar.selectbox('Fama/French 49 Industry', ff49, index=0)
industry = ff49.index(selected_ff49) + 1 # python is zero-indexed, FF49 starts at 1
rate1yr  = st.sidebar.slider('1 Year Real Treasury Yield - %',  min_value = -5.0, max_value=12.0, step=0.1, value=2.0) / 100

# P&L
sale     = st.sidebar.number_input('Sales - $ mn', min_value=0.0, max_value=100000.0, value=600.0, step=10.0)
ebitda   = st.sidebar.number_input('EBITDA - $ mn', min_value=0.0, max_value=sale, value=100.0, step=10.0)
ib       = st.sidebar.number_input('Income After Tax - $ mn', min_value=-100000.0, max_value=ebitda, value=40.0, step=10.0)

# Balancesheet
debt     = st.sidebar.number_input('Total Debt - $ mn', min_value=0.0, max_value=100000.0, value=200.0, step=10.0)
book     = st.sidebar.number_input('Book Value of Equity - $ mn', min_value=0.0, max_value=100000.0, value=300.0, step=10.0)

# Calculated items

rate1yr_mc = rate1yr
ib_eb   = ib/ebitda 
debt_eb = debt/ebitda 
book_eb = book/ebitda 
sale_eb = sale/ebitda 

loaded_model = lgb.Booster(model_file=DATA + 'base_model.txt')

# Index(['book_eb', 'debt_eb', 'ib_eq', 'industry', 'rate1yr_mc', 'sale_eb'], dtype='object')
X_dict = {'book_eb':book_eb, 'debt_eb':debt_eb, 'ib_eb':ib_eb, 'industry':industry, 'rate1yr_mc':rate1yr_mc, 'sale_eb':sale_eb}

# convert to list of values, sorted by feature name (order matters for predict)
X = [X_dict[key] for key in sorted(X_dict.keys())]


#st.write("X = ", ",".join([str(i) for i in X]))
#st.write("X = ", X)

pred = loaded_model.predict(data=[X])
multiple = round(np.exp(pred[0]),2)
discountrate = round((1/multiple)*100,2)
value = round(ebitda*multiple,0)

st.header('Estimated EBITDA Valuation Multiple')

st.write(f"EBITDA multiple = ** {multiple} x **")

st.header('Estimated Enterprise Valuation')

st.write(f"= EBITDA x EBITDA multiple = $ {ebitda} mn x {multiple} = **$ {value} mn**")

st.header('Implied EBITDA Discount Rate (Zero Growth)')

st.write(f"= 1 /  EBITDA multiple = 1 / {multiple} = ** {discountrate} % **")

st.header("Variables Used")

X_df = pd.DataFrame(X_dict, index=[0])
st.write(X_df)

