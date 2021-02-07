# Machine Valuation Project
# Copyright (C) Paul Geertsema, 2019-2021

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# This project is loosely based on some of the ideas and approaches in
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3447683
# and
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740270


import streamlit as st
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

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
st.write('*Copyright (C) Paul Geertsema, 2019-2021 - see [code] (https://github.com/pgeertsema/machinevaluation) for licence*')
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

loaded_model = lgb.Booster(model_file=DATA+'base_model.txt')

# Index(['book_eb', 'debt_eb', 'ib_eq', 'industry', 'rate1yr_mc', 'sale_eb'], dtype='object')
X_dict = {'book_eb':book_eb, 'debt_eb':debt_eb, 'ib_eb':ib_eb, 'industry':industry, 'rate1yr_mc':rate1yr_mc, 'sale_eb':sale_eb}

# convert to list of values, sorted by feature name (order matters for predict)
X = [X_dict[key] for key in sorted(X_dict.keys())]

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

"## How Does It Work?"

'''The valuation methodology used in this app is loosely based on two of my working papers (joint with Dr Helen Lu); 
[Machine Valuation] (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3447683) and 
[Relative Valuation with Machine Learning] (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740270)'''

'''We use a machine learning algorithm to learn the historical association between accounting ratios and the EBITDA valutation multiple. The app uses this machine learning model to estimate the EBITDA ratio given the inputs provided (see input panel on the left).'''

'''The machine learning algorithm we use is [LightGBM] (https://github.com/microsoft/LightGBM) from Microsoft Research. LightGBM is an implementation of a [Gradient Boosting Machine] (https://en.wikipedia.org/wiki/Gradient_boosting), which is an [ensemble] (https://en.wikipedia.org/wiki/Ensemble_learning) of [tree-based] (https://en.wikipedia.org/wiki/Decision_tree_learning) predictors.'''

'''In this app we use an [EBITDA] (https://en.wikipedia.org/wiki/Earnings_before_interest,_taxes,_depreciation_and_amortization) valuation multiple (EnterpriseValue/EBITDA), rather than the asset multiple (EnterpriseValue/TotalAssets) used in the working papers. While EBITDA [valuation multiples] (https://en.wikipedia.org/wiki/Valuation_using_multiples) are more commonly used in industry, it suffers from the drawback that it can only be used for firms with positive EBITDA (which is why we do not use it in the working papers).'''

'''The machine learning model is trained on historical data spanning 40 years (1978 to 2019). Accounting data are from the Compustat quarterly accounting file and market data are from the CRSP monthly file. Accounting data are lagged 4 months to ensure it would have been available to the market at the time enterprise value is observed. The enterprise firm value is defined as the balance sheet equity market value adjusted for post-balance sheet stock returns plus the book value of debt at the balance sheet date. (This avoids complications due to capital market transactions post balance sheet date.) Quarterly accounting data are converted to annual equivalents by taking a 4-quarter running sum - this mitigates the effect of seasonality in reported accounting numbers. The model is trained on all common domestic US stocks (share code 10 or 11) with only a single class of stock issued. We exclude data points where any of total assets, book value, sales, EBITDA or firm value are either missing or negative. In addition we only keep firms with firm values ranked above the 20th percentile in each month (no small firms) and with EBITDA multiples between 1x and 100x (exclude outliers).'''

"## About the Author (Dr Paul Geertsema)"

'''I am a finance academic and consultant in the areas of finance, data science and machine learning. My research interests include empirical asset pricing, return predictability and the application of machine learning to finance problems. I currently teach "Modern Investment Theory and Management" (final-year undergraduate) and "Financial Machine Learning" (post-graduate) at the University of Auckland Business School. Prior to my return to academia I worked at Barclays Capital as a derivatives trader in Hong Kong and as a sell-side research analyst in London. I have also held positions at Credit Suisse, Citi and Audit New Zealand. My academic background includes a Bachelor of Accounting from Stellenbosch University, a B.Sc. Computer Science from the University of Auckland, an MBA from London Business School, a Master of Management (Economics) from Massey University and a PhD in Finance from the University of Auckland. I am a full member of Chartered Accountants Australia and New Zealand.'''

'''[Linkedin] (https://www.linkedin.com/in/paul-geertsema-5a31361/), [University profile] (https://unidirectory.auckland.ac.nz/profile/p-geertsema), [Website] (https://www.paulgeertsema.com/), [SSRN] (https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=1759387), [Github] (https://github.com/pgeertsema/machinevaluation)'''

