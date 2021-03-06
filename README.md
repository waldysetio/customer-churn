# Customer Churn Prediction
Author: Waldy Setiono (waldysetiono@gmail.com)

**Introduction**: An energy company that has been serving corporate, SME, and residential customers is currently undergoing a significant churn mostly in its SME segment due to recent change in regulation that liberalizes the energy market. This project aims to analyze customer history and make predictive model of churn propensity to help the company deal with this issue. 

**Data**: The data used in this project are obtained from Boston Consulting Group (BCG GAMMA).

# Exploratory Data Analysis

## Importing Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import datetime
%matplotlib inline
sns.set(color_codes=True)
```

## Loading Data

**Data path**


```python
path_training_data = 'https://cdn.theforage.com/vinternships/companyassets/SKZxezskWgmFjRvj9/ml_case_training_data.csv'
path_history_data = 'https://cdn.theforage.com/vinternships/companyassets/SKZxezskWgmFjRvj9/ml_case_training_hist_data.csv'
path_churn_data = 'https://cdn.theforage.com/vinternships/companyassets/SKZxezskWgmFjRvj9/ml_case_training_output.csv'
```

**Loading data into dataframes**


```python
train_data = pd.read_csv(path_training_data)
history_data = pd.read_csv(path_history_data)
churn_data = pd.read_csv(path_churn_data) 
```

**Checking dataframes**


```python
train_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>activity_new</th>
      <th>campaign_disc_ele</th>
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_first_activ</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_base_bill_ele</th>
      <th>forecast_base_bill_year</th>
      <th>forecast_bill_12m</th>
      <th>forecast_cons</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>esoiiifxdlbkcsluxmfuacbdckommixw</td>
      <td>NaN</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>309275</td>
      <td>0</td>
      <td>10025</td>
      <td>2012-11-07</td>
      <td>2016-11-06</td>
      <td>NaN</td>
      <td>2012-11-07</td>
      <td>2015-11-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26520.30</td>
      <td>10025</td>
      <td>0.0</td>
      <td>359.29</td>
      <td>0.095919</td>
      <td>0.088347</td>
      <td>58.995952</td>
      <td>f</td>
      <td>831.80</td>
      <td>-41.76</td>
      <td>-41.76</td>
      <td>1</td>
      <td>1732.36</td>
      <td>3</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>180.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>0</td>
      <td>54946</td>
      <td>0</td>
      <td>2013-06-15</td>
      <td>2016-06-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-06-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.78</td>
      <td>0.114481</td>
      <td>0.098142</td>
      <td>40.606701</td>
      <td>t</td>
      <td>0.00</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.99</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>43.648</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4660</td>
      <td>0</td>
      <td>0</td>
      <td>2009-08-21</td>
      <td>2016-08-30</td>
      <td>NaN</td>
      <td>2009-08-21</td>
      <td>2015-08-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>189.95</td>
      <td>0</td>
      <td>0.0</td>
      <td>16.27</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.89</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>2010-04-16</td>
      <td>2016-04-16</td>
      <td>NaN</td>
      <td>2010-04-16</td>
      <td>2015-04-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>47.96</td>
      <td>0</td>
      <td>0.0</td>
      <td>38.72</td>
      <td>0.165794</td>
      <td>0.087899</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.60</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.856</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>1584</td>
      <td>0</td>
      <td>0</td>
      <td>2010-03-30</td>
      <td>2016-03-30</td>
      <td>NaN</td>
      <td>2010-03-30</td>
      <td>2015-03-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>240.04</td>
      <td>0</td>
      <td>0.0</td>
      <td>19.83</td>
      <td>0.146694</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>30.22</td>
      <td>30.22</td>
      <td>1</td>
      <td>25.46</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.200</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>32270</td>
      <td>47940</td>
      <td>0</td>
      <td>2012-05-24</td>
      <td>2016-05-08</td>
      <td>NaN</td>
      <td>2015-05-08</td>
      <td>2014-05-26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4648.01</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.57</td>
      <td>0.138305</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>t</td>
      <td>0.00</td>
      <td>27.88</td>
      <td>27.88</td>
      <td>2</td>
      <td>381.77</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>7223</td>
      <td>0</td>
      <td>181</td>
      <td>2012-08-27</td>
      <td>2016-08-27</td>
      <td>2012-08-27</td>
      <td>2012-08-27</td>
      <td>2015-08-28</td>
      <td>68.64</td>
      <td>68.64</td>
      <td>1254.65</td>
      <td>15.94</td>
      <td>631.69</td>
      <td>181</td>
      <td>0.0</td>
      <td>144.03</td>
      <td>0.100167</td>
      <td>0.091892</td>
      <td>58.995952</td>
      <td>f</td>
      <td>15.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>90.34</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>6.000</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>1844</td>
      <td>0</td>
      <td>179</td>
      <td>2012-02-08</td>
      <td>2016-02-07</td>
      <td>NaN</td>
      <td>2012-02-08</td>
      <td>2015-02-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>190.39</td>
      <td>179</td>
      <td>0.0</td>
      <td>129.60</td>
      <td>0.116900</td>
      <td>0.100015</td>
      <td>40.606701</td>
      <td>f</td>
      <td>18.05</td>
      <td>39.84</td>
      <td>39.84</td>
      <td>1</td>
      <td>20.38</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.935</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>2012-08-30</td>
      <td>2016-08-30</td>
      <td>NaN</td>
      <td>2012-08-30</td>
      <td>2015-08-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.34</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.18</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>13.08</td>
      <td>13.08</td>
      <td>1</td>
      <td>0.96</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8730</td>
      <td>0</td>
      <td>0</td>
      <td>2009-12-18</td>
      <td>2016-12-17</td>
      <td>NaN</td>
      <td>2009-12-18</td>
      <td>2015-12-21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>762.41</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.07</td>
      <td>0.167086</td>
      <td>0.088454</td>
      <td>45.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>11.84</td>
      <td>11.84</td>
      <td>1</td>
      <td>96.34</td>
      <td>6</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>10.392</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 32 columns</p>
</div>




```python
history_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-01-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-02-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-03-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-04-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-05-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192997</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-08-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192998</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-09-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192999</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-10-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193000</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-11-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193001</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-12-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
  </tbody>
</table>
<p>193002 rows ?? 8 columns</p>
</div>




```python
churn_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 2 columns</p>
</div>



**Merging train data and churn data**


```python
train = pd.merge(train_data, churn_data, on="id")
train
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>activity_new</th>
      <th>campaign_disc_ele</th>
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_first_activ</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_base_bill_ele</th>
      <th>forecast_base_bill_year</th>
      <th>forecast_bill_12m</th>
      <th>forecast_cons</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>esoiiifxdlbkcsluxmfuacbdckommixw</td>
      <td>NaN</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>309275</td>
      <td>0</td>
      <td>10025</td>
      <td>2012-11-07</td>
      <td>2016-11-06</td>
      <td>NaN</td>
      <td>2012-11-07</td>
      <td>2015-11-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26520.30</td>
      <td>10025</td>
      <td>0.0</td>
      <td>359.29</td>
      <td>0.095919</td>
      <td>0.088347</td>
      <td>58.995952</td>
      <td>f</td>
      <td>831.80</td>
      <td>-41.76</td>
      <td>-41.76</td>
      <td>1</td>
      <td>1732.36</td>
      <td>3</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>180.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>0</td>
      <td>54946</td>
      <td>0</td>
      <td>2013-06-15</td>
      <td>2016-06-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015-06-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.78</td>
      <td>0.114481</td>
      <td>0.098142</td>
      <td>40.606701</td>
      <td>t</td>
      <td>0.00</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.99</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>43.648</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4660</td>
      <td>0</td>
      <td>0</td>
      <td>2009-08-21</td>
      <td>2016-08-30</td>
      <td>NaN</td>
      <td>2009-08-21</td>
      <td>2015-08-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>189.95</td>
      <td>0</td>
      <td>0.0</td>
      <td>16.27</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.89</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>2010-04-16</td>
      <td>2016-04-16</td>
      <td>NaN</td>
      <td>2010-04-16</td>
      <td>2015-04-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>47.96</td>
      <td>0</td>
      <td>0.0</td>
      <td>38.72</td>
      <td>0.165794</td>
      <td>0.087899</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.60</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.856</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>1584</td>
      <td>0</td>
      <td>0</td>
      <td>2010-03-30</td>
      <td>2016-03-30</td>
      <td>NaN</td>
      <td>2010-03-30</td>
      <td>2015-03-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>240.04</td>
      <td>0</td>
      <td>0.0</td>
      <td>19.83</td>
      <td>0.146694</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>30.22</td>
      <td>30.22</td>
      <td>1</td>
      <td>25.46</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>32270</td>
      <td>47940</td>
      <td>0</td>
      <td>2012-05-24</td>
      <td>2016-05-08</td>
      <td>NaN</td>
      <td>2015-05-08</td>
      <td>2014-05-26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4648.01</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.57</td>
      <td>0.138305</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>t</td>
      <td>0.00</td>
      <td>27.88</td>
      <td>27.88</td>
      <td>2</td>
      <td>381.77</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>7223</td>
      <td>0</td>
      <td>181</td>
      <td>2012-08-27</td>
      <td>2016-08-27</td>
      <td>2012-08-27</td>
      <td>2012-08-27</td>
      <td>2015-08-28</td>
      <td>68.64</td>
      <td>68.64</td>
      <td>1254.65</td>
      <td>15.94</td>
      <td>631.69</td>
      <td>181</td>
      <td>0.0</td>
      <td>144.03</td>
      <td>0.100167</td>
      <td>0.091892</td>
      <td>58.995952</td>
      <td>f</td>
      <td>15.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>90.34</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>6.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>1844</td>
      <td>0</td>
      <td>179</td>
      <td>2012-02-08</td>
      <td>2016-02-07</td>
      <td>NaN</td>
      <td>2012-02-08</td>
      <td>2015-02-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>190.39</td>
      <td>179</td>
      <td>0.0</td>
      <td>129.60</td>
      <td>0.116900</td>
      <td>0.100015</td>
      <td>40.606701</td>
      <td>f</td>
      <td>18.05</td>
      <td>39.84</td>
      <td>39.84</td>
      <td>1</td>
      <td>20.38</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.935</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>2012-08-30</td>
      <td>2016-08-30</td>
      <td>NaN</td>
      <td>2012-08-30</td>
      <td>2015-08-31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.34</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.18</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>13.08</td>
      <td>13.08</td>
      <td>1</td>
      <td>0.96</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>11.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8730</td>
      <td>0</td>
      <td>0</td>
      <td>2009-12-18</td>
      <td>2016-12-17</td>
      <td>NaN</td>
      <td>2009-12-18</td>
      <td>2015-12-21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>762.41</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.07</td>
      <td>0.167086</td>
      <td>0.088454</td>
      <td>45.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>11.84</td>
      <td>11.84</td>
      <td>1</td>
      <td>96.34</td>
      <td>6</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>10.392</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 33 columns</p>
</div>



## Statistics

Data types of the train data:


```python
print(train.dtypes)
```

    id                           object
    activity_new                 object
    campaign_disc_ele           float64
    channel_sales                object
    cons_12m                      int64
    cons_gas_12m                  int64
    cons_last_month               int64
    date_activ                   object
    date_end                     object
    date_first_activ             object
    date_modif_prod              object
    date_renewal                 object
    forecast_base_bill_ele      float64
    forecast_base_bill_year     float64
    forecast_bill_12m           float64
    forecast_cons               float64
    forecast_cons_12m           float64
    forecast_cons_year            int64
    forecast_discount_energy    float64
    forecast_meter_rent_12m     float64
    forecast_price_energy_p1    float64
    forecast_price_energy_p2    float64
    forecast_price_pow_p1       float64
    has_gas                      object
    imp_cons                    float64
    margin_gross_pow_ele        float64
    margin_net_pow_ele          float64
    nb_prod_act                   int64
    net_margin                  float64
    num_years_antig               int64
    origin_up                    object
    pow_max                     float64
    churn                         int64
    dtype: object
    

Data types of the history data:


```python
print(history_data.dtypes)
```

    id               object
    price_date       object
    price_p1_var    float64
    price_p2_var    float64
    price_p3_var    float64
    price_p1_fix    float64
    price_p2_fix    float64
    price_p3_fix    float64
    dtype: object
    

This is the correlation among columns in train data and history data (0 means not correlated at all and 1 means perfectly correlated):


```python
train.corr()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>campaign_disc_ele</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_base_bill_ele</th>
      <th>forecast_base_bill_year</th>
      <th>forecast_bill_12m</th>
      <th>forecast_cons</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>campaign_disc_ele</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>cons_12m</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.471233</td>
      <td>0.919545</td>
      <td>0.132991</td>
      <td>0.132991</td>
      <td>0.149023</td>
      <td>0.133147</td>
      <td>0.165168</td>
      <td>0.139526</td>
      <td>-0.043708</td>
      <td>0.085996</td>
      <td>-0.033546</td>
      <td>0.146758</td>
      <td>-0.025418</td>
      <td>0.139353</td>
      <td>-0.065500</td>
      <td>-0.045779</td>
      <td>0.308567</td>
      <td>0.120491</td>
      <td>0.008810</td>
      <td>0.102423</td>
      <td>-0.051759</td>
    </tr>
    <tr>
      <th>cons_gas_12m</th>
      <td>NaN</td>
      <td>0.471233</td>
      <td>1.000000</td>
      <td>0.447209</td>
      <td>0.085733</td>
      <td>0.085733</td>
      <td>0.083604</td>
      <td>0.076854</td>
      <td>0.059525</td>
      <td>0.057619</td>
      <td>-0.014945</td>
      <td>0.040327</td>
      <td>-0.022416</td>
      <td>0.078456</td>
      <td>-0.027193</td>
      <td>0.060609</td>
      <td>-0.016867</td>
      <td>-0.008242</td>
      <td>0.272005</td>
      <td>0.058930</td>
      <td>-0.008626</td>
      <td>0.052365</td>
      <td>-0.040880</td>
    </tr>
    <tr>
      <th>cons_last_month</th>
      <td>NaN</td>
      <td>0.919545</td>
      <td>0.447209</td>
      <td>1.000000</td>
      <td>0.136207</td>
      <td>0.136207</td>
      <td>0.134066</td>
      <td>0.136816</td>
      <td>0.129574</td>
      <td>0.151476</td>
      <td>-0.037773</td>
      <td>0.076066</td>
      <td>-0.024242</td>
      <td>0.123164</td>
      <td>-0.020057</td>
      <td>0.153861</td>
      <td>-0.054114</td>
      <td>-0.037696</td>
      <td>0.350711</td>
      <td>0.096424</td>
      <td>0.004860</td>
      <td>0.089565</td>
      <td>-0.046931</td>
    </tr>
    <tr>
      <th>forecast_base_bill_ele</th>
      <td>NaN</td>
      <td>0.132991</td>
      <td>0.085733</td>
      <td>0.136207</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.794776</td>
      <td>0.964402</td>
      <td>0.750961</td>
      <td>0.958303</td>
      <td>0.011804</td>
      <td>0.425970</td>
      <td>-0.223693</td>
      <td>0.351376</td>
      <td>0.106436</td>
      <td>0.964402</td>
      <td>-0.088898</td>
      <td>-0.054790</td>
      <td>0.051419</td>
      <td>0.468836</td>
      <td>0.021869</td>
      <td>0.585426</td>
      <td>0.000433</td>
    </tr>
    <tr>
      <th>forecast_base_bill_year</th>
      <td>NaN</td>
      <td>0.132991</td>
      <td>0.085733</td>
      <td>0.136207</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.794776</td>
      <td>0.964402</td>
      <td>0.750961</td>
      <td>0.958303</td>
      <td>0.011804</td>
      <td>0.425970</td>
      <td>-0.223693</td>
      <td>0.351376</td>
      <td>0.106436</td>
      <td>0.964402</td>
      <td>-0.088898</td>
      <td>-0.054790</td>
      <td>0.051419</td>
      <td>0.468836</td>
      <td>0.021869</td>
      <td>0.585426</td>
      <td>0.000433</td>
    </tr>
    <tr>
      <th>forecast_bill_12m</th>
      <td>NaN</td>
      <td>0.149023</td>
      <td>0.083604</td>
      <td>0.134066</td>
      <td>0.794776</td>
      <td>0.794776</td>
      <td>1.000000</td>
      <td>0.751430</td>
      <td>0.970487</td>
      <td>0.797272</td>
      <td>0.003654</td>
      <td>0.487659</td>
      <td>-0.258372</td>
      <td>0.376494</td>
      <td>0.147365</td>
      <td>0.751430</td>
      <td>-0.129388</td>
      <td>-0.086826</td>
      <td>0.050683</td>
      <td>0.656937</td>
      <td>0.029917</td>
      <td>0.711502</td>
      <td>0.006909</td>
    </tr>
    <tr>
      <th>forecast_cons</th>
      <td>NaN</td>
      <td>0.133147</td>
      <td>0.076854</td>
      <td>0.136816</td>
      <td>0.964402</td>
      <td>0.964402</td>
      <td>0.751430</td>
      <td>1.000000</td>
      <td>0.758825</td>
      <td>0.974419</td>
      <td>0.027360</td>
      <td>0.328157</td>
      <td>-0.166692</td>
      <td>0.290593</td>
      <td>0.074713</td>
      <td>1.000000</td>
      <td>-0.111186</td>
      <td>-0.075863</td>
      <td>0.055135</td>
      <td>0.489346</td>
      <td>0.006925</td>
      <td>0.457566</td>
      <td>-0.005247</td>
    </tr>
    <tr>
      <th>forecast_cons_12m</th>
      <td>NaN</td>
      <td>0.165168</td>
      <td>0.059525</td>
      <td>0.129574</td>
      <td>0.750961</td>
      <td>0.750961</td>
      <td>0.970487</td>
      <td>0.758825</td>
      <td>1.000000</td>
      <td>0.746076</td>
      <td>0.014923</td>
      <td>0.390550</td>
      <td>-0.217315</td>
      <td>0.245845</td>
      <td>0.058169</td>
      <td>0.725550</td>
      <td>-0.184179</td>
      <td>-0.141642</td>
      <td>0.013283</td>
      <td>0.768871</td>
      <td>0.064431</td>
      <td>0.583119</td>
      <td>0.007395</td>
    </tr>
    <tr>
      <th>forecast_cons_year</th>
      <td>NaN</td>
      <td>0.139526</td>
      <td>0.057619</td>
      <td>0.151476</td>
      <td>0.958303</td>
      <td>0.958303</td>
      <td>0.797272</td>
      <td>0.974419</td>
      <td>0.746076</td>
      <td>1.000000</td>
      <td>-0.009000</td>
      <td>0.329201</td>
      <td>-0.206041</td>
      <td>0.225691</td>
      <td>0.053678</td>
      <td>0.981732</td>
      <td>-0.139177</td>
      <td>-0.106576</td>
      <td>0.013811</td>
      <td>0.537701</td>
      <td>0.066105</td>
      <td>0.442228</td>
      <td>0.002756</td>
    </tr>
    <tr>
      <th>forecast_discount_energy</th>
      <td>NaN</td>
      <td>-0.043708</td>
      <td>-0.014945</td>
      <td>-0.037773</td>
      <td>0.011804</td>
      <td>0.011804</td>
      <td>0.003654</td>
      <td>0.027360</td>
      <td>0.014923</td>
      <td>-0.009000</td>
      <td>1.000000</td>
      <td>-0.019469</td>
      <td>0.319202</td>
      <td>0.049174</td>
      <td>0.024477</td>
      <td>0.011383</td>
      <td>0.199609</td>
      <td>0.151140</td>
      <td>0.055162</td>
      <td>0.013500</td>
      <td>-0.071723</td>
      <td>-0.022646</td>
      <td>0.012344</td>
    </tr>
    <tr>
      <th>forecast_meter_rent_12m</th>
      <td>NaN</td>
      <td>0.085996</td>
      <td>0.040327</td>
      <td>0.076066</td>
      <td>0.425970</td>
      <td>0.425970</td>
      <td>0.487659</td>
      <td>0.328157</td>
      <td>0.390550</td>
      <td>0.329201</td>
      <td>-0.019469</td>
      <td>1.000000</td>
      <td>-0.558751</td>
      <td>0.636761</td>
      <td>0.013597</td>
      <td>0.296259</td>
      <td>-0.018957</td>
      <td>0.000856</td>
      <td>0.000050</td>
      <td>0.336343</td>
      <td>0.112271</td>
      <td>0.600594</td>
      <td>0.029971</td>
    </tr>
    <tr>
      <th>forecast_price_energy_p1</th>
      <td>NaN</td>
      <td>-0.033546</td>
      <td>-0.022416</td>
      <td>-0.024242</td>
      <td>-0.223693</td>
      <td>-0.223693</td>
      <td>-0.258372</td>
      <td>-0.166692</td>
      <td>-0.217315</td>
      <td>-0.206041</td>
      <td>0.319202</td>
      <td>-0.558751</td>
      <td>1.000000</td>
      <td>-0.364849</td>
      <td>0.389218</td>
      <td>-0.164657</td>
      <td>0.184782</td>
      <td>0.029119</td>
      <td>0.025854</td>
      <td>-0.185221</td>
      <td>-0.199922</td>
      <td>-0.352961</td>
      <td>-0.003337</td>
    </tr>
    <tr>
      <th>forecast_price_energy_p2</th>
      <td>NaN</td>
      <td>0.146758</td>
      <td>0.078456</td>
      <td>0.123164</td>
      <td>0.351376</td>
      <td>0.351376</td>
      <td>0.376494</td>
      <td>0.290593</td>
      <td>0.245845</td>
      <td>0.225691</td>
      <td>0.049174</td>
      <td>0.636761</td>
      <td>-0.364849</td>
      <td>1.000000</td>
      <td>-0.137244</td>
      <td>0.211061</td>
      <td>0.063421</td>
      <td>0.074075</td>
      <td>0.025949</td>
      <td>0.251761</td>
      <td>0.102997</td>
      <td>0.339373</td>
      <td>0.025597</td>
    </tr>
    <tr>
      <th>forecast_price_pow_p1</th>
      <td>NaN</td>
      <td>-0.025418</td>
      <td>-0.027193</td>
      <td>-0.020057</td>
      <td>0.106436</td>
      <td>0.106436</td>
      <td>0.147365</td>
      <td>0.074713</td>
      <td>0.058169</td>
      <td>0.053678</td>
      <td>0.024477</td>
      <td>0.013597</td>
      <td>0.389218</td>
      <td>-0.137244</td>
      <td>1.000000</td>
      <td>0.051517</td>
      <td>-0.114822</td>
      <td>-0.134192</td>
      <td>-0.011416</td>
      <td>-0.005513</td>
      <td>-0.037951</td>
      <td>0.052583</td>
      <td>0.004034</td>
    </tr>
    <tr>
      <th>imp_cons</th>
      <td>NaN</td>
      <td>0.139353</td>
      <td>0.060609</td>
      <td>0.153861</td>
      <td>0.964402</td>
      <td>0.964402</td>
      <td>0.751430</td>
      <td>1.000000</td>
      <td>0.725550</td>
      <td>0.981732</td>
      <td>0.011383</td>
      <td>0.296259</td>
      <td>-0.164657</td>
      <td>0.211061</td>
      <td>0.051517</td>
      <td>1.000000</td>
      <td>-0.122137</td>
      <td>-0.092279</td>
      <td>0.019056</td>
      <td>0.536779</td>
      <td>0.051019</td>
      <td>0.407694</td>
      <td>0.003417</td>
    </tr>
    <tr>
      <th>margin_gross_pow_ele</th>
      <td>NaN</td>
      <td>-0.065500</td>
      <td>-0.016867</td>
      <td>-0.054114</td>
      <td>-0.088898</td>
      <td>-0.088898</td>
      <td>-0.129388</td>
      <td>-0.111186</td>
      <td>-0.184179</td>
      <td>-0.139177</td>
      <td>0.199609</td>
      <td>-0.018957</td>
      <td>0.184782</td>
      <td>0.063421</td>
      <td>-0.114822</td>
      <td>-0.122137</td>
      <td>1.000000</td>
      <td>0.766521</td>
      <td>-0.043771</td>
      <td>-0.098609</td>
      <td>-0.081516</td>
      <td>-0.013654</td>
      <td>0.080158</td>
    </tr>
    <tr>
      <th>margin_net_pow_ele</th>
      <td>NaN</td>
      <td>-0.045779</td>
      <td>-0.008242</td>
      <td>-0.037696</td>
      <td>-0.054790</td>
      <td>-0.054790</td>
      <td>-0.086826</td>
      <td>-0.075863</td>
      <td>-0.141642</td>
      <td>-0.106576</td>
      <td>0.151140</td>
      <td>0.000856</td>
      <td>0.029119</td>
      <td>0.074075</td>
      <td>-0.134192</td>
      <td>-0.092279</td>
      <td>0.766521</td>
      <td>1.000000</td>
      <td>-0.032199</td>
      <td>-0.086364</td>
      <td>-0.037913</td>
      <td>-0.001202</td>
      <td>0.063187</td>
    </tr>
    <tr>
      <th>nb_prod_act</th>
      <td>NaN</td>
      <td>0.308567</td>
      <td>0.272005</td>
      <td>0.350711</td>
      <td>0.051419</td>
      <td>0.051419</td>
      <td>0.050683</td>
      <td>0.055135</td>
      <td>0.013283</td>
      <td>0.013811</td>
      <td>0.055162</td>
      <td>0.000050</td>
      <td>0.025854</td>
      <td>0.025949</td>
      <td>-0.011416</td>
      <td>0.019056</td>
      <td>-0.043771</td>
      <td>-0.032199</td>
      <td>1.000000</td>
      <td>0.004547</td>
      <td>0.009384</td>
      <td>0.018390</td>
      <td>-0.022609</td>
    </tr>
    <tr>
      <th>net_margin</th>
      <td>NaN</td>
      <td>0.120491</td>
      <td>0.058930</td>
      <td>0.096424</td>
      <td>0.468836</td>
      <td>0.468836</td>
      <td>0.656937</td>
      <td>0.489346</td>
      <td>0.768871</td>
      <td>0.537701</td>
      <td>0.013500</td>
      <td>0.336343</td>
      <td>-0.185221</td>
      <td>0.251761</td>
      <td>-0.005513</td>
      <td>0.536779</td>
      <td>-0.098609</td>
      <td>-0.086364</td>
      <td>0.004547</td>
      <td>1.000000</td>
      <td>0.034960</td>
      <td>0.457120</td>
      <td>0.029308</td>
    </tr>
    <tr>
      <th>num_years_antig</th>
      <td>NaN</td>
      <td>0.008810</td>
      <td>-0.008626</td>
      <td>0.004860</td>
      <td>0.021869</td>
      <td>0.021869</td>
      <td>0.029917</td>
      <td>0.006925</td>
      <td>0.064431</td>
      <td>0.066105</td>
      <td>-0.071723</td>
      <td>0.112271</td>
      <td>-0.199922</td>
      <td>0.102997</td>
      <td>-0.037951</td>
      <td>0.051019</td>
      <td>-0.081516</td>
      <td>-0.037913</td>
      <td>0.009384</td>
      <td>0.034960</td>
      <td>1.000000</td>
      <td>0.084789</td>
      <td>-0.071565</td>
    </tr>
    <tr>
      <th>pow_max</th>
      <td>NaN</td>
      <td>0.102423</td>
      <td>0.052365</td>
      <td>0.089565</td>
      <td>0.585426</td>
      <td>0.585426</td>
      <td>0.711502</td>
      <td>0.457566</td>
      <td>0.583119</td>
      <td>0.442228</td>
      <td>-0.022646</td>
      <td>0.600594</td>
      <td>-0.352961</td>
      <td>0.339373</td>
      <td>0.052583</td>
      <td>0.407694</td>
      <td>-0.013654</td>
      <td>-0.001202</td>
      <td>0.018390</td>
      <td>0.457120</td>
      <td>0.084789</td>
      <td>1.000000</td>
      <td>0.009456</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>NaN</td>
      <td>-0.051759</td>
      <td>-0.040880</td>
      <td>-0.046931</td>
      <td>0.000433</td>
      <td>0.000433</td>
      <td>0.006909</td>
      <td>-0.005247</td>
      <td>0.007395</td>
      <td>0.002756</td>
      <td>0.012344</td>
      <td>0.029971</td>
      <td>-0.003337</td>
      <td>0.025597</td>
      <td>0.004034</td>
      <td>0.003417</td>
      <td>0.080158</td>
      <td>0.063187</td>
      <td>-0.022609</td>
      <td>0.029308</td>
      <td>-0.071565</td>
      <td>0.009456</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
history_data.corr()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price_p1_var</th>
      <td>1.000000</td>
      <td>-0.329950</td>
      <td>-0.595257</td>
      <td>0.416443</td>
      <td>-0.630465</td>
      <td>-0.572522</td>
    </tr>
    <tr>
      <th>price_p2_var</th>
      <td>-0.329950</td>
      <td>1.000000</td>
      <td>0.828230</td>
      <td>-0.099764</td>
      <td>0.802757</td>
      <td>0.814439</td>
    </tr>
    <tr>
      <th>price_p3_var</th>
      <td>-0.595257</td>
      <td>0.828230</td>
      <td>1.000000</td>
      <td>-0.137346</td>
      <td>0.973831</td>
      <td>0.979617</td>
    </tr>
    <tr>
      <th>price_p1_fix</th>
      <td>0.416443</td>
      <td>-0.099764</td>
      <td>-0.137346</td>
      <td>1.000000</td>
      <td>0.000941</td>
      <td>-0.251511</td>
    </tr>
    <tr>
      <th>price_p2_fix</th>
      <td>-0.630465</td>
      <td>0.802757</td>
      <td>0.973831</td>
      <td>0.000941</td>
      <td>1.000000</td>
      <td>0.926955</td>
    </tr>
    <tr>
      <th>price_p3_fix</th>
      <td>-0.572522</td>
      <td>0.814439</td>
      <td>0.979617</td>
      <td>-0.251511</td>
      <td>0.926955</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Basic statistics of the data.


```python
train.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>campaign_disc_ele</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_base_bill_ele</th>
      <th>forecast_base_bill_year</th>
      <th>forecast_bill_12m</th>
      <th>forecast_cons</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>1.609600e+04</td>
      <td>1.609600e+04</td>
      <td>1.609600e+04</td>
      <td>3508.000000</td>
      <td>3508.000000</td>
      <td>3508.000000</td>
      <td>3508.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>15970.000000</td>
      <td>16096.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>16096.000000</td>
      <td>16083.000000</td>
      <td>16083.000000</td>
      <td>16096.000000</td>
      <td>16081.000000</td>
      <td>16096.000000</td>
      <td>16093.000000</td>
      <td>16096.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>1.948044e+05</td>
      <td>3.191164e+04</td>
      <td>1.946154e+04</td>
      <td>335.843857</td>
      <td>335.843857</td>
      <td>3837.441866</td>
      <td>206.845165</td>
      <td>2370.555949</td>
      <td>1907.347229</td>
      <td>0.991547</td>
      <td>70.309945</td>
      <td>0.135901</td>
      <td>0.052951</td>
      <td>43.533496</td>
      <td>196.123447</td>
      <td>22.462276</td>
      <td>21.460318</td>
      <td>1.347788</td>
      <td>217.987028</td>
      <td>5.030629</td>
      <td>20.604131</td>
      <td>0.099093</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>6.795151e+05</td>
      <td>1.775885e+05</td>
      <td>8.235676e+04</td>
      <td>649.406000</td>
      <td>649.406000</td>
      <td>5425.744327</td>
      <td>455.634288</td>
      <td>4035.085664</td>
      <td>5257.364759</td>
      <td>5.160969</td>
      <td>79.023251</td>
      <td>0.026252</td>
      <td>0.048617</td>
      <td>5.212252</td>
      <td>494.366979</td>
      <td>23.700883</td>
      <td>27.917349</td>
      <td>1.459808</td>
      <td>366.742030</td>
      <td>1.676101</td>
      <td>21.772421</td>
      <td>0.298796</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>-1.252760e+05</td>
      <td>-3.037000e+03</td>
      <td>-9.138600e+04</td>
      <td>-364.940000</td>
      <td>-364.940000</td>
      <td>-2503.480000</td>
      <td>0.000000</td>
      <td>-16689.260000</td>
      <td>-85627.000000</td>
      <td>0.000000</td>
      <td>-242.960000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.122184</td>
      <td>-9038.210000</td>
      <td>-525.540000</td>
      <td>-615.660000</td>
      <td>1.000000</td>
      <td>-4148.990000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>5.906250e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1158.175000</td>
      <td>0.000000</td>
      <td>513.230000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.230000</td>
      <td>0.115237</td>
      <td>0.000000</td>
      <td>40.606701</td>
      <td>0.000000</td>
      <td>11.960000</td>
      <td>11.950000</td>
      <td>1.000000</td>
      <td>51.970000</td>
      <td>4.000000</td>
      <td>12.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>1.533250e+04</td>
      <td>0.000000e+00</td>
      <td>9.010000e+02</td>
      <td>162.955000</td>
      <td>162.955000</td>
      <td>2187.230000</td>
      <td>42.215000</td>
      <td>1179.160000</td>
      <td>378.000000</td>
      <td>0.000000</td>
      <td>19.440000</td>
      <td>0.142881</td>
      <td>0.086163</td>
      <td>44.311378</td>
      <td>44.465000</td>
      <td>21.090000</td>
      <td>20.970000</td>
      <td>1.000000</td>
      <td>119.680000</td>
      <td>5.000000</td>
      <td>13.856000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>5.022150e+04</td>
      <td>0.000000e+00</td>
      <td>4.127000e+03</td>
      <td>396.185000</td>
      <td>396.185000</td>
      <td>4246.555000</td>
      <td>228.117500</td>
      <td>2692.077500</td>
      <td>1994.250000</td>
      <td>0.000000</td>
      <td>131.470000</td>
      <td>0.146348</td>
      <td>0.098837</td>
      <td>44.311378</td>
      <td>218.090000</td>
      <td>29.640000</td>
      <td>29.640000</td>
      <td>1.000000</td>
      <td>275.810000</td>
      <td>6.000000</td>
      <td>19.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>1.609711e+07</td>
      <td>4.188440e+06</td>
      <td>4.538720e+06</td>
      <td>12566.080000</td>
      <td>12566.080000</td>
      <td>81122.630000</td>
      <td>9682.890000</td>
      <td>103801.930000</td>
      <td>175375.000000</td>
      <td>50.000000</td>
      <td>2411.690000</td>
      <td>0.273963</td>
      <td>0.195975</td>
      <td>59.444710</td>
      <td>15042.790000</td>
      <td>374.640000</td>
      <td>374.640000</td>
      <td>32.000000</td>
      <td>24570.650000</td>
      <td>16.000000</td>
      <td>500.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
history_data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>191643.000000</td>
      <td>191643.000000</td>
      <td>191643.000000</td>
      <td>191643.000000</td>
      <td>191643.000000</td>
      <td>191643.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.140991</td>
      <td>0.054412</td>
      <td>0.030712</td>
      <td>43.325546</td>
      <td>10.698201</td>
      <td>6.455436</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.025117</td>
      <td>0.050033</td>
      <td>0.036335</td>
      <td>5.437952</td>
      <td>12.856046</td>
      <td>7.782279</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.177779</td>
      <td>-0.097752</td>
      <td>-0.065172</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.125976</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.728885</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.146033</td>
      <td>0.085483</td>
      <td>0.000000</td>
      <td>44.266930</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.151635</td>
      <td>0.101780</td>
      <td>0.072558</td>
      <td>44.444710</td>
      <td>24.339581</td>
      <td>16.226389</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.280700</td>
      <td>0.229788</td>
      <td>0.114102</td>
      <td>59.444710</td>
      <td>36.490692</td>
      <td>17.458221</td>
    </tr>
  </tbody>
</table>
</div>



**Missing data**

Missing train data


```python
missing_train_data = train.isnull()
missing_train_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>activity_new</th>
      <th>campaign_disc_ele</th>
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_first_activ</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_base_bill_ele</th>
      <th>forecast_base_bill_year</th>
      <th>forecast_bill_12m</th>
      <th>forecast_cons</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 33 columns</p>
</div>




```python
for column in missing_train_data.columns.values.tolist():
    print(column)
    print (missing_train_data[column].value_counts())
    print("")    
```

    id
    False    16096
    Name: id, dtype: int64
    
    activity_new
    True     9545
    False    6551
    Name: activity_new, dtype: int64
    
    campaign_disc_ele
    True    16096
    Name: campaign_disc_ele, dtype: int64
    
    channel_sales
    False    11878
    True      4218
    Name: channel_sales, dtype: int64
    
    cons_12m
    False    16096
    Name: cons_12m, dtype: int64
    
    cons_gas_12m
    False    16096
    Name: cons_gas_12m, dtype: int64
    
    cons_last_month
    False    16096
    Name: cons_last_month, dtype: int64
    
    date_activ
    False    16096
    Name: date_activ, dtype: int64
    
    date_end
    False    16094
    True         2
    Name: date_end, dtype: int64
    
    date_first_activ
    True     12588
    False     3508
    Name: date_first_activ, dtype: int64
    
    date_modif_prod
    False    15939
    True       157
    Name: date_modif_prod, dtype: int64
    
    date_renewal
    False    16056
    True        40
    Name: date_renewal, dtype: int64
    
    forecast_base_bill_ele
    True     12588
    False     3508
    Name: forecast_base_bill_ele, dtype: int64
    
    forecast_base_bill_year
    True     12588
    False     3508
    Name: forecast_base_bill_year, dtype: int64
    
    forecast_bill_12m
    True     12588
    False     3508
    Name: forecast_bill_12m, dtype: int64
    
    forecast_cons
    True     12588
    False     3508
    Name: forecast_cons, dtype: int64
    
    forecast_cons_12m
    False    16096
    Name: forecast_cons_12m, dtype: int64
    
    forecast_cons_year
    False    16096
    Name: forecast_cons_year, dtype: int64
    
    forecast_discount_energy
    False    15970
    True       126
    Name: forecast_discount_energy, dtype: int64
    
    forecast_meter_rent_12m
    False    16096
    Name: forecast_meter_rent_12m, dtype: int64
    
    forecast_price_energy_p1
    False    15970
    True       126
    Name: forecast_price_energy_p1, dtype: int64
    
    forecast_price_energy_p2
    False    15970
    True       126
    Name: forecast_price_energy_p2, dtype: int64
    
    forecast_price_pow_p1
    False    15970
    True       126
    Name: forecast_price_pow_p1, dtype: int64
    
    has_gas
    False    16096
    Name: has_gas, dtype: int64
    
    imp_cons
    False    16096
    Name: imp_cons, dtype: int64
    
    margin_gross_pow_ele
    False    16083
    True        13
    Name: margin_gross_pow_ele, dtype: int64
    
    margin_net_pow_ele
    False    16083
    True        13
    Name: margin_net_pow_ele, dtype: int64
    
    nb_prod_act
    False    16096
    Name: nb_prod_act, dtype: int64
    
    net_margin
    False    16081
    True        15
    Name: net_margin, dtype: int64
    
    num_years_antig
    False    16096
    Name: num_years_antig, dtype: int64
    
    origin_up
    False    16009
    True        87
    Name: origin_up, dtype: int64
    
    pow_max
    False    16093
    True         3
    Name: pow_max, dtype: int64
    
    churn
    False    16096
    Name: churn, dtype: int64
    
    


```python
pd.DataFrame({"Missing values (%)": train.isnull().sum()/len(train.index)*100})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing values (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>activity_new</th>
      <td>59.300447</td>
    </tr>
    <tr>
      <th>campaign_disc_ele</th>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>channel_sales</th>
      <td>26.205268</td>
    </tr>
    <tr>
      <th>cons_12m</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cons_gas_12m</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>cons_last_month</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>date_activ</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>date_end</th>
      <td>0.012425</td>
    </tr>
    <tr>
      <th>date_first_activ</th>
      <td>78.205765</td>
    </tr>
    <tr>
      <th>date_modif_prod</th>
      <td>0.975398</td>
    </tr>
    <tr>
      <th>date_renewal</th>
      <td>0.248509</td>
    </tr>
    <tr>
      <th>forecast_base_bill_ele</th>
      <td>78.205765</td>
    </tr>
    <tr>
      <th>forecast_base_bill_year</th>
      <td>78.205765</td>
    </tr>
    <tr>
      <th>forecast_bill_12m</th>
      <td>78.205765</td>
    </tr>
    <tr>
      <th>forecast_cons</th>
      <td>78.205765</td>
    </tr>
    <tr>
      <th>forecast_cons_12m</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>forecast_cons_year</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>forecast_discount_energy</th>
      <td>0.782803</td>
    </tr>
    <tr>
      <th>forecast_meter_rent_12m</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>forecast_price_energy_p1</th>
      <td>0.782803</td>
    </tr>
    <tr>
      <th>forecast_price_energy_p2</th>
      <td>0.782803</td>
    </tr>
    <tr>
      <th>forecast_price_pow_p1</th>
      <td>0.782803</td>
    </tr>
    <tr>
      <th>has_gas</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>imp_cons</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>margin_gross_pow_ele</th>
      <td>0.080765</td>
    </tr>
    <tr>
      <th>margin_net_pow_ele</th>
      <td>0.080765</td>
    </tr>
    <tr>
      <th>nb_prod_act</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>net_margin</th>
      <td>0.093191</td>
    </tr>
    <tr>
      <th>num_years_antig</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>origin_up</th>
      <td>0.540507</td>
    </tr>
    <tr>
      <th>pow_max</th>
      <td>0.018638</td>
    </tr>
    <tr>
      <th>churn</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Missing history data


```python
missing_history_data = history_data.isnull()
missing_history_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192997</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>192998</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>192999</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>193000</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>193001</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>193002 rows ?? 8 columns</p>
</div>




```python
for column in missing_history_data.columns.values.tolist():
    print(column)
    print (missing_history_data[column].value_counts())
    print("")    
```

    id
    False    193002
    Name: id, dtype: int64
    
    price_date
    False    193002
    Name: price_date, dtype: int64
    
    price_p1_var
    False    191643
    True       1359
    Name: price_p1_var, dtype: int64
    
    price_p2_var
    False    191643
    True       1359
    Name: price_p2_var, dtype: int64
    
    price_p3_var
    False    191643
    True       1359
    Name: price_p3_var, dtype: int64
    
    price_p1_fix
    False    191643
    True       1359
    Name: price_p1_fix, dtype: int64
    
    price_p2_fix
    False    191643
    True       1359
    Name: price_p2_fix, dtype: int64
    
    price_p3_fix
    False    191643
    True       1359
    Name: price_p3_fix, dtype: int64
    
    


```python
pd.DataFrame({"Missing values (%)": history_data.isnull().sum()/len(history_data.index)*100})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing values (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>price_date</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>price_p1_var</th>
      <td>0.704138</td>
    </tr>
    <tr>
      <th>price_p2_var</th>
      <td>0.704138</td>
    </tr>
    <tr>
      <th>price_p3_var</th>
      <td>0.704138</td>
    </tr>
    <tr>
      <th>price_p1_fix</th>
      <td>0.704138</td>
    </tr>
    <tr>
      <th>price_p2_fix</th>
      <td>0.704138</td>
    </tr>
    <tr>
      <th>price_p3_fix</th>
      <td>0.704138</td>
    </tr>
  </tbody>
</table>
</div>



## Data Visualization

**Churn Rate**


```python
churn = train[['id','churn']]
churn
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 2 columns</p>
</div>




```python
churn.columns = ["companies", "churn"]
```


```python
def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    ax = dataframe.plot(kind="bar",
                        stacked=True,
                        figsize=size_,
                        rot=rot_,
                        title=title_)
    annotate_stacked_bars(ax, textsize=14)
    plt.legend(["Retention", "Churn"], loc=legend_)
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    for p in ax.patches:
        value = str(round(p.get_height(),1))
        if value == '0.0':
          continue
        ax.annotate(value,
                    ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
                    color=colour,
                    size=textsize,
                   )
```


```python
churn_total = churn.groupby(churn["churn"]).count()
churn_percentage = churn_total/churn_total.sum()*100
```


```python
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5,5), legend_="lower right")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_39_0.png)


9.9% of the customers have churned.

**SME activity**


```python
activity = train[["id", "activity_new", "churn"]]
```


```python
activity = activity.groupby([activity["activity_new"],
                             activity["churn"]])["id"].count().unstack(level=1).sort_values(by=[0], ascending=False)
```


```python
activity.plot(kind="bar",
              figsize=(18, 10),
              width=2,
              stacked=True,
              title="SME Activity")

plt.ylabel("Number of companies")
plt.xlabel("Activity")

plt.legend(["Retention", "Churn"], loc="upper right")

plt.xticks([])
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_44_0.png)


The bar chart above shows that churn is not specifically related to any SME category in particular.

Let's take a look at the churn percentage.


```python
activity_total = activity.fillna(0)[0]+activity.fillna(0)[1]
activity_percentage = activity.fillna(0)[1]/(activity_total)*100
pd.DataFrame({"Percentage churn": activity_percentage,
 "Total companies": activity_total }).sort_values(by="Percentage churn",
 ascending=False).head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percentage churn</th>
      <th>Total companies</th>
    </tr>
    <tr>
      <th>activity_new</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xwkaesbkfsacseixxksofpddwfkbobki</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>wkwdccuiboaeaalcaawlwmldiwmpewma</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ikiucmkuisupefxcxfxxulkpwssppfuo</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>opoiuuwdmxdssidluooopfswlkkkcsxf</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>pfcocskbxlmofswiflsbcefcpufbopuo</th>
      <td>100.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>oeacexidmflusdkwuuicmpiaklkxulxm</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>wceaopxmdpccxfmcdpopulcaubcxibuw</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>kmlwkmxoocpieebifumobckeafmidpxf</th>
      <td>100.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cwouwoubfifoafkxifokoidcuoamebea</th>
      <td>66.666667</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>wfiuolfffsekuoimxdsasfwcmwssewoi</th>
      <td>50.000000</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Some companies have churned 100% but this is due to the fact that only a few companies belong to that activity.

Our predictive model is likely to struggle accurately predicting the the SME activity due to the large number of categories and low
number of companies belonging to each category.


**Sales channel**

The sales channel seems to be an important feature when predecting the churning of a user.


```python
channel = train[["id","channel_sales", "churn"]]
```


```python
channel = channel.groupby([channel["channel_sales"],
 channel["churn"]])["id"].count().unstack(level=1).fillna(0)
```


```python
channel_churn = (channel.div(channel.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)
```


```python
plot_stacked_bars(channel_churn, "Sales Channel", rot_=30)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_55_0.png)



```python
channel_total = channel.fillna(0)[0]+channel.fillna(0)[1]
channel_percentage = channel.fillna(0)[1]/(channel_total)*100
pd.DataFrame({"Churn percentage": channel_percentage,
 "Total companies": channel_total }).sort_values(by="Churn percentage",
 ascending=False).head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Churn percentage</th>
      <th>Total companies</th>
    </tr>
    <tr>
      <th>channel_sales</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>foosdfpfkusacimwkcsosbicdxkicaua</th>
      <td>12.498306</td>
      <td>7377.0</td>
    </tr>
    <tr>
      <th>usilxuppasemubllopkaafesmlibmsdf</th>
      <td>10.387812</td>
      <td>1444.0</td>
    </tr>
    <tr>
      <th>ewpakwlliwisiwduibdlfmalxowmwpci</th>
      <td>8.488613</td>
      <td>966.0</td>
    </tr>
    <tr>
      <th>lmkebamcaaclubfxadlmueccxoimlema</th>
      <td>5.595755</td>
      <td>2073.0</td>
    </tr>
    <tr>
      <th>epumfxlbckeskwekxbiuasklxalciiuu</th>
      <td>0.000000</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>fixdbufsefwooaasfcxdxadsiekoceaa</th>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>sddiedcslfslkckwlfkdpoeeailfpeds</th>
      <td>0.000000</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>



**Consumption**

We will see the distribution of the consumption over the last year and last month.


```python
consumption = train[["id", "cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons", "has_gas", "churn"]]

def plot_distribution(dataframe, column, ax, bins_=50):

    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
                         "Churn":dataframe[dataframe["churn"]==1][column]})

    temp[["Retention", "Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)

    ax.set_xlabel(column)
    ax.ticklabel_format(style='plain', axis='x')
```


```python
fig, axs = plt.subplots(nrows=4, figsize=(18,25))

plot_distribution(consumption, "cons_12m", axs[0])

plot_distribution(consumption[consumption["has_gas"] == "t"], "cons_gas_12m", axs[1])
plot_distribution(consumption, "cons_last_month", axs[2])
plot_distribution(consumption, "imp_cons", axs[3])
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_60_0.png)



```python
fig, axs = plt.subplots(nrows=4, figsize=(18,25))
# Plot histogram
sns.boxplot(consumption["cons_12m"], ax=axs[0])
sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
sns.boxplot(consumption["cons_last_month"], ax=axs[2])
sns.boxplot(consumption["imp_cons"], ax=axs[3])
# Remove scientific notation
for ax in axs:
 ax.ticklabel_format(style='plain', axis='x')
# Set x-axis limit
axs[0].set_xlim(-200000, 2000000)
axs[1].set_xlim(-200000, 2000000)
axs[2].set_xlim(-20000, 100000)
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_61_1.png)


The consumption data are highly skewed to the right. The values on the higher and the lower end of the distribution are likely to be outliers. We will do data cleaning to deal with this issue.

**Dates**


```python
dates = train[["id","date_activ","date_end", "date_modif_prod","date_renewal","churn"]].copy()
```


```python
# Transform date columns to datetime type
dates["date_activ"] = pd.to_datetime(dates["date_activ"], format='%Y-%m-%d')
dates["date_end"] = pd.to_datetime(dates["date_end"], format='%Y-%m-%d')
dates["date_modif_prod"] = pd.to_datetime(dates["date_modif_prod"], format='%Y-%m-%d')
dates["date_renewal"] = pd.to_datetime(dates["date_renewal"], format='%Y-%m-%d')
```


```python
def plot_dates(dataframe, column, fontsize_=12):
    """
    Plot monthly churn and retention distribution
    """
    # Group by month
    temp = dataframe[[column,
    "churn",
    "id"]].set_index(column).groupby([pd.Grouper(freq='M'), "churn"]).count().unstack(level=1)
    # Plot
    ax=temp.plot(kind="bar", stacked=True, figsize=(18,10), rot=0)
    # Change x-axis labels to months
    ax.set_xticklabels(map(lambda x: line_format(x), temp.index))
    # Change xlabel size
    plt.xticks(fontsize=fontsize_)
    # Rename y-axis
    plt.ylabel("Number of companies")
    # Rename legend
    plt.legend(["Retention", "Churn"], loc="upper right")
    plt.show()

def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month = label.month_name()[:1]
    if label.month_name() == "January":
        month += f'\n{label.year}'
    return month
```


```python
plot_dates(dates, "date_activ", fontsize_=8)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_67_0.png)



```python
plot_dates(dates, "date_end")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_68_0.png)



```python
plot_dates(dates, "date_modif_prod", fontsize_=8)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_69_0.png)



```python
plot_dates(dates, "date_renewal")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_70_0.png)


Visualizing data does not provide us with any useful insight. Those charts are used to see the distribution of churned companies according to the date.

**Forecast**


```python
forecast = train[["id","forecast_base_bill_ele","forecast_base_bill_year",
                  "forecast_bill_12m","forecast_cons","forecast_cons_12m",
                  "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
                  "forecast_price_energy_p1","forecast_price_energy_p2",
                  "forecast_price_pow_p1","churn"]]
```


```python
fig, axs = plt.subplots(nrows=11, figsize=(18,50))
# Plot histogram
plot_distribution(train, "forecast_base_bill_ele", axs[0])
plot_distribution(train, "forecast_base_bill_year", axs[1])
plot_distribution(train, "forecast_bill_12m", axs[2])
plot_distribution(train, "forecast_cons", axs[3])
plot_distribution(train, "forecast_cons_12m", axs[4])
plot_distribution(train, "forecast_cons_year", axs[5])
plot_distribution(train, "forecast_discount_energy", axs[6])
plot_distribution(train, "forecast_meter_rent_12m", axs[7])
plot_distribution(train, "forecast_price_energy_p1", axs[8])
plot_distribution(train, "forecast_price_energy_p2", axs[9])
plot_distribution(train, "forecast_price_pow_p1", axs[10])
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_74_0.png)


As we can see, a lot of variables are skewed to the right. We will also address this skewness later.

**Contract type**


```python
contract_type = train[["id", "has_gas", "churn"]]

contract = contract_type.groupby([contract_type["churn"],
                                  contract_type["has_gas"]])["id"].count().unstack(level=0)
                                
contract_percentage = (contract.div(contract.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)

plot_stacked_bars(contract_percentage, "Contract type (with gas)")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_77_0.png)


**Margins**


```python
margin = train[["id","margin_gross_pow_ele","margin_net_pow_ele","net_margin"]]
```


```python
fig, axs = plt.subplots(nrows=3, figsize=(18,20))
# Plot histogram
sns.boxplot(margin["margin_gross_pow_ele"], ax=axs[0])
sns.boxplot(margin["margin_net_pow_ele"],ax=axs[1])
sns.boxplot(margin["net_margin"], ax=axs[2])
# Remove scientific notation
axs[0].ticklabel_format(style='plain', axis='x')
axs[1].ticklabel_format(style='plain', axis='x')
axs[2].ticklabel_format(style='plain', axis='x')
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_80_1.png)


Outliers are present here too.

**Subscribed power**


```python
power = train[["id", "pow_max", "churn"]].fillna(0)

fig, axs = plt.subplots(nrows=1, figsize=(18,10))
plot_distribution(power, "pow_max", axs)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_83_0.png)


## Data Cleaning

**Missing data**

Let's plot the missing data.


```python
(train.isnull().sum()/len(train.index)*100).plot(kind="bar", figsize=(18,10))

plt.xlabel("Variables")
plt.ylabel("Missing values (%")
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_87_0.png)


We will remove variables with more than 60% of the values missing for simplicity reason but we might reuse these variables later on if our model is not good enough.



```python
train.drop(columns=["campaign_disc_ele", "date_first_activ",
                    "forecast_base_bill_ele", "forecast_base_bill_year",
                    "forecast_bill_12m", "forecast_cons"], inplace=True)
```


```python
train.columns
```




    Index(['id', 'activity_new', 'channel_sales', 'cons_12m', 'cons_gas_12m',
           'cons_last_month', 'date_activ', 'date_end', 'date_modif_prod',
           'date_renewal', 'forecast_cons_12m', 'forecast_cons_year',
           'forecast_discount_energy', 'forecast_meter_rent_12m',
           'forecast_price_energy_p1', 'forecast_price_energy_p2',
           'forecast_price_pow_p1', 'has_gas', 'imp_cons', 'margin_gross_pow_ele',
           'margin_net_pow_ele', 'nb_prod_act', 'net_margin', 'num_years_antig',
           'origin_up', 'pow_max', 'churn'],
          dtype='object')



**Checking duplicates**


```python
train[train.duplicated()]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>activity_new</th>
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## Formatting data

**Missing dates**

There are some kinds of way to deal with this issue and one of those is by replacing the missing values with the median as what we are gonna do now.


```python
train.loc[train["date_modif_prod"].isnull(),"date_modif_prod"] = train["date_modif_prod"].value_counts().index[0]
train.loc[train["date_end"].isnull(), "date_end"] = train["date_end"].value_counts().index[0]
train.loc[train["date_renewal"].isnull(), "date_renewal"] = train["date_renewal"].value_counts().index[0]
```

**Missing data**


```python
missing_data_pecentage = history_data.isnull().sum()/len(history_data.index)*100
```


```python
missing_data_pecentage.plot(kind="bar", figsize=(18,10))

plt.xlabel("Variables")
plt.ylabel("Missing values (%)")
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_99_0.png)


We will also use median to replace the missing values here.


```python
history_data[history_data.isnull().any(axis=1)]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>ef716222bbd97a8bdfcbb831e3575560</td>
      <td>2015-04-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>221</th>
      <td>0f5231100b2febab862f8dd8eaab3f43</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>377</th>
      <td>2f93639de582fadfbe3e86ce1c8d8f35</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>413</th>
      <td>f83c1ab1ca1d1802bb1df4d72820243c</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>461</th>
      <td>3076c6d4a060e12a049d1700d9b09cf3</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192767</th>
      <td>2dc2c9a9f6e6896d9a07d7bcbb9d0ce9</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>192788</th>
      <td>e4053a0ad6c55e4665e8e9adb9f75db5</td>
      <td>2015-03-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>192875</th>
      <td>1a788ca3bfb16ce443dcf7d75e702b5d</td>
      <td>2015-06-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>192876</th>
      <td>1a788ca3bfb16ce443dcf7d75e702b5d</td>
      <td>2015-07-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>192886</th>
      <td>d625f9e90d4af9986197444361e99235</td>
      <td>2015-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1359 rows ?? 8 columns</p>
</div>



Subtituting with median


```python
history_data.loc[history_data["price_p1_var"].isnull(),"price_p1_var"] = history_data["price_p1_var"].median()
history_data.loc[history_data["price_p2_var"].isnull(),"price_p2_var"] = history_data["price_p2_var"].median()
history_data.loc[history_data["price_p3_var"].isnull(),"price_p3_var"] = history_data["price_p3_var"].median()
history_data.loc[history_data["price_p1_fix"].isnull(),"price_p1_fix"] = history_data["price_p1_fix"].median()
history_data.loc[history_data["price_p2_fix"].isnull(),"price_p2_fix"] = history_data["price_p2_fix"].median()
history_data.loc[history_data["price_p3_fix"].isnull(),"price_p3_fix"] = history_data["price_p3_fix"].median()
```

**Formatting dates**

In order to use the dates in our churn prediction model we are going to change the representation of these dates. Instead of using the date itself, we will be transforming it in number of months. In order to make this transformation we need to change the dates to datetime and create a reference date which will be January 2016.

Train data


```python
# Transform date columns to datetime type
train["date_activ"] = pd.to_datetime(train["date_activ"], format='%Y-%m-%d')
train["date_end"] = pd.to_datetime(train["date_end"], format='%Y-%m-%d')
train["date_modif_prod"] = pd.to_datetime(train["date_modif_prod"], format='%Y-%m-%d')
train["date_renewal"] = pd.to_datetime(train["date_renewal"], format='%Y-%m-%d')
```

History data


```python
history_data["price_date"] = pd.to_datetime(history_data["price_date"], format='%Y-%m-%d')
```


```python
fig, axs = plt.subplots(nrows=7, figsize=(18,50))
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/output_110_1.png)


**Negative data**


```python
history_data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>193002.000000</td>
      <td>193002.000000</td>
      <td>193002.000000</td>
      <td>193002.000000</td>
      <td>193002.000000</td>
      <td>193002.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.141027</td>
      <td>0.054630</td>
      <td>0.030496</td>
      <td>43.332175</td>
      <td>10.622871</td>
      <td>6.409981</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.025032</td>
      <td>0.049924</td>
      <td>0.036298</td>
      <td>5.419345</td>
      <td>12.841899</td>
      <td>7.773595</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.177779</td>
      <td>-0.097752</td>
      <td>-0.065172</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.125976</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.728885</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.146033</td>
      <td>0.085483</td>
      <td>0.000000</td>
      <td>44.266930</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.151635</td>
      <td>0.101673</td>
      <td>0.072558</td>
      <td>44.444710</td>
      <td>24.339581</td>
      <td>16.226389</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.280700</td>
      <td>0.229788</td>
      <td>0.114102</td>
      <td>59.444710</td>
      <td>36.490692</td>
      <td>17.458221</td>
    </tr>
  </tbody>
</table>
</div>



We can see that there are negative values for price_p1_fix , price_p2_fix and price_p3_fix. We will replace the negative values with the median.


```python
history_data[(history_data.price_p1_fix < 0) | (history_data.price_p2_fix < 0) | (history_data.price_p3_fix < 0)]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23138</th>
      <td>951d99fe07ca94c2139f43bc37095139</td>
      <td>2015-03-01</td>
      <td>0.125976</td>
      <td>0.103395</td>
      <td>0.071536</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>28350</th>
      <td>f7bdc6fa1067cd26fd80bfb9f3fca28f</td>
      <td>2015-03-01</td>
      <td>0.131032</td>
      <td>0.108896</td>
      <td>0.076955</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>98575</th>
      <td>9b523ad5ba8aa2e524dcda5b3d54dab2</td>
      <td>2015-02-01</td>
      <td>0.129444</td>
      <td>0.106863</td>
      <td>0.075004</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>113467</th>
      <td>cfd098ee6c567eb32374c77d20571bc7</td>
      <td>2015-02-01</td>
      <td>0.123086</td>
      <td>0.100505</td>
      <td>0.068646</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>118467</th>
      <td>51d7d8a0bf6b8bd94f8c1de7942c66ea</td>
      <td>2015-07-01</td>
      <td>0.128132</td>
      <td>0.105996</td>
      <td>0.074056</td>
      <td>-0.162912</td>
      <td>-0.097752</td>
      <td>-0.065172</td>
    </tr>
    <tr>
      <th>125819</th>
      <td>decc0a647016e183ded972595cd2b9fb</td>
      <td>2015-03-01</td>
      <td>0.124937</td>
      <td>0.102814</td>
      <td>0.069071</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>128761</th>
      <td>cc214d7c05de3ee17a7691e274ac488e</td>
      <td>2015-06-01</td>
      <td>0.124675</td>
      <td>0.102539</td>
      <td>0.070596</td>
      <td>-0.162912</td>
      <td>-0.097752</td>
      <td>-0.065172</td>
    </tr>
    <tr>
      <th>141011</th>
      <td>2a4ed325054472e03cdcc9a34693be4b</td>
      <td>2015-02-01</td>
      <td>0.167317</td>
      <td>0.083347</td>
      <td>0.000000</td>
      <td>-0.177779</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>160827</th>
      <td>395a6f41bbd1a0f23a64f00645264e78</td>
      <td>2015-04-01</td>
      <td>0.121352</td>
      <td>0.098771</td>
      <td>0.066912</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
    <tr>
      <th>181811</th>
      <td>d4a84ff4ec620151ef05bdef0cf27eab</td>
      <td>2015-05-01</td>
      <td>0.125976</td>
      <td>0.103395</td>
      <td>0.071536</td>
      <td>-0.162916</td>
      <td>-0.097749</td>
      <td>-0.065166</td>
    </tr>
  </tbody>
</table>
</div>




```python
history_data.loc[history_data["price_p1_fix"] < 0,"price_p1_fix"] = history_data["price_p1_fix"].median()
history_data.loc[history_data["price_p2_fix"] < 0,"price_p2_fix"] = history_data["price_p2_fix"].median()
history_data.loc[history_data["price_p3_fix"] < 0,"price_p3_fix"] = history_data["price_p3_fix"].median()
```

## Saving data to csv


```python
train.to_csv(r'clean_train_data.csv', index = False, header=True)
```


```python
history_data.to_csv(r'clean_history_data.csv', index = False, header=True)
```
# Feature Engineering

## 1. Loading data

Let's load the data that have previously been processed and cleaned.


```python
train = pd.read_csv('https://raw.githubusercontent.com/waldysetio/customer-churn-analysis/main/clean-data/clean_train_data.csv')
history = pd.read_csv('https://raw.githubusercontent.com/waldysetio/customer-churn-analysis/main/clean-data/clean_history_data.csv')
```


```python
train
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>activity_new</th>
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>esoiiifxdlbkcsluxmfuacbdckommixw</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>309275</td>
      <td>0</td>
      <td>10025</td>
      <td>2012-11-07</td>
      <td>2016-11-06</td>
      <td>2012-11-07</td>
      <td>2015-11-09</td>
      <td>26520.30</td>
      <td>10025</td>
      <td>0.0</td>
      <td>359.29</td>
      <td>0.095919</td>
      <td>0.088347</td>
      <td>58.995952</td>
      <td>f</td>
      <td>831.80</td>
      <td>-41.76</td>
      <td>-41.76</td>
      <td>1</td>
      <td>1732.36</td>
      <td>3</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>180.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>0</td>
      <td>54946</td>
      <td>0</td>
      <td>2013-06-15</td>
      <td>2016-06-15</td>
      <td>2015-11-01</td>
      <td>2015-06-23</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.78</td>
      <td>0.114481</td>
      <td>0.098142</td>
      <td>40.606701</td>
      <td>t</td>
      <td>0.00</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.99</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>43.648</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4660</td>
      <td>0</td>
      <td>0</td>
      <td>2009-08-21</td>
      <td>2016-08-30</td>
      <td>2009-08-21</td>
      <td>2015-08-31</td>
      <td>189.95</td>
      <td>0</td>
      <td>0.0</td>
      <td>16.27</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.89</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>2010-04-16</td>
      <td>2016-04-16</td>
      <td>2010-04-16</td>
      <td>2015-04-17</td>
      <td>47.96</td>
      <td>0</td>
      <td>0.0</td>
      <td>38.72</td>
      <td>0.165794</td>
      <td>0.087899</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.60</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.856</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>NaN</td>
      <td>lmkebamcaaclubfxadlmueccxoimlema</td>
      <td>1584</td>
      <td>0</td>
      <td>0</td>
      <td>2010-03-30</td>
      <td>2016-03-30</td>
      <td>2010-03-30</td>
      <td>2015-03-31</td>
      <td>240.04</td>
      <td>0</td>
      <td>0.0</td>
      <td>19.83</td>
      <td>0.146694</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>30.22</td>
      <td>30.22</td>
      <td>1</td>
      <td>25.46</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>32270</td>
      <td>47940</td>
      <td>0</td>
      <td>2012-05-24</td>
      <td>2016-05-08</td>
      <td>2015-05-08</td>
      <td>2014-05-26</td>
      <td>4648.01</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.57</td>
      <td>0.138305</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>t</td>
      <td>0.00</td>
      <td>27.88</td>
      <td>27.88</td>
      <td>2</td>
      <td>381.77</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>7223</td>
      <td>0</td>
      <td>181</td>
      <td>2012-08-27</td>
      <td>2016-08-27</td>
      <td>2012-08-27</td>
      <td>2015-08-28</td>
      <td>631.69</td>
      <td>181</td>
      <td>0.0</td>
      <td>144.03</td>
      <td>0.100167</td>
      <td>0.091892</td>
      <td>58.995952</td>
      <td>f</td>
      <td>15.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>90.34</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>6.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>1844</td>
      <td>0</td>
      <td>179</td>
      <td>2012-02-08</td>
      <td>2016-02-07</td>
      <td>2012-02-08</td>
      <td>2015-02-09</td>
      <td>190.39</td>
      <td>179</td>
      <td>0.0</td>
      <td>129.60</td>
      <td>0.116900</td>
      <td>0.100015</td>
      <td>40.606701</td>
      <td>f</td>
      <td>18.05</td>
      <td>39.84</td>
      <td>39.84</td>
      <td>1</td>
      <td>20.38</td>
      <td>4</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>15.935</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>NaN</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>2012-08-30</td>
      <td>2016-08-30</td>
      <td>2012-08-30</td>
      <td>2015-08-31</td>
      <td>19.34</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.18</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>13.08</td>
      <td>13.08</td>
      <td>1</td>
      <td>0.96</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>11.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8730</td>
      <td>0</td>
      <td>0</td>
      <td>2009-12-18</td>
      <td>2016-12-17</td>
      <td>2009-12-18</td>
      <td>2015-12-21</td>
      <td>762.41</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.07</td>
      <td>0.167086</td>
      <td>0.088454</td>
      <td>45.311378</td>
      <td>f</td>
      <td>0.00</td>
      <td>11.84</td>
      <td>11.84</td>
      <td>1</td>
      <td>96.34</td>
      <td>6</td>
      <td>ldkssxwpmemidmecebumciepifcamkci</td>
      <td>10.392</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 27 columns</p>
</div>




```python
history
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-01-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-02-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-03-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-04-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-05-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192997</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-08-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192998</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-09-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192999</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-10-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193000</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-11-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193001</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-12-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
  </tbody>
</table>
<p>193002 rows ?? 8 columns</p>
</div>



## 2. Feature Engineering

We will create new features using the average of the year, the last six month, and the last three month to our model.


```python
mean_year = history.groupby(["id"]).mean().reset_index()
```


```python
mean_6m = history[history["price_date"] > "2015-06-01"].groupby(["id"]).mean().reset_index()
```


```python
mean_3m = history[history["price_date"] > "2015-10-01"].groupby(["id"]).mean().reset_index()
```

Let's combine each of them in a single dataframe.


```python
mean_year = mean_year.rename(index=str, columns={"price_p1_var": "mean_year_price_p1_var",
                                                 "price_p2_var": "mean_year_price_p2_var",
                                                 "price_p3_var": "mean_year_price_p3_var",
                                                 "price_p1_fix": "mean_year_price_p1_fix",
                                                 "price_p2_fix": "mean_year_price_p2_fix",
                                                 "price_p3_fix": "mean_year_price_p3_fix",})
mean_year["mean_year_price_p1"] = mean_year["mean_year_price_p1_var"] + mean_year["mean_year_price_p1_fix"]
mean_year["mean_year_price_p2"] = mean_year["mean_year_price_p2_var"] + mean_year["mean_year_price_p2_fix"]
mean_year["mean_year_price_p3"] = mean_year["mean_year_price_p3_var"] + mean_year["mean_year_price_p3_fix"]
```


```python
mean_6m = mean_6m.rename(index=str, columns={"price_p1_var": "mean_6m_price_p1_var",
                                             "price_p2_var": "mean_6m_price_p2_var",
                                             "price_p3_var": "mean_6m_price_p3_var",
                                             "price_p1_fix": "mean_6m_price_p1_fix",
                                             "price_p2_fix": "mean_6m_price_p2_fix",
                                             "price_p3_fix": "mean_6m_price_p3_fix",})
mean_6m["mean_6m_price_p1"] = mean_6m["mean_6m_price_p1_var"] + mean_6m["mean_6m_price_p1_fix"]
mean_6m["mean_6m_price_p2"] = mean_6m["mean_6m_price_p2_var"] + mean_6m["mean_6m_price_p2_fix"]
mean_6m["mean_6m_price_p3"] = mean_6m["mean_6m_price_p3_var"] + mean_6m["mean_6m_price_p3_fix"]
```


```python
mean_3m = mean_3m.rename(index=str, columns={"price_p1_var": "mean_3m_price_p1_var",
                                             "price_p2_var": "mean_3m_price_p2_var",
                                             "price_p3_var": "mean_3m_price_p3_var",
                                             "price_p1_fix": "mean_3m_price_p1_fix",
                                             "price_p2_fix": "mean_3m_price_p2_fix",
                                             "price_p3_fix": "mean_3m_price_p3_fix",})
mean_3m["mean_3m_price_p1"] = mean_3m["mean_3m_price_p1_var"] + mean_3m["mean_3m_price_p1_fix"]
mean_3m["mean_3m_price_p2"] = mean_3m["mean_3m_price_p2_var"] + mean_3m["mean_3m_price_p2_fix"]
mean_3m["mean_3m_price_p3"] = mean_3m["mean_3m_price_p3_var"] + mean_3m["mean_3m_price_p3_fix"]
```

Since it's not convincing enough that mean_6m and mean_3m can help the predictoin model, we will use only mean_year for now.


```python
# features = pd.merge(mean_year, mean_6m, on="id")
# features = pd.merge(mean_year, mean_3m, on="id")
features = mean_year
```

**Feature engineering**

We are going to define a new variable to know the correlation between the tenure of the customers and the churn rate.


```python
train['date_end'] = pd.to_datetime(train['date_end'])
train['date_activ'] = pd.to_datetime(train['date_activ'])
```


```python
train["tenure"] = ((train["date_end"]-train["date_activ"])/ np.timedelta64(1, "Y")).astype(int)
```


```python
tenure = train[["tenure", "churn", "id"]].groupby(["tenure", "churn"])["id"].count().unstack(level=1)
tenure_percentage = (tenure.div(tenure.sum(axis=1), axis=0)*100)
```


```python
tenure.plot(kind="bar",
            figsize=(18,10),
            stacked=True,
            rot=0,
            title= "Tenure")

plt.legend(["Retention", "Churn"], loc="upper right")

plt.ylabel("No. of companies")
plt.xlabel("No. of years")
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_23_0.png)


As we can see, churn is significantly lower for companies which joined recently or a long time ago. Higher number of churners are within those with 3 to 7 years of tenure.

We will also transform the dates provided in such a way that we can make more sense out of those.

months_activ : Number of months active until reference date (Jan 2016)

months_to_end : Number of months of the contract left at reference date (Jan 2016)

months_modif_prod : Number of months since last modification at reference date (Jan 2016)

months_renewal : Number of months since last renewal at reference date (Jan 2016)


```python
def convert_months(reference_date, dataframe, column):

    time_delta = REFERENCE_DATE - dataframe[column]
    months = (time_delta / np.timedelta64(1, "M")).astype(int)
    return months
```


```python
REFERENCE_DATE = datetime.datetime(2016,1,1)
```


```python
train['date_modif_prod'] = pd.to_datetime(train['date_modif_prod'])
train['date_renewal'] = pd.to_datetime(train['date_renewal'])
```


```python
train["months_activ"] = convert_months(REFERENCE_DATE, train, "date_activ")
train["months_to_end"] = -convert_months(REFERENCE_DATE, train, "date_end")
train["months_modif_prod"] = convert_months(REFERENCE_DATE, train, "date_modif_prod")
train["months_renewal"] = convert_months(REFERENCE_DATE, train, "date_renewal")
```

Now let's see if there is some insight.


```python
def plot_churn_by_month(dataframe, column, fontsize_=11):

    temp = dataframe[[column, "churn", "id"]].groupby([column, "churn"])["id"].count().unstack(level=1)
    temp.plot(kind="bar",
              figsize=(18,10),
              stacked=True,
              rot=0,
              title= column)

    plt.legend(["Retention", "Churn"], loc="upper right")

    plt.ylabel("No. of companies")
    plt.xlabel("No. of months")

    plt.xticks(fontsize=fontsize_)
    plt.show()
```


```python
plot_churn_by_month(train, "months_activ", 6)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_36_0.png)



```python
plot_churn_by_month(train, "months_to_end")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_37_0.png)



```python
plot_churn_by_month(train, "months_modif_prod", 6)
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_38_0.png)



```python
plot_churn_by_month(train, "months_renewal")
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_39_0.png)


Remove the date columns


```python
train.drop(columns=["date_activ", "date_end", "date_modif_prod", "date_renewal"],inplace=True)
```

Now let's do onehot encoding for the colunmn "has_gas"


```python
train["has_gas"]=train["has_gas"].replace(["t", "f"],[1,0])
```

**Categorical data and dummy variables**



Categorical data of "channel_sales"

We want to convert each category into a new dummy variable which will have 0 s and 1 s depending whether than entry belongs to that particular category or not.

First, we will replace the NaN values with a string called null_values_channel.


```python
train["channel_sales"] = train["channel_sales"].fillna("null_values_channel")
```


```python
# Transform to categorical data type
train["channel_sales"] = train["channel_sales"].astype("category")
```


```python
pd.DataFrame({"Samples in category": train["channel_sales"].value_counts()})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Samples in category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>foosdfpfkusacimwkcsosbicdxkicaua</th>
      <td>7377</td>
    </tr>
    <tr>
      <th>null_values_channel</th>
      <td>4218</td>
    </tr>
    <tr>
      <th>lmkebamcaaclubfxadlmueccxoimlema</th>
      <td>2073</td>
    </tr>
    <tr>
      <th>usilxuppasemubllopkaafesmlibmsdf</th>
      <td>1444</td>
    </tr>
    <tr>
      <th>ewpakwlliwisiwduibdlfmalxowmwpci</th>
      <td>966</td>
    </tr>
    <tr>
      <th>sddiedcslfslkckwlfkdpoeeailfpeds</th>
      <td>12</td>
    </tr>
    <tr>
      <th>epumfxlbckeskwekxbiuasklxalciiuu</th>
      <td>4</td>
    </tr>
    <tr>
      <th>fixdbufsefwooaasfcxdxadsiekoceaa</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We will create 8 different dummy variables . Each variable will become a different column.


```python
#Create dummy variables
categories_channel = pd.get_dummies(train["channel_sales"], prefix = "channel")
```


```python
#Rename columns for simplicity
categories_channel.columns = [col_name[:11] for col_name in categories_channel.columns]
```


```python
categories_channel
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channel_epu</th>
      <th>channel_ewp</th>
      <th>channel_fix</th>
      <th>channel_foo</th>
      <th>channel_lmk</th>
      <th>channel_nul</th>
      <th>channel_sdd</th>
      <th>channel_usi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 8 columns</p>
</div>



Now we will remove the channel_nul to avoid multicollinearity.


```python
categories_channel.drop(columns=["channel_nul"],inplace=True)
```

Categorical data of "origin_up"

First of all let's replace the Nan values with a string called null_values_origin.


```python
train["origin_up"] = train["origin_up"].fillna("null_values_origin")
```

Now transform the origin_up column into categorical data type.


```python
train["origin_up"] = train["origin_up"].astype("category")
```


```python
pd.DataFrame({"Samples in category": train["origin_up"].value_counts()})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Samples in category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lxidpiddsbxsbosboudacockeimpuepw</th>
      <td>7825</td>
    </tr>
    <tr>
      <th>kamkkxfxxuwbdslkwifmmcsiusiuosws</th>
      <td>4517</td>
    </tr>
    <tr>
      <th>ldkssxwpmemidmecebumciepifcamkci</th>
      <td>3664</td>
    </tr>
    <tr>
      <th>null_values_origin</th>
      <td>87</td>
    </tr>
    <tr>
      <th>usapbepcfoloekilkwsdiboslwaxobdp</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ewxeelcelemmiwuafmddpobolfuxioce</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We will create 6 different dummy variables.


```python
# Create dummy variables
categories_origin = pd.get_dummies(train["origin_up"], prefix = "origin")
# Rename columns for simplicity
categories_origin.columns = [col_name[:10] for col_name in categories_origin.columns]
```


```python
categories_origin
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_ewx</th>
      <th>origin_kam</th>
      <th>origin_ldk</th>
      <th>origin_lxi</th>
      <th>origin_nul</th>
      <th>origin_usa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 6 columns</p>
</div>



Now remove one column to avoid the dummy variable trap.


```python
categories_origin.drop(columns=["origin_nul"],inplace=True)
```

**Categorical data - Feature engineering**

First, let's replace the Nan values with a string called null_values_activity.


```python
train["activity_new"] = train["activity_new"].fillna("null_values_activity")
```


```python
categories_activity = pd.DataFrame({"Activity samples":train["activity_new"].value_counts()})
categories_activity
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Activity samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>null_values_activity</th>
      <td>9545</td>
    </tr>
    <tr>
      <th>apdekpcbwosbxepsfxclislboipuxpop</th>
      <td>1577</td>
    </tr>
    <tr>
      <th>kkklcdamwfafdcfwofuscwfwadblfmce</th>
      <td>422</td>
    </tr>
    <tr>
      <th>kwuslieomapmswolewpobpplkaooaaew</th>
      <td>230</td>
    </tr>
    <tr>
      <th>fmwdwsxillemwbbwelxsampiuwwpcdcb</th>
      <td>219</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>wcweaxoxmefpfbpfbifcwmfeeubwwkmc</th>
      <td>1</td>
    </tr>
    <tr>
      <th>kcioolmpmuxpoeuicskiafwcmadeflfc</th>
      <td>1</td>
    </tr>
    <tr>
      <th>kpkesxdaobicuwwkukxwmdpsbowwbomd</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ocskiadudoffubcmbomoslkcddxwfsuf</th>
      <td>1</td>
    </tr>
    <tr>
      <th>beplffiwdfsmiuodulsfscelscscbdix</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>420 rows ?? 1 columns</p>
</div>



As we can see, there are too many categories with very few number of samples. So we will replace any category with less than 75 samples as
null_values_category.


```python
# Get the categories with less than 75 samples
to_replace = list(categories_activity[categories_activity["Activity samples"] <= 75].index)
# Replace them with `null_values_categories`
train["activity_new"]=train["activity_new"].replace(to_replace,"null_values_activity")
```


```python
# Create dummy variables
categories_activity = pd.get_dummies(train["activity_new"], prefix = "activity")
# Rename columns for simplicity
categories_activity.columns = [col_name[:12] for col_name in categories_activity.columns]
```


```python
categories_activity
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_apd</th>
      <th>activity_ckf</th>
      <th>activity_clu</th>
      <th>activity_cwo</th>
      <th>activity_fmw</th>
      <th>activity_kkk</th>
      <th>activity_kwu</th>
      <th>activity_nul</th>
      <th>activity_sfi</th>
      <th>activity_wxe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 10 columns</p>
</div>



Remove one column to avoid the dummy variable trap.


```python
categories_activity.drop(columns=["activity_nul"],inplace=True)
```

Merge dummy variables to main dataframe.


```python
# Use common index to merge
train = pd.merge(train, categories_channel, left_index=True, right_index=True)
train = pd.merge(train, categories_origin, left_index=True, right_index=True)
train = pd.merge(train, categories_activity, left_index=True, right_index=True)
```


```python
train.drop(columns=["channel_sales", "origin_up", "activity_new"],inplace=True)
```


```python
train
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>pow_max</th>
      <th>churn</th>
      <th>tenure</th>
      <th>months_activ</th>
      <th>months_to_end</th>
      <th>months_modif_prod</th>
      <th>months_renewal</th>
      <th>channel_epu</th>
      <th>channel_ewp</th>
      <th>channel_fix</th>
      <th>channel_foo</th>
      <th>channel_lmk</th>
      <th>channel_sdd</th>
      <th>channel_usi</th>
      <th>origin_ewx</th>
      <th>origin_kam</th>
      <th>origin_ldk</th>
      <th>origin_lxi</th>
      <th>origin_usa</th>
      <th>activity_apd</th>
      <th>activity_ckf</th>
      <th>activity_clu</th>
      <th>activity_cwo</th>
      <th>activity_fmw</th>
      <th>activity_kkk</th>
      <th>activity_kwu</th>
      <th>activity_sfi</th>
      <th>activity_wxe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>309275</td>
      <td>0</td>
      <td>10025</td>
      <td>26520.30</td>
      <td>10025</td>
      <td>0.0</td>
      <td>359.29</td>
      <td>0.095919</td>
      <td>0.088347</td>
      <td>58.995952</td>
      <td>0</td>
      <td>831.80</td>
      <td>-41.76</td>
      <td>-41.76</td>
      <td>1</td>
      <td>1732.36</td>
      <td>3</td>
      <td>180.000</td>
      <td>0</td>
      <td>3</td>
      <td>37</td>
      <td>10</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>0</td>
      <td>54946</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.78</td>
      <td>0.114481</td>
      <td>0.098142</td>
      <td>40.606701</td>
      <td>1</td>
      <td>0.00</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.99</td>
      <td>3</td>
      <td>43.648</td>
      <td>1</td>
      <td>3</td>
      <td>30</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>4660</td>
      <td>0</td>
      <td>0</td>
      <td>189.95</td>
      <td>0</td>
      <td>0.0</td>
      <td>16.27</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.00</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.89</td>
      <td>6</td>
      <td>13.800</td>
      <td>0</td>
      <td>7</td>
      <td>76</td>
      <td>7</td>
      <td>76</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>47.96</td>
      <td>0</td>
      <td>0.0</td>
      <td>38.72</td>
      <td>0.165794</td>
      <td>0.087899</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.00</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.60</td>
      <td>6</td>
      <td>13.856</td>
      <td>0</td>
      <td>6</td>
      <td>68</td>
      <td>3</td>
      <td>68</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>1584</td>
      <td>0</td>
      <td>0</td>
      <td>240.04</td>
      <td>0</td>
      <td>0.0</td>
      <td>19.83</td>
      <td>0.146694</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.00</td>
      <td>30.22</td>
      <td>30.22</td>
      <td>1</td>
      <td>25.46</td>
      <td>6</td>
      <td>13.200</td>
      <td>0</td>
      <td>6</td>
      <td>69</td>
      <td>2</td>
      <td>69</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>32270</td>
      <td>47940</td>
      <td>0</td>
      <td>4648.01</td>
      <td>0</td>
      <td>0.0</td>
      <td>18.57</td>
      <td>0.138305</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>1</td>
      <td>0.00</td>
      <td>27.88</td>
      <td>27.88</td>
      <td>2</td>
      <td>381.77</td>
      <td>4</td>
      <td>15.000</td>
      <td>0</td>
      <td>3</td>
      <td>43</td>
      <td>4</td>
      <td>7</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>7223</td>
      <td>0</td>
      <td>181</td>
      <td>631.69</td>
      <td>181</td>
      <td>0.0</td>
      <td>144.03</td>
      <td>0.100167</td>
      <td>0.091892</td>
      <td>58.995952</td>
      <td>0</td>
      <td>15.94</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>90.34</td>
      <td>3</td>
      <td>6.000</td>
      <td>1</td>
      <td>4</td>
      <td>40</td>
      <td>7</td>
      <td>40</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>1844</td>
      <td>0</td>
      <td>179</td>
      <td>190.39</td>
      <td>179</td>
      <td>0.0</td>
      <td>129.60</td>
      <td>0.116900</td>
      <td>0.100015</td>
      <td>40.606701</td>
      <td>0</td>
      <td>18.05</td>
      <td>39.84</td>
      <td>39.84</td>
      <td>1</td>
      <td>20.38</td>
      <td>4</td>
      <td>15.935</td>
      <td>1</td>
      <td>3</td>
      <td>46</td>
      <td>1</td>
      <td>46</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>19.34</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.18</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.00</td>
      <td>13.08</td>
      <td>13.08</td>
      <td>1</td>
      <td>0.96</td>
      <td>3</td>
      <td>11.000</td>
      <td>0</td>
      <td>4</td>
      <td>40</td>
      <td>7</td>
      <td>40</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>8730</td>
      <td>0</td>
      <td>0</td>
      <td>762.41</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.07</td>
      <td>0.167086</td>
      <td>0.088454</td>
      <td>45.311378</td>
      <td>0</td>
      <td>0.00</td>
      <td>11.84</td>
      <td>11.84</td>
      <td>1</td>
      <td>96.34</td>
      <td>6</td>
      <td>10.392</td>
      <td>0</td>
      <td>6</td>
      <td>72</td>
      <td>11</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 46 columns</p>
</div>



**Log transformation**

We know from the previous exploratory data analysis that a lot of the variables we are dealing with are highly skewed to the right and this could make our model perform poorly.

There are several ways to deal with skewness such as square root, cubic root, and log. In this case, we are going to use log transformation which is usually recommended for right skewed data.


```python
train.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>pow_max</th>
      <th>churn</th>
      <th>tenure</th>
      <th>months_activ</th>
      <th>months_to_end</th>
      <th>months_modif_prod</th>
      <th>months_renewal</th>
      <th>channel_epu</th>
      <th>channel_ewp</th>
      <th>channel_fix</th>
      <th>channel_foo</th>
      <th>channel_lmk</th>
      <th>channel_sdd</th>
      <th>channel_usi</th>
      <th>origin_ewx</th>
      <th>origin_kam</th>
      <th>origin_ldk</th>
      <th>origin_lxi</th>
      <th>origin_usa</th>
      <th>activity_apd</th>
      <th>activity_ckf</th>
      <th>activity_clu</th>
      <th>activity_cwo</th>
      <th>activity_fmw</th>
      <th>activity_kkk</th>
      <th>activity_kwu</th>
      <th>activity_sfi</th>
      <th>activity_wxe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.609600e+04</td>
      <td>1.609600e+04</td>
      <td>1.609600e+04</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>15970.000000</td>
      <td>16096.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16083.000000</td>
      <td>16083.000000</td>
      <td>16096.000000</td>
      <td>16081.000000</td>
      <td>16096.000000</td>
      <td>16093.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.948044e+05</td>
      <td>3.191164e+04</td>
      <td>1.946154e+04</td>
      <td>2370.555949</td>
      <td>1907.347229</td>
      <td>0.991547</td>
      <td>70.309945</td>
      <td>0.135901</td>
      <td>0.052951</td>
      <td>43.533496</td>
      <td>0.184145</td>
      <td>196.123447</td>
      <td>22.462276</td>
      <td>21.460318</td>
      <td>1.347788</td>
      <td>217.987028</td>
      <td>5.030629</td>
      <td>20.604131</td>
      <td>0.099093</td>
      <td>5.329958</td>
      <td>58.929858</td>
      <td>6.376615</td>
      <td>35.741240</td>
      <td>4.924640</td>
      <td>0.000249</td>
      <td>0.060015</td>
      <td>0.000124</td>
      <td>0.458313</td>
      <td>0.128790</td>
      <td>0.000746</td>
      <td>0.089712</td>
      <td>0.000062</td>
      <td>0.280629</td>
      <td>0.227634</td>
      <td>0.486146</td>
      <td>0.000124</td>
      <td>0.097975</td>
      <td>0.011742</td>
      <td>0.007393</td>
      <td>0.007580</td>
      <td>0.013606</td>
      <td>0.026218</td>
      <td>0.014289</td>
      <td>0.005157</td>
      <td>0.007393</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.795151e+05</td>
      <td>1.775885e+05</td>
      <td>8.235676e+04</td>
      <td>4035.085664</td>
      <td>5257.364759</td>
      <td>5.160969</td>
      <td>79.023251</td>
      <td>0.026252</td>
      <td>0.048617</td>
      <td>5.212252</td>
      <td>0.387615</td>
      <td>494.366979</td>
      <td>23.700883</td>
      <td>27.917349</td>
      <td>1.459808</td>
      <td>366.742030</td>
      <td>1.676101</td>
      <td>21.772421</td>
      <td>0.298796</td>
      <td>1.749248</td>
      <td>20.125024</td>
      <td>3.633479</td>
      <td>30.609746</td>
      <td>3.812127</td>
      <td>0.015763</td>
      <td>0.237522</td>
      <td>0.011147</td>
      <td>0.498275</td>
      <td>0.334978</td>
      <td>0.027295</td>
      <td>0.285777</td>
      <td>0.007882</td>
      <td>0.449320</td>
      <td>0.419318</td>
      <td>0.499824</td>
      <td>0.011147</td>
      <td>0.297290</td>
      <td>0.107726</td>
      <td>0.085668</td>
      <td>0.086733</td>
      <td>0.115852</td>
      <td>0.159787</td>
      <td>0.118684</td>
      <td>0.071626</td>
      <td>0.085668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.252760e+05</td>
      <td>-3.037000e+03</td>
      <td>-9.138600e+04</td>
      <td>-16689.260000</td>
      <td>-85627.000000</td>
      <td>0.000000</td>
      <td>-242.960000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.122184</td>
      <td>0.000000</td>
      <td>-9038.210000</td>
      <td>-525.540000</td>
      <td>-615.660000</td>
      <td>1.000000</td>
      <td>-4148.990000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>-112.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.906250e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>513.230000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.230000</td>
      <td>0.115237</td>
      <td>0.000000</td>
      <td>40.606701</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.960000</td>
      <td>11.950000</td>
      <td>1.000000</td>
      <td>51.970000</td>
      <td>4.000000</td>
      <td>12.500000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>44.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.533250e+04</td>
      <td>0.000000e+00</td>
      <td>9.010000e+02</td>
      <td>1179.160000</td>
      <td>378.000000</td>
      <td>0.000000</td>
      <td>19.440000</td>
      <td>0.142881</td>
      <td>0.086163</td>
      <td>44.311378</td>
      <td>0.000000</td>
      <td>44.465000</td>
      <td>21.090000</td>
      <td>20.970000</td>
      <td>1.000000</td>
      <td>119.680000</td>
      <td>5.000000</td>
      <td>13.856000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>57.000000</td>
      <td>6.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.022150e+04</td>
      <td>0.000000e+00</td>
      <td>4.127000e+03</td>
      <td>2692.077500</td>
      <td>1994.250000</td>
      <td>0.000000</td>
      <td>131.470000</td>
      <td>0.146348</td>
      <td>0.098837</td>
      <td>44.311378</td>
      <td>0.000000</td>
      <td>218.090000</td>
      <td>29.640000</td>
      <td>29.640000</td>
      <td>1.000000</td>
      <td>275.810000</td>
      <td>6.000000</td>
      <td>19.800000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>71.000000</td>
      <td>9.000000</td>
      <td>64.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.609711e+07</td>
      <td>4.188440e+06</td>
      <td>4.538720e+06</td>
      <td>103801.930000</td>
      <td>175375.000000</td>
      <td>50.000000</td>
      <td>2411.690000</td>
      <td>0.273963</td>
      <td>0.195975</td>
      <td>59.444710</td>
      <td>1.000000</td>
      <td>15042.790000</td>
      <td>374.640000</td>
      <td>374.640000</td>
      <td>32.000000</td>
      <td>24570.650000</td>
      <td>16.000000</td>
      <td>500.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>185.000000</td>
      <td>17.000000</td>
      <td>185.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the standard deviation is very high for some variables.

Now we will convert the negative values to NaN since Log transformation does not work with negative data.

We can't apply Log transformation to 0 too, so we will add a constant 1.


```python
# Remove negative values
train.loc[train.cons_12m < 0,"cons_12m"] = np.nan
train.loc[train.cons_gas_12m < 0,"cons_gas_12m"] = np.nan
train.loc[train.cons_last_month < 0,"cons_last_month"] = np.nan
train.loc[train.forecast_cons_12m < 0,"forecast_cons_12m"] = np.nan
train.loc[train.forecast_cons_year < 0,"forecast_cons_year"] = np.nan
train.loc[train.forecast_meter_rent_12m < 0,"forecast_meter_rent_12m"] = np.nan
train.loc[train.imp_cons < 0,"imp_cons"] = np.nan
```


```python
# Apply log10 transformation
train["cons_12m"] = np.log10(train["cons_12m"]+1)
train["cons_gas_12m"] = np.log10(train["cons_gas_12m"]+1)
train["cons_last_month"] = np.log10(train["cons_last_month"]+1)
train["forecast_cons_12m"] = np.log10(train["forecast_cons_12m"]+1)
train["forecast_cons_year"] = np.log10(train["forecast_cons_year"]+1)
train["forecast_meter_rent_12m"] = np.log10(train["forecast_meter_rent_12m"]+1)
train["imp_cons"] = np.log10(train["imp_cons"]+1)
```

Now let's see how the distribution looks like.


```python
fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot histograms
sns.distplot((train["cons_12m"].dropna()), ax=axs[0])
sns.distplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.distplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.distplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
sns.distplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.distplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.distplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_92_1.png)



```python
fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot boxplots
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_93_1.png)



```python
train.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_cons_12m</th>
      <th>forecast_cons_year</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>pow_max</th>
      <th>churn</th>
      <th>tenure</th>
      <th>months_activ</th>
      <th>months_to_end</th>
      <th>months_modif_prod</th>
      <th>months_renewal</th>
      <th>channel_epu</th>
      <th>channel_ewp</th>
      <th>channel_fix</th>
      <th>channel_foo</th>
      <th>channel_lmk</th>
      <th>channel_sdd</th>
      <th>channel_usi</th>
      <th>origin_ewx</th>
      <th>origin_kam</th>
      <th>origin_ldk</th>
      <th>origin_lxi</th>
      <th>origin_usa</th>
      <th>activity_apd</th>
      <th>activity_ckf</th>
      <th>activity_clu</th>
      <th>activity_cwo</th>
      <th>activity_fmw</th>
      <th>activity_kkk</th>
      <th>activity_kwu</th>
      <th>activity_sfi</th>
      <th>activity_wxe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16069.000000</td>
      <td>16090.000000</td>
      <td>16050.000000</td>
      <td>16055.000000</td>
      <td>16071.000000</td>
      <td>15970.000000</td>
      <td>16092.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>15970.000000</td>
      <td>16096.000000</td>
      <td>16069.000000</td>
      <td>16083.000000</td>
      <td>16083.000000</td>
      <td>16096.000000</td>
      <td>16081.000000</td>
      <td>16096.000000</td>
      <td>16093.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
      <td>16096.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.283812</td>
      <td>0.800300</td>
      <td>2.359281</td>
      <td>3.006826</td>
      <td>1.869956</td>
      <td>0.991547</td>
      <td>1.549610</td>
      <td>0.135901</td>
      <td>0.052951</td>
      <td>43.533496</td>
      <td>0.184145</td>
      <td>1.305021</td>
      <td>22.462276</td>
      <td>21.460318</td>
      <td>1.347788</td>
      <td>217.987028</td>
      <td>5.030629</td>
      <td>20.604131</td>
      <td>0.099093</td>
      <td>5.329958</td>
      <td>58.929858</td>
      <td>6.376615</td>
      <td>35.741240</td>
      <td>4.924640</td>
      <td>0.000249</td>
      <td>0.060015</td>
      <td>0.000124</td>
      <td>0.458313</td>
      <td>0.128790</td>
      <td>0.000746</td>
      <td>0.089712</td>
      <td>0.000062</td>
      <td>0.280629</td>
      <td>0.227634</td>
      <td>0.486146</td>
      <td>0.000124</td>
      <td>0.097975</td>
      <td>0.011742</td>
      <td>0.007393</td>
      <td>0.007580</td>
      <td>0.013606</td>
      <td>0.026218</td>
      <td>0.014289</td>
      <td>0.005157</td>
      <td>0.007393</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.915265</td>
      <td>1.748833</td>
      <td>1.789067</td>
      <td>0.709778</td>
      <td>1.612963</td>
      <td>5.160969</td>
      <td>0.589394</td>
      <td>0.026252</td>
      <td>0.048617</td>
      <td>5.212252</td>
      <td>0.387615</td>
      <td>1.165532</td>
      <td>23.700883</td>
      <td>27.917349</td>
      <td>1.459808</td>
      <td>366.742030</td>
      <td>1.676101</td>
      <td>21.772421</td>
      <td>0.298796</td>
      <td>1.749248</td>
      <td>20.125024</td>
      <td>3.633479</td>
      <td>30.609746</td>
      <td>3.812127</td>
      <td>0.015763</td>
      <td>0.237522</td>
      <td>0.011147</td>
      <td>0.498275</td>
      <td>0.334978</td>
      <td>0.027295</td>
      <td>0.285777</td>
      <td>0.007882</td>
      <td>0.449320</td>
      <td>0.419318</td>
      <td>0.499824</td>
      <td>0.011147</td>
      <td>0.297290</td>
      <td>0.107726</td>
      <td>0.085668</td>
      <td>0.086733</td>
      <td>0.115852</td>
      <td>0.159787</td>
      <td>0.118684</td>
      <td>0.071626</td>
      <td>0.085668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.122184</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-525.540000</td>
      <td>-615.660000</td>
      <td>1.000000</td>
      <td>-4148.990000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>-112.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.773786</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.713952</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.236285</td>
      <td>0.115237</td>
      <td>0.000000</td>
      <td>40.606701</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.960000</td>
      <td>11.950000</td>
      <td>1.000000</td>
      <td>51.970000</td>
      <td>4.000000</td>
      <td>12.500000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>44.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.187408</td>
      <td>0.000000</td>
      <td>2.959041</td>
      <td>3.073579</td>
      <td>2.583199</td>
      <td>0.000000</td>
      <td>1.310481</td>
      <td>0.142881</td>
      <td>0.086163</td>
      <td>44.311378</td>
      <td>0.000000</td>
      <td>1.662380</td>
      <td>21.090000</td>
      <td>20.970000</td>
      <td>1.000000</td>
      <td>119.680000</td>
      <td>5.000000</td>
      <td>13.856000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>57.000000</td>
      <td>6.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.701508</td>
      <td>0.000000</td>
      <td>3.617000</td>
      <td>3.430950</td>
      <td>3.301030</td>
      <td>0.000000</td>
      <td>2.122126</td>
      <td>0.146348</td>
      <td>0.098837</td>
      <td>44.311378</td>
      <td>0.000000</td>
      <td>2.341118</td>
      <td>29.640000</td>
      <td>29.640000</td>
      <td>1.000000</td>
      <td>275.810000</td>
      <td>6.000000</td>
      <td>19.800000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>71.000000</td>
      <td>9.000000</td>
      <td>64.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.206748</td>
      <td>6.622052</td>
      <td>6.656933</td>
      <td>5.016210</td>
      <td>5.243970</td>
      <td>50.000000</td>
      <td>3.382502</td>
      <td>0.273963</td>
      <td>0.195975</td>
      <td>59.444710</td>
      <td>1.000000</td>
      <td>4.177357</td>
      <td>374.640000</td>
      <td>374.640000</td>
      <td>32.000000</td>
      <td>24570.650000</td>
      <td>16.000000</td>
      <td>500.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>185.000000</td>
      <td>17.000000</td>
      <td>185.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now the distribution looks more like normal distribution and the standard deviation has changed.

We notice that there are still some values that are quite far from the range. We will deal with these outliers later.

## 3. Calculating the correlation of the variables**


```python
correlation = features.corr()
```


```python
# Plot correlation
plt.figure(figsize=(19,15))
sns.heatmap(correlation, xticklabels=correlation.columns.values,
yticklabels=correlation.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_99_0.png)


Multicolliearity happens when one predictor variable in a multiple regression model can be linearly predicted from the others with a high degree of accuracy. This can lead to skewed or misleading results. Luckily, decision trees and boosted trees algorithms are immune to multicollinearity by nature. When they decide to split, the tree will choose only one of the perfectly correlated features. However, other algorithms like Logistic Regression or Linear Regression are not immune to that problem and should be fixed before training the model.


```python
# Calculate correlation of variables
correlation = train.corr()
```


```python
# Plot correlation
plt.figure(figsize=(20,18))
sns.heatmap(correlation, xticklabels=correlation.columns.values,
yticklabels=correlation.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_102_0.png)


We can see that num_years_antig has a high correlation with months_activ (it provides us the same information). We can remove variables with very high correlation.


```python
train.drop(columns=["num_years_antig", "forecast_cons_year"],inplace=True)
```

## 4. Removing outliers

The most common way to identify an outlier is:

1. Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile.

Or

2. Data point that falls outside of 3 standard deviations.


As we identified during the exploratory phase, the consumption data has several outliers. We are going to remove those outliers.

There are several ways to handle with those outliers such as removing them (this works well for massive datasets) or replacing them with sensible data (works better when the dataset is not that big).


We will replace the outliers with the mean (average of the values excluding outliers).


```python
def replace_outliers_z_score(dataframe, column, Z=3):
    """
    Replace outliers with the mean values using the Z score.
    Nan values are also replaced with the mean values.
    """

    from scipy.stats import zscore
    df = dataframe.copy(deep=True)
    df.dropna(inplace=True, subset=[column])

    # Calculate mean without outliers
    df["zscore"] = zscore(df[column])
    mean_ = df[(df["zscore"] > -Z) & (df["zscore"] < Z)][column].mean()

    # Replace with mean values
    dataframe[column] = dataframe[column].fillna(mean_)
    dataframe["zscore"] = zscore(dataframe[column])
    no_outliers = dataframe[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z)].shape[0]
    dataframe.loc[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z),column] = mean_
    
    # Print message
    print("Replaced:", no_outliers, " outliers in ", column)
    return dataframe.drop(columns="zscore")
```


```python
for c in features.columns:
    if c != "id":
        features = replace_outliers_z_score(features,c)
```

    Replaced: 276  outliers in  mean_year_price_p1_var
    Replaced: 0  outliers in  mean_year_price_p2_var
    Replaced: 0  outliers in  mean_year_price_p3_var
    Replaced: 120  outliers in  mean_year_price_p1_fix
    Replaced: 0  outliers in  mean_year_price_p2_fix
    Replaced: 0  outliers in  mean_year_price_p3_fix
    Replaced: 122  outliers in  mean_year_price_p1
    Replaced: 0  outliers in  mean_year_price_p2
    Replaced: 0  outliers in  mean_year_price_p3
    


```python
features.reset_index(drop=True, inplace=True)
```

As we identified during the exploratory phase, and when carrying out the log transformation , the dataset has several outliers.

We will also replace the outliers with the mean (average of the values excluding outliers).


```python
def _find_outliers_iqr(dataframe, column):
    # Find outliers using the 1.5*IQR rule.

    col = sorted(dataframe[column])
    q1, q3= np.percentile(col,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    results = {"iqr": iqr, "lower_bound":lower_bound, "upper_bound":upper_bound}
    return results
```


```python
def remove_outliers_iqr(dataframe, column):
    # Remove outliers using the 1.5*IQR rule.
    outliers = _find_outliers_iqr(dataframe, column)
    removed = dataframe[(dataframe[column] < outliers["lower_bound"]) |
                          (dataframe[column] > outliers["upper_bound"])].shape
    dataframe = dataframe[(dataframe[column] > outliers["lower_bound"]) &
                          (dataframe[column] < outliers["upper_bound"])]
    print("Removed:", removed[0], " outliers")
    return dataframe
```


```python
def remove_outliers_z_score(dataframe, column, Z=3):
    # Remove outliers using the Z score. Values with more than 3 are removed.
    from scipy.stats import zscore
    dataframe["zscore"] = zscore(dataframe[column])
    removed = dataframe[(dataframe["zscore"] < -Z) |
                        (dataframe["zscore"] > Z)].shape
    dataframe = dataframe[(dataframe["zscore"] > -Z) &
                          (dataframe["zscore"] < Z)]
    print("Removed:", removed[0], " outliers of ", column)
    return dataframe.drop(columns="zscore")
```


```python
def replace_outliers_z_score(dataframe, column, Z=3):
    # Replace outliers with the mean values using the Z score.
    # Nan values are also replaced with the mean values.

    from scipy.stats import zscore
    df = dataframe.copy(deep=True)
    df.dropna(inplace=True, subset=[column])
    
    # Calculate mean without outliers
    df["zscore"] = zscore(df[column])
    mean_ = df[(df["zscore"] > -Z) & (df["zscore"] < Z)][column].mean()

    # Replace with mean values
    no_outliers = dataframe[column].isnull().sum()
    dataframe[column] = dataframe[column].fillna(mean_)
    dataframe["zscore"] = zscore(dataframe[column])
    dataframe.loc[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z),column] = mean_
    
    # Print message
    print("Replaced:", no_outliers, " outliers in ", column)
    return dataframe.drop(columns="zscore")
```


```python
train = replace_outliers_z_score(train,"cons_12m")
train = replace_outliers_z_score(train,"cons_gas_12m")
train = replace_outliers_z_score(train,"cons_last_month")
train = replace_outliers_z_score(train,"forecast_cons_12m")

#train = replace_outliers_z_score(train,"forecast_cons_year")
train = replace_outliers_z_score(train,"forecast_discount_energy")
train = replace_outliers_z_score(train,"forecast_meter_rent_12m")
train = replace_outliers_z_score(train,"forecast_price_energy_p1")
train = replace_outliers_z_score(train,"forecast_price_energy_p2")
train = replace_outliers_z_score(train,"forecast_price_pow_p1")
train = replace_outliers_z_score(train,"imp_cons")
train = replace_outliers_z_score(train,"margin_gross_pow_ele")
train = replace_outliers_z_score(train,"margin_net_pow_ele")
train = replace_outliers_z_score(train,"net_margin")
train = replace_outliers_z_score(train,"pow_max")
train = replace_outliers_z_score(train,"months_activ")
train = replace_outliers_z_score(train,"months_to_end")
train = replace_outliers_z_score(train,"months_modif_prod")
train = replace_outliers_z_score(train,"months_renewal")
```

    Replaced: 27  outliers in  cons_12m
    Replaced: 6  outliers in  cons_gas_12m
    Replaced: 46  outliers in  cons_last_month
    Replaced: 41  outliers in  forecast_cons_12m
    Replaced: 126  outliers in  forecast_discount_energy
    Replaced: 4  outliers in  forecast_meter_rent_12m
    Replaced: 126  outliers in  forecast_price_energy_p1
    Replaced: 126  outliers in  forecast_price_energy_p2
    Replaced: 126  outliers in  forecast_price_pow_p1
    Replaced: 27  outliers in  imp_cons
    Replaced: 13  outliers in  margin_gross_pow_ele
    Replaced: 13  outliers in  margin_net_pow_ele
    Replaced: 15  outliers in  net_margin
    Replaced: 3  outliers in  pow_max
    Replaced: 0  outliers in  months_activ
    Replaced: 0  outliers in  months_to_end
    Replaced: 0  outliers in  months_modif_prod
    Replaced: 0  outliers in  months_renewal
    


```python
train.reset_index(drop=True, inplace=True)
```

Now, let's see how the boxplots changed!


```python
fig, axs = plt.subplots(nrows=7, figsize=(18,50))

# Plot boxplots
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])

#sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](https://github.com/waldysetio/customer-churn/blob/main/images/feature-engineering/output_125_1.png)


## 6. Saving data to csv

Let's check the dataframes before saving data.


```python
train
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>forecast_cons_12m</th>
      <th>forecast_discount_energy</th>
      <th>forecast_meter_rent_12m</th>
      <th>forecast_price_energy_p1</th>
      <th>forecast_price_energy_p2</th>
      <th>forecast_price_pow_p1</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>pow_max</th>
      <th>churn</th>
      <th>tenure</th>
      <th>months_activ</th>
      <th>months_to_end</th>
      <th>months_modif_prod</th>
      <th>months_renewal</th>
      <th>channel_epu</th>
      <th>channel_ewp</th>
      <th>channel_fix</th>
      <th>channel_foo</th>
      <th>channel_lmk</th>
      <th>channel_sdd</th>
      <th>channel_usi</th>
      <th>origin_ewx</th>
      <th>origin_kam</th>
      <th>origin_ldk</th>
      <th>origin_lxi</th>
      <th>origin_usa</th>
      <th>activity_apd</th>
      <th>activity_ckf</th>
      <th>activity_clu</th>
      <th>activity_cwo</th>
      <th>activity_fmw</th>
      <th>activity_kkk</th>
      <th>activity_kwu</th>
      <th>activity_sfi</th>
      <th>activity_wxe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48ada52261e7cf58715202705a0451c9</td>
      <td>5.490346</td>
      <td>0.000000</td>
      <td>4.001128</td>
      <td>4.423595</td>
      <td>0.0</td>
      <td>2.556652</td>
      <td>0.095919</td>
      <td>0.088347</td>
      <td>58.995952</td>
      <td>0</td>
      <td>2.920541</td>
      <td>-41.76</td>
      <td>-41.76</td>
      <td>1</td>
      <td>198.346424</td>
      <td>18.402912</td>
      <td>0</td>
      <td>3</td>
      <td>37.0</td>
      <td>10.0</td>
      <td>37.0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>4.327104</td>
      <td>4.739944</td>
      <td>0.000000</td>
      <td>3.085953</td>
      <td>0.0</td>
      <td>0.444045</td>
      <td>0.114481</td>
      <td>0.098142</td>
      <td>40.606701</td>
      <td>1</td>
      <td>0.000000</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.990000</td>
      <td>43.648000</td>
      <td>1</td>
      <td>3</td>
      <td>30.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>6.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>3.668479</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.280920</td>
      <td>0.0</td>
      <td>1.237292</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.000000</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.890000</td>
      <td>13.800000</td>
      <td>0</td>
      <td>7</td>
      <td>76.0</td>
      <td>7.0</td>
      <td>76.0</td>
      <td>4.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>2.736397</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.689841</td>
      <td>0.0</td>
      <td>1.599009</td>
      <td>0.165794</td>
      <td>0.087899</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.000000</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.600000</td>
      <td>13.856000</td>
      <td>0</td>
      <td>6</td>
      <td>68.0</td>
      <td>3.0</td>
      <td>68.0</td>
      <td>8.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bba03439a292a1e166f80264c16191cb</td>
      <td>3.200029</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.382089</td>
      <td>0.0</td>
      <td>1.318689</td>
      <td>0.146694</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.000000</td>
      <td>30.22</td>
      <td>30.22</td>
      <td>1</td>
      <td>25.460000</td>
      <td>13.200000</td>
      <td>0</td>
      <td>6</td>
      <td>69.0</td>
      <td>2.0</td>
      <td>69.0</td>
      <td>9.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>18463073fb097fc0ac5d3e040f356987</td>
      <td>4.508812</td>
      <td>4.680707</td>
      <td>0.000000</td>
      <td>3.667360</td>
      <td>0.0</td>
      <td>1.291591</td>
      <td>0.138305</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>1</td>
      <td>0.000000</td>
      <td>27.88</td>
      <td>27.88</td>
      <td>2</td>
      <td>381.770000</td>
      <td>15.000000</td>
      <td>0</td>
      <td>3</td>
      <td>43.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.792645</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>d0a6f71671571ed83b2645d23af6de00</td>
      <td>3.858778</td>
      <td>0.000000</td>
      <td>2.260071</td>
      <td>2.801191</td>
      <td>0.0</td>
      <td>2.161458</td>
      <td>0.100167</td>
      <td>0.091892</td>
      <td>58.995952</td>
      <td>0</td>
      <td>1.228913</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>90.340000</td>
      <td>6.000000</td>
      <td>1</td>
      <td>4</td>
      <td>40.0</td>
      <td>7.0</td>
      <td>40.0</td>
      <td>4.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>10e6828ddd62cbcf687cb74928c4c2d2</td>
      <td>3.265996</td>
      <td>0.000000</td>
      <td>2.255273</td>
      <td>2.281919</td>
      <td>0.0</td>
      <td>2.115943</td>
      <td>0.116900</td>
      <td>0.100015</td>
      <td>40.606701</td>
      <td>0</td>
      <td>1.279895</td>
      <td>39.84</td>
      <td>39.84</td>
      <td>1</td>
      <td>20.380000</td>
      <td>15.935000</td>
      <td>1</td>
      <td>3</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>46.0</td>
      <td>10.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>1cf20fd6206d7678d5bcafd28c53b4db</td>
      <td>2.120574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.308351</td>
      <td>0.0</td>
      <td>0.912753</td>
      <td>0.145711</td>
      <td>0.000000</td>
      <td>44.311378</td>
      <td>0</td>
      <td>0.000000</td>
      <td>13.08</td>
      <td>13.08</td>
      <td>1</td>
      <td>0.960000</td>
      <td>11.000000</td>
      <td>0</td>
      <td>4</td>
      <td>40.0</td>
      <td>7.0</td>
      <td>40.0</td>
      <td>4.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>563dde550fd624d7352f3de77c0cdfcd</td>
      <td>3.941064</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.882758</td>
      <td>0.0</td>
      <td>0.315970</td>
      <td>0.167086</td>
      <td>0.088454</td>
      <td>45.311378</td>
      <td>0</td>
      <td>0.000000</td>
      <td>11.84</td>
      <td>11.84</td>
      <td>1</td>
      <td>96.340000</td>
      <td>10.392000</td>
      <td>0</td>
      <td>6</td>
      <td>72.0</td>
      <td>11.0</td>
      <td>72.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 44 columns</p>
</div>




```python
history
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_date</th>
      <th>price_p1_var</th>
      <th>price_p2_var</th>
      <th>price_p3_var</th>
      <th>price_p1_fix</th>
      <th>price_p2_fix</th>
      <th>price_p3_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-01-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-02-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-03-01</td>
      <td>0.151367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-04-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-05-01</td>
      <td>0.149626</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266931</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192997</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-08-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192998</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-09-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>192999</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-10-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193000</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-11-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
    <tr>
      <th>193001</th>
      <td>16f51cdc2baa19af0b940ee1b3dd17d5</td>
      <td>2015-12-01</td>
      <td>0.119916</td>
      <td>0.102232</td>
      <td>0.076257</td>
      <td>40.728885</td>
      <td>24.43733</td>
      <td>16.291555</td>
    </tr>
  </tbody>
</table>
<p>193002 rows ?? 8 columns</p>
</div>




```python
features
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>mean_year_price_p1_var</th>
      <th>mean_year_price_p2_var</th>
      <th>mean_year_price_p3_var</th>
      <th>mean_year_price_p1_fix</th>
      <th>mean_year_price_p2_fix</th>
      <th>mean_year_price_p3_fix</th>
      <th>mean_year_price_p1</th>
      <th>mean_year_price_p2</th>
      <th>mean_year_price_p3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002203ffbb812588b632b9e628cc38d</td>
      <td>0.124338</td>
      <td>0.103794</td>
      <td>0.073160</td>
      <td>40.701732</td>
      <td>24.421038</td>
      <td>16.280694</td>
      <td>40.826071</td>
      <td>24.524832</td>
      <td>16.353854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0004351ebdd665e6ee664792efc4fd13</td>
      <td>0.146426</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.385450</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.531877</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0010bcc39e42b3c2131ed2ce55246e3c</td>
      <td>0.181558</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.319710</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.501268</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0010ee3855fdea87602a5b7aba8e42de</td>
      <td>0.118757</td>
      <td>0.098292</td>
      <td>0.069032</td>
      <td>40.647427</td>
      <td>24.388455</td>
      <td>16.258971</td>
      <td>40.766185</td>
      <td>24.486748</td>
      <td>16.328003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00114d74e963e47177db89bc70108537</td>
      <td>0.147926</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.266930</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.414856</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16091</th>
      <td>ffef185810e44254c3a4c6395e6b4d8a</td>
      <td>0.138863</td>
      <td>0.115125</td>
      <td>0.080780</td>
      <td>40.896427</td>
      <td>24.637456</td>
      <td>16.507972</td>
      <td>41.035291</td>
      <td>24.752581</td>
      <td>16.588752</td>
    </tr>
    <tr>
      <th>16092</th>
      <td>fffac626da707b1b5ab11e8431a4d0a2</td>
      <td>0.147137</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.311375</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>44.458512</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16093</th>
      <td>fffc0cacd305dd51f316424bbb08d1bd</td>
      <td>0.153879</td>
      <td>0.129497</td>
      <td>0.094842</td>
      <td>41.160171</td>
      <td>24.895768</td>
      <td>16.763569</td>
      <td>41.314049</td>
      <td>25.025265</td>
      <td>16.858411</td>
    </tr>
    <tr>
      <th>16094</th>
      <td>fffe4f5646aa39c7f97f95ae2679ce64</td>
      <td>0.123858</td>
      <td>0.103499</td>
      <td>0.073735</td>
      <td>40.606699</td>
      <td>24.364017</td>
      <td>16.242678</td>
      <td>40.730558</td>
      <td>24.467516</td>
      <td>16.316414</td>
    </tr>
    <tr>
      <th>16095</th>
      <td>ffff7fa066f1fb305ae285bb03bf325a</td>
      <td>0.125360</td>
      <td>0.104895</td>
      <td>0.075635</td>
      <td>40.647427</td>
      <td>24.388455</td>
      <td>16.258971</td>
      <td>40.772788</td>
      <td>24.493350</td>
      <td>16.334606</td>
    </tr>
  </tbody>
</table>
<p>16096 rows ?? 10 columns</p>
</div>



Now, save them to csv files.


```python
train.to_csv(r'processed_train.csv', index = False, header=True)
```


```python
history.to_csv(r'processed_history.csv', index = False, header=True)
```


```python
history.to_csv(r'features.csv', index = False, header=True)
```

# Modeling and Evaluation

## 1. Importing packages


```python
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
```


```python
%matplotlib inline
```


```python
sns.set(color_codes=True)
```

## 2. Loading data


```python
train_data = pd.read_csv('https://raw.githubusercontent.com/waldysetio/customer-churn-analysis/main/processed-data/processed_train.csv')
features = pd.read_csv('https://raw.githubusercontent.com/waldysetio/customer-churn-analysis/main/processed-data/features.csv')
```

Let's merge both dataframes.


```python
train = pd.merge(train_data, features, on="id")
```


```python
pd.DataFrame({"Dataframe columns": train.columns})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataframe columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cons_12m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cons_gas_12m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cons_last_month</td>
    </tr>
    <tr>
      <th>4</th>
      <td>forecast_cons_12m</td>
    </tr>
    <tr>
      <th>5</th>
      <td>forecast_discount_energy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>forecast_meter_rent_12m</td>
    </tr>
    <tr>
      <th>7</th>
      <td>forecast_price_energy_p1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>forecast_price_energy_p2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>forecast_price_pow_p1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>has_gas</td>
    </tr>
    <tr>
      <th>11</th>
      <td>imp_cons</td>
    </tr>
    <tr>
      <th>12</th>
      <td>margin_gross_pow_ele</td>
    </tr>
    <tr>
      <th>13</th>
      <td>margin_net_pow_ele</td>
    </tr>
    <tr>
      <th>14</th>
      <td>nb_prod_act</td>
    </tr>
    <tr>
      <th>15</th>
      <td>net_margin</td>
    </tr>
    <tr>
      <th>16</th>
      <td>pow_max</td>
    </tr>
    <tr>
      <th>17</th>
      <td>churn</td>
    </tr>
    <tr>
      <th>18</th>
      <td>tenure</td>
    </tr>
    <tr>
      <th>19</th>
      <td>months_activ</td>
    </tr>
    <tr>
      <th>20</th>
      <td>months_to_end</td>
    </tr>
    <tr>
      <th>21</th>
      <td>months_modif_prod</td>
    </tr>
    <tr>
      <th>22</th>
      <td>months_renewal</td>
    </tr>
    <tr>
      <th>23</th>
      <td>channel_epu</td>
    </tr>
    <tr>
      <th>24</th>
      <td>channel_ewp</td>
    </tr>
    <tr>
      <th>25</th>
      <td>channel_fix</td>
    </tr>
    <tr>
      <th>26</th>
      <td>channel_foo</td>
    </tr>
    <tr>
      <th>27</th>
      <td>channel_lmk</td>
    </tr>
    <tr>
      <th>28</th>
      <td>channel_sdd</td>
    </tr>
    <tr>
      <th>29</th>
      <td>channel_usi</td>
    </tr>
    <tr>
      <th>30</th>
      <td>origin_ewx</td>
    </tr>
    <tr>
      <th>31</th>
      <td>origin_kam</td>
    </tr>
    <tr>
      <th>32</th>
      <td>origin_ldk</td>
    </tr>
    <tr>
      <th>33</th>
      <td>origin_lxi</td>
    </tr>
    <tr>
      <th>34</th>
      <td>origin_usa</td>
    </tr>
    <tr>
      <th>35</th>
      <td>activity_apd</td>
    </tr>
    <tr>
      <th>36</th>
      <td>activity_ckf</td>
    </tr>
    <tr>
      <th>37</th>
      <td>activity_clu</td>
    </tr>
    <tr>
      <th>38</th>
      <td>activity_cwo</td>
    </tr>
    <tr>
      <th>39</th>
      <td>activity_fmw</td>
    </tr>
    <tr>
      <th>40</th>
      <td>activity_kkk</td>
    </tr>
    <tr>
      <th>41</th>
      <td>activity_kwu</td>
    </tr>
    <tr>
      <th>42</th>
      <td>activity_sfi</td>
    </tr>
    <tr>
      <th>43</th>
      <td>activity_wxe</td>
    </tr>
    <tr>
      <th>44</th>
      <td>mean_year_price_p1_var</td>
    </tr>
    <tr>
      <th>45</th>
      <td>mean_year_price_p2_var</td>
    </tr>
    <tr>
      <th>46</th>
      <td>mean_year_price_p3_var</td>
    </tr>
    <tr>
      <th>47</th>
      <td>mean_year_price_p1_fix</td>
    </tr>
    <tr>
      <th>48</th>
      <td>mean_year_price_p2_fix</td>
    </tr>
    <tr>
      <th>49</th>
      <td>mean_year_price_p3_fix</td>
    </tr>
    <tr>
      <th>50</th>
      <td>mean_year_price_p1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>mean_year_price_p2</td>
    </tr>
    <tr>
      <th>52</th>
      <td>mean_year_price_p3</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Splitting data

We will use "churn" as the output or response and the rest columns as the features of our model.


```python
y = train["churn"]
X = train.drop(labels = ["id","churn"], axis = 1)
```

Next we will split the data into training and validation data. The percentages of each test can be changed but a 75%-25% is a good ratio.
We also use a random state generator in order to split it randomly.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)
```

## 4. Modelling


```python
model = xgb.XGBClassifier(learning_rate=0.1,max_depth=6,n_estimators=500,n_jobs=-1)
result = model.fit(X_train,y_train)
```

## 5. Model evaluation

Let's evaluate our model using the evaluation metrics of:

Accuracy : The most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.

Precision : The ratio of correctly predicted positive observations to the total predicted positive observations.

Recall (Sensitivity): The ratio of correctly predicted positive observations to the all observations in actual class.

**Accuracy, Precision, Recall**

This is the confusion matrix that shows the predicted values compared to the true values.


```python
from sklearn.metrics import plot_confusion_matrix

class_names = ['0', '1']
disp = plot_confusion_matrix(result, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 values_format = '.0f')
plt.grid(False)  
plt.show(disp)


```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_23_0.png)



```python
def evaluate(model_, X_test_, y_test_): 

    # Get the model predictions
    prediction_test_ = model_.predict(X_test_)

    # Print the evaluation metrics as pandas dataframe 
    results = pd.DataFrame({"Accuracy" : [metrics.accuracy_score(y_test_, prediction_test_)], 
                            "Precision" : [metrics.precision_score(y_test_, prediction_test_)], 
                            "Recall" : [metrics.recall_score(y_test_, prediction_test_)]})
    return results
```


```python
evaluate(model, X_test, y_test)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.907306</td>
      <td>0.769231</td>
      <td>0.144578</td>
    </tr>
  </tbody>
</table>
</div>



**ROC-AUC**

Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity
and specificity.


```python
def calculate_roc_auc(model_, X_test_, y_test_):
    """
    Evaluate the roc-auc score
    """
    # Get the model predictions
    # We are using the prediction for the class 1 -> churn
    prediction_test_ = model_.predict_proba(X_test_)[:,1] 
    
    # Compute roc-auc
    fpr, tpr, thresholds = metrics.roc_curve(y_test_, prediction_test_)

    # Print the evaluation metrics as pandas dataframe
    score = pd.DataFrame({"ROC-AUC" : [metrics.auc(fpr, tpr)]}) 
   
    return fpr, tpr, score
```


```python
def plot_roc_auc(fpr,tpr): 
    """
    Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates.
    """
    # Initialize plot
    f, ax = plt.subplots(figsize=(14,8)) # Plot ROC
    
    # Plot ROC
    roc_auc = metrics.auc(fpr, tpr) 
    ax.plot(fpr, tpr, lw=2, alpha=0.3, label="AUC = %0.2f" % (roc_auc)) 

    # Plot the random line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r', label="Random", alpha=.8)
  
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05]) 
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel("False Positive Rate (FPR)") 
    ax.set_ylabel("True Positive Rate (TPR)") 
    ax.set_title("ROC-AUC") 
    ax.legend(loc="lower right")
    plt.show()
```


```python
fpr, tpr, auc_score = calculate_roc_auc(model, X_test, y_test)
```


```python
auc_score
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ROC-AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.684637</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_roc_auc(fpr, tpr)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_32_0.png)


**Stratified K-fold validation**


```python
def plot_roc_curve(fprs, tprs): 
    """ 
    Plot the Receiver Operating Characteristic from a list of true positive rates and false positive rates. 
    """ 
    # Initialize useful lists + the plot axes. 
    tprs_interp = [] 
    aucs = [] 
    mean_fpr = np.linspace(0, 1, 100) 
    f, ax = plt.subplots(figsize=(18,10))
    # Plot ROC for each K-Fold + compute AUC scores. 
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)): 
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr)) 
        tprs_interp[-1][0] = 0.0 
        roc_auc = metrics.auc(fpr, tpr) 
        aucs.append(roc_auc) 
        ax.plot(fpr, tpr, lw=2, alpha=0.3, 
        label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc)) 

    # Plot the luck line. 
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r',
              label="Random", alpha=.8) 
    
    # Plot the mean ROC. 
    mean_tpr = np.mean(tprs_interp, axis=0) 
    mean_tpr[-1] = 1.0 
    mean_auc = metrics.auc(mean_fpr, mean_tpr) 
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', 
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), 
            lw=4, alpha=.8) 
    
    # Plot the standard deviation around the mean ROC. 
    std_tpr = np.std(tprs_interp, axis=0) 
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1) 
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0) 
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,
                    label=r"$\pm$ 1 std. dev.")
    
    # Fine tune and show the plot. 
    ax.set_xlim([-0.05, 1.05]) 
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel("False Positive Rate (FPR)") 
    ax.set_ylabel("True Positive Rate (TPR)") 
    ax.set_title("ROC-AUC") 
    ax.legend(loc="lower right") 
    plt.show() 
    return (f, ax)
def compute_roc_auc(model_, index): 
    y_predict = model_.predict_proba(X.iloc[index])[:,1] 
    fpr, tpr, thresholds = metrics.roc_curve(y.iloc[index], y_predict) 
    auc_score = metrics.auc(fpr, tpr) 
    return fpr, tpr, auc_score
```


```python
cv = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)
fprs, tprs, scores = [], [], []
```


```python
for (train, test), i in zip(cv.split(X, y), range(5)): 
    model.fit(X.iloc[train], y.iloc[train]) 
    _, _, auc_score_train = compute_roc_auc(model, train) 
    fpr, tpr, auc_score = compute_roc_auc(model, test) 
    scores.append((auc_score_train, auc_score)) 
    fprs.append(fpr) 
    tprs.append(tpr)
```


```python
plot_roc_curve(fprs, tprs)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_37_0.png)


## 6. Model finetuning

**Random search cross validation**


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
# Create the random grid
params = {
        'min_child_weight': [i for i in np.arange(1,15,1)], 
        'gamma': [i for i in np.arange(0,6,0.5)], 
        'subsample': [i for i in np.arange(0,1.1,0.1)], 
        'colsample_bytree': [i for i in np.arange(0,1.1,0.1)], 
        'max_depth': [i for i in np.arange(1,15,1)], 
        'scale_pos_weight':[i for i in np.arange(1,15,1)], 
        'learning_rate': [i for i in np.arange(0,0.15,0.01)], 
        'n_estimators' : [i for i in np.arange(0,2000,100)]}

```

We will create a new base mode.


```python
xg = xgb.XGBClassifier(objective='binary:logistic', 
                       silent=True, nthread=1)
```


```python
# Random search of parameters, using 5
xg_random = RandomizedSearchCV(xg, param_distributions=params, 
                               n_iter=1, scoring= "roc_auc", 
                               n_jobs=4, cv=5, verbose=3, random_state=1001)
# Fit the random search model
xg_random.fit(X_train, y_train)

```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   2 out of   5 | elapsed:    6.8s remaining:   10.2s
    [Parallel(n_jobs=4)]: Done   5 out of   5 | elapsed:    8.3s finished
    




    RandomizedSearchCV(cv=5, error_score=nan,
                       estimator=XGBClassifier(base_score=0.5, booster='gbtree',
                                               colsample_bylevel=1,
                                               colsample_bynode=1,
                                               colsample_bytree=1, gamma=0,
                                               learning_rate=0.1, max_delta_step=0,
                                               max_depth=3, min_child_weight=1,
                                               missing=None, n_estimators=100,
                                               n_jobs=1, nthread=1,
                                               objective='binary:logistic',
                                               random_state=0, reg_alpha=0,
                                               reg_lambda=1, scale...
                                            'n_estimators': [0, 100, 200, 300, 400,
                                                             500, 600, 700, 800,
                                                             900, 1000, 1100, 1200,
                                                             1300, 1400, 1500, 1600,
                                                             1700, 1800, 1900],
                                            'scale_pos_weight': [1, 2, 3, 4, 5, 6,
                                                                 7, 8, 9, 10, 11,
                                                                 12, 13, 14],
                                            'subsample': [0.0, 0.1, 0.2,
                                                          0.30000000000000004, 0.4,
                                                          0.5, 0.6000000000000001,
                                                          0.7000000000000001, 0.8,
                                                          0.9, 1.0]},
                       pre_dispatch='2*n_jobs', random_state=1001, refit=True,
                       return_train_score=False, scoring='roc_auc', verbose=3)




```python
best_random = xg_random.best_params_
best_random = {'subsample': 0.8,
  'scale_pos_weight': 1, 
  'n_estimators': 1100, 
  'min_child_weight': 1, 
  'max_depth': 12, 
  'learning_rate': 0.01, 
  'gamma': 4.0, 
  'colsample_bytree': 0.60}

```


```python
# Create a model with the parameters found
model_random = xgb.XGBClassifier(objective='binary:logistic', 
                        silent=True, nthread=1, **best_random)
fprs, tprs, scores = [], [], []
```


```python
for (train, test), i in zip(cv.split(X, y), range(5)): 
    model_random.fit(X.iloc[train], y.iloc[train]) 
    _, _, auc_score_train = compute_roc_auc(model_random, train) 
    fpr, tpr, auc_score = compute_roc_auc(model_random, test) 
    scores.append((auc_score_train, auc_score)) 
    fprs.append(fpr) 
    tprs.append(tpr)

```


```python
plot_roc_curve(fprs,tprs)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_48_0.png)


**Grid search with cross validation (calculating over weekend, then make smaller)**


```python
from sklearn.model_selection import GridSearchCV
```


```python
# Create the parameter grid based on the results of random search 
param_grid = {'subsample': [0.7],
              'scale_pos_weight': [1], 
              'n_estimators': [1100], 
              'min_child_weight': [1], 
              'max_depth': [12, 13, 14], 
              'learning_rate': [0.005, 0.01], 
              'gamma': [4.0], 
              'colsample_bytree': [0.6]}

```


```python
# Create model
xg = xgb.XGBClassifier(objective='binary:logistic', 
                       silent=True, nthread=1)
```


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, 
                            cv = 5, n_jobs = -1, verbose = 2, scoring = "roc_auc")
```


```python
# Fit the grid search to the data
grid_search.fit(X_train,y_train)
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 23.2min finished
    




    GridSearchCV(cv=5, error_score=nan,
                 estimator=XGBClassifier(base_score=0.5, booster='gbtree',
                                         colsample_bylevel=1, colsample_bynode=1,
                                         colsample_bytree=1, gamma=0,
                                         learning_rate=0.1, max_delta_step=0,
                                         max_depth=3, min_child_weight=1,
                                         missing=None, n_estimators=100, n_jobs=1,
                                         nthread=1, objective='binary:logistic',
                                         random_state=0, reg_alpha=0, reg_lambda=1,
                                         scale_pos_w...ed=None, silent=True,
                                         subsample=1, verbosity=1),
                 iid='deprecated', n_jobs=-1,
                 param_grid={'colsample_bytree': [0.6], 'gamma': [4.0],
                             'learning_rate': [0.005, 0.01],
                             'max_depth': [12, 13, 14], 'min_child_weight': [1],
                             'n_estimators': [1100], 'scale_pos_weight': [1],
                             'subsample': [0.7]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=2)




```python
best_grid = grid_search.best_params_
best_grid
```




    {'colsample_bytree': 0.6,
     'gamma': 4.0,
     'learning_rate': 0.005,
     'max_depth': 12,
     'min_child_weight': 1,
     'n_estimators': 1100,
     'scale_pos_weight': 1,
     'subsample': 0.7}




```python
# Create a model with the parameters found
model_grid = xgb.XGBClassifier(objective='binary:logistic', 
                               silent=True, nthread=1, **best_grid)
fprs, tprs, scores = [], [], []

```


```python
for (train, test), i in zip(cv.split(X, y), range(5)):
    model_grid.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(model_grid, train)
    fpr, tpr, auc_score = compute_roc_auc(model_grid, test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
```


```python
plot_roc_curve(fprs, tprs)
plt.show()
```


![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_58_0.png)


## 7. Understanding the model


**Feature importance**

One simple way of boserving the feature importance is through counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear.


```python
fig, ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(model_grid, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8c0ad7f210>




![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_62_1.png)


In the feature importance graph above we can see that cons_12m and some other are the features that appear the most in our model and we could infere that these two features have a significant importnace in our model.

**Partial dependence plot**


```python
from sklearn.inspection import plot_partial_dependence
```


```python
# Create a model with the parameters found
model_grid_v2 = xgb.XGBClassifier(objective='binary:logistic',
                                  silent=True, nthread=1, **best_grid)
model_grid_v2.fit(X_train.values,y_train.values)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.6, gamma=4.0,
                  learning_rate=0.005, max_delta_step=0, max_depth=12,
                  min_child_weight=1, missing=None, n_estimators=1100, n_jobs=1,
                  nthread=1, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=True, subsample=0.7, verbosity=1)




```python
fig = plt.figure(figsize=(15,15))
plot_partial_dependence(model_grid_v2, X_test.values, features=[16, 49], 
                        feature_names=X_test.columns.tolist(), fig=fig)
```

    /usr/local/lib/python3.7/dist-packages/sklearn/inspection/_partial_dependence.py:715: FutureWarning: The fig parameter is deprecated in version 0.22 and will be removed in version 0.24
      FutureWarning)
    




    <sklearn.inspection._partial_dependence.PartialDependenceDisplay at 0x7f8c0aeb9890>




![png](https://github.com/waldysetio/customer-churn/blob/main/images/model-evaluation/output_67_2.png)


tenure

The overall trend is unchaged as compared to our previous models. We can see the trend spikes at slighly different times of the tenure ( 6y ) but then it goes down again and bottoms around 10 years. Then, it starts recovering a bit.

mean_year_price_p2

In our previous models, we saw a sort of "stairshape", in this case we see the pdp is almost flat with some spikes on the extreme values, which hints us that the variable mean_year_price_p2 is not very relevant in this model.
