# Customer Churn Prediction
Author: Waldy Setiono (waldysetiono@gmail.com)

**Introduction**: An energy company that has been serving corporate, SME, and residential customers is currently undergoing a significant churn mostly in its SME segment due to recent change in regulation that liberalizes the energy market. This project aims to analyze customer history and make predictive model of churn propensity to help the company deal with this issue. 

**Data**: The data used in this project are obtained from Boston Consulting Group (BCG GAMMA).

# Exploratory Data Analysis

## 1. Importing Packages


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

## 2. Loading Data

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
<p>16096 rows × 32 columns</p>
</div>




```python
history_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>193002 rows × 8 columns</p>
</div>




```python
churn_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>16096 rows × 2 columns</p>
</div>



**Merging train data and churn data**


```python
train = pd.merge(train_data, churn_data, on="id")
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>16096 rows × 33 columns</p>
</div>



## 3. Statistics

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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>16096 rows × 33 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>193002 rows × 8 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



## 4. Data Visualization

**Churn Rate**


```python
churn = train[['id','churn']]
churn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>16096 rows × 2 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


## 5. Data Cleaning

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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



## 6. Formatting data

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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>1359 rows × 8 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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

## 7. Saving data to csv


```python
train.to_csv(r'clean_train_data.csv', index = False, header=True)
```


```python
history_data.to_csv(r'clean_history_data.csv', index = False, header=True)
```
