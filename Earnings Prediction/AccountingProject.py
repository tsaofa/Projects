# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:00:57 2020

@author: tsaof
"""


import pandas as pd
import wrds
conn = wrds.Connection(wrds_username='tsaofa')
import numpy as np
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
import pickle


# extract funda data
def get_funda(start_year, end_year):
    ''' Given a time period, obtain data from WRDS databale (funda),
    return the analytical data table for building earnings prediction model using EPS as the target variable'''

    funda = conn.raw_sql("""
                          select distinct gvkey, datadate, fyear, fyr, oiadp, at, act, che, lct, dlc, txp, dp, exchg, sich, BKVLPS, CSHO, epspx, dvc, ceq, ivao , lt, dltt, ivst, pstk
                          from compa.funda where
                          (consol='C' and indfmt='INDL' and datafmt='STD' and popsrc='D') and
                          fyear <= %d and fyear >= %d and (exchg=11 or exchg=12)"""% (end_year, start_year))
    company = conn.raw_sql("""
                            select gvkey, sic
                            from compa.company
                            """)

    # Merge tables, standardize sic code, and drop duplicates or missing values
    data=pd.merge(funda,company,on=['gvkey'])
    data['sic1']=np.where(data['sich']>0,data['sich'],data['sic'])
    data=data.drop(['sich','sic'],axis=1)
    data['sic1']=data['sic1'].astype(int)
    data = data.dropna()
    data = data.drop_duplicates(['gvkey','fyear'],keep='last') 
        
    ## Get lag variable
    lag1=data.copy()
    lag1['fyear']=lag1['fyear']+1
    lag1=lag1.rename(columns={'at':'at_lag1','act':'act_lag1','che':'che_lag1','lct':'lct_lag1','dlc':'dlc_lag1','txp':'txp_lag1',
                                                  'ivao':'ivao_lag1','lt':'lt_lag1','dltt':'dltt_lag1','ivst':'ivst_lag1','pstk':'pstk_lag1'})
    lag1['WC_lag1'] = (lag1['act_lag1']-lag1['che_lag1'])-(lag1['lct_lag1']-lag1['dlc_lag1'])
    lag1['NCO_lag1'] = (lag1['at_lag1']-lag1['act_lag1']-lag1['ivao_lag1'])-(lag1['lt_lag1']-lag1['lct_lag1']-lag1['dltt_lag1'])
    lag1['FIN_lag1'] = (lag1['ivst_lag1']+lag1['ivao_lag1'])-(lag1['dltt_lag1']+lag1['dlc_lag1']+lag1['pstk_lag1'])
    lag1=lag1[['gvkey','fyear','at_lag1','act_lag1','che_lag1','lct_lag1', 'dlc_lag1','txp_lag1','WC_lag1','NCO_lag1','FIN_lag1']]
    data=pd.merge(data,lag1,on=['gvkey','fyear'])
    
    ## Get lead variable
    lead1=data.copy()
    lead1['fyear']=lead1['fyear']-1
    lead1=lead1.rename(columns={'epspx':'epspx_lead1'})
    lead1=lead1[['gvkey','fyear','epspx_lead1']]
    data=pd.merge(data,lead1,on=['gvkey','fyear'])
    
    ## construct multiple variables for predictive modeling
    data['b']=(data['ceq'])/(data['csho'])
    data.loc[data['epspx'] <= 0, 'NegEPS'] = 1 
    data.loc[data['epspx'] > 0, 'NegEPS'] = 0
    data['EPS*NegEPS'] = data['epspx']*data['NegEPS']
    data['WC'] = (data['act']-data['che'])-(data['lct']-data['dlc'])
    data['NCO'] = (data['at']-data['act']-data['ivao'])-(data['lt']-data['lct']-data['dltt'])
    data['FIN'] = (data['ivst']+data['ivao'])-(data['dltt']+data['dlc']+data['pstk'])
    data['tacc'] = ((data['WC']-data['WC_lag1'])+(data['NCO']-data['NCO_lag1'])+(data['FIN']-data['FIN_lag1']))/data['csho']
       
    ## construct additional features
    data['nc_assets'] = ((data['act']-data['act_lag1'])-(data['che']-data['che_lag1']))/data['csho']
    data['nc_liab'] = ((data['lct']-data['lct_lag1'])-(data['dlc']-data['dlc_lag1'])-(data['txp']-data['txp_lag1']))/data['csho']
    data = data.rename(columns={'dvc':'paid_dividend'})
    data['deprec'] = data['dp']/data['csho']
    data['paid_dividend'] = data['paid_dividend']/data['csho']
    data['total_asset'] = data['at']/data['csho']
    data['total_accrual'] = data['nc_assets']-data['nc_liab']-data['deprec']
    data.loc[data['paid_dividend'] > 0, 'Y_Div']=1
    data.loc[data['paid_dividend']<= 0, 'Y_Div']=0
    
    ## Generate industry code
    data['industry']=data['sic1'].astype(str).str[:2].astype(int)
    
    ## final data view
    data=data[['gvkey','fyear','fyr','datadate','epspx_lead1','epspx','NegEPS','EPS*NegEPS',
                             'b','tacc','nc_assets', 'nc_liab', 'deprec','paid_dividend','Y_Div', 'industry',
                             'total_asset','total_accrual']]
    data=data.dropna() 
    data['fyear']=data['fyear'].astype(int)
    #fundaclean1=pd.merge(fundaclean1,security,on=['gvkey'])
    data = data.drop_duplicates(keep='last') 
    
    return data

train_data = get_funda(2005, 2016)
test_data = get_funda(2013, 2017)
recomm_data = get_funda(2015, 2019)

## Random Walk Model
def RW_regression(data):
    
    ''' The random walk model predicts the earning for the next time period based on the earning of the current time period.
    This function trains a random walk model given a dataset containing the necessary predictor variables'''
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx']]

    for var in ['epspx_lead1', 'epspx']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['epspx']])
    result = sm.OLS(Y,X).fit()
    
    print(result.summary()) 
    
    return result
    

RW_result = RW_regression(train_data)


## Earnings Persistence model
def EP_regression(data):
    
    ''' The earnings persistance model predicts the earning for the next time period based on more extended earnings data in the current time period.
    This function trains a earnings persistance model given a dataset containing the necessary predictor variables'''
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'NegEPS','epspx','EPS*NegEPS']]

    for var in ['epspx_lead1', 'NegEPS','epspx','EPS*NegEPS']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['NegEPS','epspx','EPS*NegEPS']])
    result = sm.OLS(Y,X).fit()
    
    print(result.summary()) 
    
    return result
 

EP_result = EP_regression(train_data)

## Residual Income model
def RI_regression(data):
    
    ''' The residual income model predicts the earning for the next time period based on price, earning, and book values in the current time period.
    This function trains a residual income model given a dataset containing the necessary predictor variables'''
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'NegEPS','epspx','EPS*NegEPS','b','tacc']]

    for var in ['epspx_lead1', 'NegEPS','epspx','EPS*NegEPS','b','tacc']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['NegEPS','epspx','EPS*NegEPS','b','tacc']])
    result = sm.OLS(Y,X).fit()
    
    print(result.summary()) 
    
    return result
 
RI_result =RI_regression(train_data)


## Fit GDP
def GDP_regression(data, table_type = 1, save = False):
    
    ''' In an effort to improve the EP model, GDP forecasting data was added (source: OECD) to investigate the effect of GDP on earnings prediction.
    This function trains a GDP model given a dataset containing the necessary predictor variables.
    Table_type = 1 trains the model across all industries and prints out pooled result.
    Table_type = 2 trains the model within industries.
    save = False (default) does not save the model as .sav files'''
    
    metric = pd.read_csv('GDP.csv')
    metric = metric.loc[metric['LOCATION']=='WLD']
    metric = metric[['TIME', 'Value']]
    metric = metric.rename(columns={'TIME':'Year'})
    metric['Year'] = metric['Year']-1
    
    Table = data.merge(metric, how = 'left', left_on = ['fyear'], right_on = ['Year'])
    
    Table = Table[['gvkey', 'fyear', 'epspx_lead1', 'epspx', 'NegEPS',
           'EPS*NegEPS', 'Value','industry']]
    
    for var in ['epspx_lead1', 'epspx', 'NegEPS','EPS*NegEPS']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
    
    if table_type == 1:
        Y = Table['epspx_lead1']
        X = sm.add_constant(Table[['epspx', 'NegEPS','EPS*NegEPS', 'Value']])
        result = sm.OLS(Y,X).fit()      
        
        print(result.summary()) 
        return result
    
    elif table_type == 2:
        Et = []
        NegEt = []
        Et_NegEt = []
        wrd_GDP = []
        intercept = []
        adj_r2 = []
        num_samples = []
        industry_list = []
        for industry in Table['industry'].unique():
            regress_data = Table[Table['industry']==industry]
            if regress_data.shape[0] >= 3:
                industry_list.append(industry)
                Y = regress_data['epspx_lead1']        
                X = sm.add_constant(regress_data[['epspx', 'NegEPS','EPS*NegEPS', 'Value']])
                result = sm.OLS(Y, X).fit()
                intercept.append((result.params[0],result.tvalues[0]))
                Et.append((result.params[1],result.tvalues[1]))
                NegEt.append((result.params[2],result.tvalues[2]))
                Et_NegEt.append((result.params[3],result.tvalues[3]))
                wrd_GDP.append((result.params[4],result.tvalues[4]))
                adj_r2.append(result.rsquared_adj)
                num_samples.append(regress_data.shape[0])
                #Save model
                if save == True:
                    filename = 'GDPmodel_%s' %industry
                    pickle.dump(result,open(filename,'wb'))
                else:
                    continue
            else:
                print("Note: Industry number %d contains less than 3 entries and is skipped" %industry)
            
        result_table = pd.DataFrame(
                {'industry': industry_list,
                 'n': num_samples,
                 'intercept': [entry[0] for entry in intercept],
                 'intercept_t': [entry[1] for entry in intercept],
                 'Et': [entry[0] for entry in Et],
                 'Et_t': [entry[1] for entry in Et],
                 'NegEt': [entry[0] for entry in NegEt],
                 'NegEt_t': [entry[1] for entry in NegEt],
                 'Et_NegEt': [entry[0] for entry in Et_NegEt],
                 'Et_NegEt_t': [entry[1] for entry in Et_NegEt],
                 'wrd_GDP': [entry[0] for entry in wrd_GDP],
                 'wrd_GDP_t': [entry[1] for entry in wrd_GDP],
                 'adj_r2': adj_r2
                 })
        result_table = result_table.sort_values(by = ['industry'])
        
        print(ind_GDP.describe().transpose())
        return result_table
    
pooled_GDP = GDP_regression(train_data)

ind_GDP = GDP_regression(train_data,table_type = 2)


## HVZ regression 
def HVZ_regression(data, table_type = 1, save = False):
    
    ''' In an effort to improve the EP model, an model similar to the HVZ model was developed (source: Hou, K., van Dijk, M.A., Zhang, Y., 2012. The implied cost of capital: A new approach. Journal
    of Accounting & Economics, 53, 504–526.) The only difference from the HVZ models is that all variables are scaled by outstanding shares.
    This function trains an HVZ model given a dataset containing the necessary predictor variables.
    Table_type = 1 trains the model across all industries and prints out pooled result.
    Table_type = 2 trains the model within industries.
    save = False (default) does not save the model as .sav files'''
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual','industry']]

    for var in ['epspx_lead1', 'epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
    if table_type ==1:        
        Y = Table['epspx_lead1']
        X = sm.add_constant(Table[['epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual']])
        result = sm.OLS(Y,X).fit()
        
        print(result.summary()) 
        return result
    
    elif table_type == 2:
        earning = []
        total_asset = []
        paid_dividend = []
        Y_Div = []
        NegEPS = []
        total_accrual = []
        intercept = []
        adj_r2 = []
        num_samples = []
        industry_list = []
        for industry in Table['industry'].unique():
            regress_data = Table[Table['industry']==industry]
            if regress_data.shape[0] >= 3:
                industry_list.append(industry)         
                Y = regress_data['epspx_lead1']        
                X = sm.add_constant(regress_data[['epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual']])
                result = sm.OLS(Y, X).fit()
                intercept.append((result.params[0],result.tvalues[0]))
                earning.append((result.params[1],result.tvalues[1]))
                total_asset.append((result.params[2],result.tvalues[2]))
                paid_dividend.append((result.params[3],result.tvalues[3]))
                Y_Div.append((result.params[4],result.tvalues[4]))
                NegEPS.append((result.params[5],result.tvalues[5]))
                total_accrual.append((result.params[6],result.tvalues[6]))
                adj_r2.append(result.rsquared_adj)
                num_samples.append(regress_data.shape[0])
                #Save model
                if save == True:                    
                    filename = 'HVZmodel_%s' %industry
                    pickle.dump(result,open(filename,'wb'))
                else:
                    continue
            else:
                print("Note: Industry number %d contains less than 3 entries and is skipped" %industry)
                
        result_table = pd.DataFrame(
                {'industry': industry_list,
                 'n': num_samples,
                 'intercept': [entry[0] for entry in intercept],
                 'intercept_t': [entry[1] for entry in intercept],
                 'earning': [entry[0] for entry in earning],
                 'earning_t': [entry[1] for entry in earning],
                 'total_asset': [entry[0] for entry in total_asset],
                 'total_asset_t': [entry[1] for entry in total_asset],
                 'paid_dividend': [entry[0] for entry in paid_dividend],
                 'paid_dividend_t': [entry[1] for entry in paid_dividend],
                 'Y_Div': [entry[0] for entry in Y_Div],
                 'Y_Div_t': [entry[1] for entry in Y_Div],
                 'NegEPS': [entry[0] for entry in NegEPS],
                 'NegEPS_t': [entry[1] for entry in NegEPS],
                 'total_accrual': [entry[0] for entry in total_accrual],
                 'total_accrual_t': [entry[1] for entry in total_accrual],
                 'adj_r2': adj_r2
                 })
        result_table = result_table.sort_values(by = ['industry'])
        
        print(result_table.describe().transpose())
        return result_table

pooled_HVZ = HVZ_regression(train_data)

ind_HVZ = HVZ_regression(train_data,table_type = 2)

 
##Earnings volatility model
def EV_regression(data, table_type = 1, save = False):
    
    ''' In an effort to improve the EP model, a new model was developed to take into account earnings volatility.
    Earnings volatility is defined as the standard deviation of the earnings in the past 3 years.
    This function trains an earnings volatility model given a dataset containing the necessary predictor variables.
    Table_type = 1 trains the model across all industries and prints out pooled result.
    Table_type = 2 trains the model within industries.
    save = False (default) does not save the model as .sav files'''
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx', 'NegEPS','EPS*NegEPS','industry']]
      
    temp1 = Table[['gvkey','fyear','epspx']].copy()
    temp1['fyear'] = temp1['fyear']+1
    temp1 = temp1.rename(columns={'epspx':'earning_lag1'})
    Table = Table.merge(temp1, how = 'left', on=['gvkey','fyear'])
    temp2 = Table[['gvkey','fyear','epspx']].copy()
    temp2['fyear'] = temp2['fyear']+2
    temp2 = temp2.rename(columns={'epspx':'earning_lag2'})
    Table = Table.merge(temp2, how = 'left', on=['gvkey','fyear'])
    Table['earn_volat'] = Table[['earning_lag1', 'earning_lag2', 'epspx']].std(axis = 1)
    Table = Table.dropna()
    Table['earn_volat'] = Table['earn_volat']
    
    for var in ['epspx_lead1', 'epspx', 'NegEPS', 'EPS*NegEPS','earn_volat']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))

    if table_type ==1:        
        Y = Table['epspx_lead1']
        X = sm.add_constant(Table[['epspx', 'NegEPS', 'EPS*NegEPS', 'earn_volat']])
        result = sm.OLS(Y,X).fit()
        
        print(result.summary())
        return result
    
    elif table_type == 2:
        Et = []
        NegEt = []
        Et_NegEt = []
        earn_volat = []
        intercept = []
        adj_r2 = []
        num_samples = []
        industry_list = []
        for industry in Table['industry'].unique():
            regress_data = Table[Table['industry']==industry]
            if regress_data.shape[0] >= 3:
                industry_list.append(industry)
                Y = regress_data['epspx_lead1']        
                X = sm.add_constant(regress_data[['epspx', 'NegEPS','EPS*NegEPS', 'earn_volat']])
                result = sm.OLS(Y, X).fit()
                intercept.append((result.params[0],result.tvalues[0]))
                Et.append((result.params[1],result.tvalues[1]))
                NegEt.append((result.params[2],result.tvalues[2]))
                Et_NegEt.append((result.params[3],result.tvalues[3]))
                earn_volat.append((result.params[4],result.tvalues[4]))
                adj_r2.append(result.rsquared_adj)
                num_samples.append(regress_data.shape[0])
                #Save model
                if save == True:
                    filename = 'EVmodel_%s' %industry
                    pickle.dump(result,open(filename,'wb'))
                else:
                    continue
            else:
                print("Note: Industry number %d contains less than 3 entries and is skipped" %industry)
           
          
        result_table = pd.DataFrame(
                {'industry': industry_list,
                 'n': num_samples,
                 'intercept': [entry[0] for entry in intercept],
                 'intercept_t': [entry[1] for entry in intercept],
                 'Et': [entry[0] for entry in Et],
                 'Et_t': [entry[1] for entry in Et],
                 'NegEt': [entry[0] for entry in NegEt],
                 'NegEt_t': [entry[1] for entry in NegEt],
                 'Et_NegEt': [entry[0] for entry in Et_NegEt],
                 'Et_NegEt_t': [entry[1] for entry in Et_NegEt],
                 'earn_volat': [entry[0] for entry in earn_volat],
                 'earn_volat_t': [entry[1] for entry in earn_volat],
                 'adj_r2': adj_r2
                 })
        result_table = result_table.sort_values(by = ['industry'])
        
        print(result_table.describe().transpose())
        return result_table      

pooled_EV = EV_regression(train_data)

ind_EV = EV_regression(train_data,table_type = 2, save = True)

### Earnings volatility model was the best-fitted out of all the ones investigated, and thus was used to make final predictions. 

## Predict future earnings using trained earnings volatility models (industry-specific)
def EV_predict(data):    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx', 'NegEPS','EPS*NegEPS','industry']]      
    temp1 = Table[['gvkey','fyear','epspx']].copy()
    temp1['fyear'] = temp1['fyear']+1
    temp1 = temp1.rename(columns={'epspx':'earning_lag1'})
    Table = Table.merge(temp1, how = 'left', on=['gvkey','fyear'])
    temp2 = Table[['gvkey','fyear','epspx']].copy()
    temp2['fyear'] = temp2['fyear']+2
    temp2 = temp2.rename(columns={'epspx':'earning_lag2'})
    Table = Table.merge(temp2, how = 'left', on=['gvkey','fyear'])
    Table['earn_volat'] = Table[['earning_lag1', 'earning_lag2', 'epspx']].std(axis = 1)
    Table = Table.dropna()
    
    for var in ['epspx_lead1', 'epspx', 'NegEPS', 'EPS*NegEPS','earn_volat']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
    Table['model'] = ['EVmodel_%s' %ind_num for ind_num in Table['industry']]
        
    projection_2017 = []
        
    for index, row in Table.iterrows():
        model = pickle.load(open(row['model'],'rb'))
        X_test = pd.DataFrame(row[['epspx','NegEPS','EPS*NegEPS', 'earn_volat']]).transpose()
        X_test = sm.add_constant(X_test,has_constant='add')
        pred_value = model.predict(X_test)  
        projection_2017.append(pred_value.values[0])
        
    Table['projection_2017'] = projection_2017
        
    Table = Table[['gvkey', 'industry', 'fyear', 'epspx_lead1','projection_2017']]
        
    return Table
       
prediction_2017 = EV_predict(test_data)
        
## Recommend Future stocks based on industry-specific earnings volatility model
def EV_recommend(data):    
    Table = data[['gvkey', 'fyear', 'epspx', 'NegEPS','EPS*NegEPS','industry']]      
    temp1 = Table[['gvkey','fyear','epspx']].copy()
    temp1['fyear'] = temp1['fyear']+1
    temp1 = temp1.rename(columns={'epspx':'earning_lag1'})
    Table = Table.merge(temp1, how = 'left', on=['gvkey','fyear'])
    temp2 = Table[['gvkey','fyear','epspx']].copy()
    temp2['fyear'] = temp2['fyear']+2
    temp2 = temp2.rename(columns={'epspx':'earning_lag2'})
    Table = Table.merge(temp2, how = 'left', on=['gvkey','fyear'])
    Table['earn_volat'] = Table[['earning_lag1', 'earning_lag2', 'epspx']].std(axis = 1)
    Table = Table.dropna()
    
    for var in ['epspx', 'NegEPS', 'EPS*NegEPS','earn_volat']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
    Table['model'] = ['EVmodel_%s' %ind_num for ind_num in Table['industry']]
        
    projection_2019 = []
        
    for index, row in Table.iterrows():
        model = pickle.load(open(row['model'],'rb'))
        X_test = pd.DataFrame(row[['epspx','NegEPS','EPS*NegEPS', 'earn_volat']]).transpose()
        X_test = sm.add_constant(X_test,has_constant='add')
        pred_value = model.predict(X_test)  
        projection_2019.append(pred_value.values[0])
        
    Table['projection_2019'] = projection_2019
    Table['change_eps'] = Table['projection_2019'] - Table['epspx']
        
    Table = Table[['gvkey', 'industry', 'fyear', 'epspx', 'projection_2019','change_eps']]
    Table = Table.sort_values(by = 'change_eps', ascending  = False)
        
    return Table

recommendation = EV_recommend(recomm_data)


