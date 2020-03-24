# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:00:57 2020

@author: tsaof
"""


import pandas as pd
import wrds
conn = wrds.Connection(wrds_username='tsaofa')
import numpy as np
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets, linear_model
import pickle


# extract funda data
def get_funda(start_year, end_year):

    funda = conn.raw_sql("""
                          select distinct gvkey, datadate, fyear, fyr, oiadp, at, act, che, lct, dlc, txp, dp, exchg, sich, BKVLPS, CSHO, epspx, dvc, ceq, ivao , lt, dltt, ivst, pstk
                          from compa.funda where
                          (consol='C' and indfmt='INDL' and datafmt='STD' and popsrc='D') and
                          fyear <= %d and fyear >= %d and (exchg=11 or exchg=12)"""% (end_year, start_year))
    company = conn.raw_sql("""
                            select gvkey, sic
                            from compa.company
                            """)
    
    security = conn.raw_sql("""
                            select distinct gvkey, ibtic
                            from compa.security
                            """)

    #clean funda data
    fundaclean=pd.merge(funda,company,on=['gvkey'])
    fundaclean['sic1']=np.where(fundaclean['sich']>0,fundaclean['sich'],fundaclean['sic'])
    fundaclean=fundaclean.drop(['sich','sic'],axis=1)
    fundaclean['sic1']=fundaclean['sic1'].astype(int)
    fundaclean = fundaclean.dropna()
    fundaclean = fundaclean.drop_duplicates(['gvkey','fyear'],keep='last') 
        
    ## lag variable
    fundacleanlag1=fundaclean.copy()
    fundacleanlag1['fyear']=fundacleanlag1['fyear']+1
    fundacleanlag1=fundacleanlag1.rename(columns={'at':'at_lag1','act':'act_lag1','che':'che_lag1','lct':'lct_lag1','dlc':'dlc_lag1','txp':'txp_lag1',
                                                  'ivao':'ivao_lag1','lt':'lt_lag1','dltt':'dltt_lag1','ivst':'ivst_lag1','pstk':'pstk_lag1'})
    fundacleanlag1['WC_lag1'] = (fundacleanlag1['act_lag1']-fundacleanlag1['che_lag1'])-(fundacleanlag1['lct_lag1']-fundacleanlag1['dlc_lag1'])
    fundacleanlag1['NCO_lag1'] = (fundacleanlag1['at_lag1']-fundacleanlag1['act_lag1']-fundacleanlag1['ivao_lag1'])-(fundacleanlag1['lt_lag1']-fundacleanlag1['lct_lag1']-fundacleanlag1['dltt_lag1'])
    fundacleanlag1['FIN_lag1'] = (fundacleanlag1['ivst_lag1']+fundacleanlag1['ivao_lag1'])-(fundacleanlag1['dltt_lag1']+fundacleanlag1['dlc_lag1']+fundacleanlag1['pstk_lag1'])
    fundacleanlag1=fundacleanlag1[['gvkey','fyear','at_lag1','act_lag1','che_lag1','lct_lag1', 'dlc_lag1','txp_lag1','WC_lag1','NCO_lag1','FIN_lag1']]
    fundaclean1=pd.merge(fundaclean,fundacleanlag1,on=['gvkey','fyear'])

    
    ## lead variable
    fundacleanlead1=fundaclean.copy()
    fundacleanlead1['fyear']=fundacleanlead1['fyear']-1
    fundacleanlead1=fundacleanlead1.rename(columns={'epspx':'epspx_lead1'})
    fundacleanlead1=fundacleanlead1[['gvkey','fyear','epspx_lead1']]
    fundaclean1=pd.merge(fundaclean1,fundacleanlead1,on=['gvkey','fyear'])
    
    ## construct variables (base models)
    fundaclean1['b']=(fundaclean1['ceq'])/(fundaclean1['csho'])
    fundaclean1.loc[fundaclean1['epspx'] <= 0, 'NegEPS'] = 1 
    fundaclean1.loc[fundaclean1['epspx'] > 0, 'NegEPS'] = 0
    fundaclean1['EPS*NegEPS'] = fundaclean1['epspx']*fundaclean1['NegEPS']
    fundaclean1['WC'] = (fundaclean1['act']-fundaclean1['che'])-(fundaclean1['lct']-fundaclean1['dlc'])
    fundaclean1['NCO'] = (fundaclean1['at']-fundaclean1['act']-fundaclean1['ivao'])-(fundaclean1['lt']-fundaclean1['lct']-fundaclean1['dltt'])
    fundaclean1['FIN'] = (fundaclean1['ivst']+fundaclean1['ivao'])-(fundaclean1['dltt']+fundaclean1['dlc']+fundaclean1['pstk'])
    fundaclean1['tacc'] = ((fundaclean1['WC']-fundaclean1['WC_lag1'])+(fundaclean1['NCO']-fundaclean1['NCO_lag1'])+(fundaclean1['FIN']-fundaclean1['FIN_lag1']))/fundaclean1['csho']
       
    ## construct additional features
    fundaclean1['nc_assets'] = ((fundaclean1['act']-fundaclean1['act_lag1'])-(fundaclean1['che']-fundaclean1['che_lag1']))/fundaclean1['csho']
    fundaclean1['nc_liab'] = ((fundaclean1['lct']-fundaclean1['lct_lag1'])-(fundaclean1['dlc']-fundaclean1['dlc_lag1'])-(fundaclean1['txp']-fundaclean1['txp_lag1']))/fundaclean1['csho']
    fundaclean1 = fundaclean1.rename(columns={'dvc':'paid_dividend'})
    fundaclean1['deprec'] = fundaclean1['dp']/fundaclean1['csho']
    fundaclean1['paid_dividend'] = fundaclean1['paid_dividend']/fundaclean1['csho']
    fundaclean1['total_asset'] = fundaclean1['at']/fundaclean1['csho']
    fundaclean1['total_accrual'] = fundaclean1['nc_assets']-fundaclean1['nc_liab']-fundaclean1['deprec']
    fundaclean1.loc[fundaclean1['paid_dividend'] > 0, 'Y_Div']=1
    fundaclean1.loc[fundaclean1['paid_dividend']<= 0, 'Y_Div']=0
    
    #industry code
    #fundaclean1['industry']=fundaclean1['sic1'].floordiv(other=100)
    fundaclean1['industry']=fundaclean1['sic1'].astype(str).str[:2].astype(int)
    
    #final funda data
    fundaclean1=fundaclean1[['gvkey','fyear','fyr','datadate','epspx_lead1','epspx','NegEPS','EPS*NegEPS',
                             'b','tacc','nc_assets', 'nc_liab', 'deprec','paid_dividend','Y_Div', 'industry',
                             'total_asset','total_accrual']]
    fundaclean1=fundaclean1.dropna() 
    fundaclean1['fyear']=fundaclean1['fyear'].astype(int)
    #fundaclean1=pd.merge(fundaclean1,security,on=['gvkey'])
    fundaclean1 = fundaclean1.drop_duplicates(keep='last') 
    
    return fundaclean1

train_data = get_funda(2005, 2016)
test_data = get_funda(2013, 2017)
recomm_data = get_funda(2015, 2019)

## Random Walk Model
def RW_regression(data):
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx']]

    for var in ['epspx_lead1', 'epspx']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['epspx']])
    result = sm.OLS(Y,X).fit()
    return result
 

RW_result = RW_regression(train_data)
RW_result.summary()

## EP model
def EP_regression(data):
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'NegEPS','epspx','EPS*NegEPS']]

    for var in ['epspx_lead1', 'NegEPS','epspx','EPS*NegEPS']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['NegEPS','epspx','EPS*NegEPS']])
    result = sm.OLS(Y,X).fit()
    return result
 

EP_result = EP_regression(train_data)
EP_result.summary()


## IR model
def IR_regression(data):
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'NegEPS','epspx','EPS*NegEPS','b','tacc']]

    for var in ['epspx_lead1', 'NegEPS','epspx','EPS*NegEPS','b','tacc']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
      
    Y = Table['epspx_lead1']
    X = sm.add_constant(Table[['NegEPS','epspx','EPS*NegEPS','b','tacc']])
    result = sm.OLS(Y,X).fit()
    return result
 

IR_result =IR_regression(train_data)
IR_result.summary()

## Fit GDP
def GDP_regression(data, table_type = 1, save = False):
    
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
        
        return result_table
    
pooled_GDP = GDP_regression(train_data)
pooled_GDP.summary()

ind_GDP = GDP_regression(train_data,table_type = 2)
ind_GDP_summ = ind_GDP.describe().transpose()
 
ind_GDP.to_csv('ind_GDP.csv')


## HVZ regression 
def HVZ_regression(data, table_type = 1, save = False):
    
    Table = data[['gvkey', 'fyear', 'epspx_lead1', 'epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual','industry']]

    for var in ['epspx_lead1', 'epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual']:    
        Table[var]=np.where(Table[var].isnull(), np.nan, winsorize(Table[var], limits=(0.01,0.01)))
        
    if table_type ==1:        
        Y = Table['epspx_lead1']
        X = sm.add_constant(Table[['epspx', 'total_asset', 'paid_dividend','Y_Div','NegEPS','total_accrual']])
        result = sm.OLS(Y,X).fit()
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
        
        return result_table

pooled_HVZ = HVZ_regression(train_data)
pooled_HVZ.summary()

ind_HVZ = HVZ_regression(train_data,table_type = 2)
ind_HVZ_summ = ind_HVZ.describe().transpose()
 
ind_HVZ.to_csv('ind_HVZ.csv')
 
##Earnings volatility model
def EV_regression(data, table_type = 1, save = False):
    
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
                
        return result_table      

pooled_EV = EV_regression(train_data)
pooled_EV.summary()

ind_EV = EV_regression(train_data,table_type = 2, save = True)
ind_EV_summ = ind_EV.describe().transpose()

## Market share regression
#CRSP Data
ccm = conn.raw_sql("""select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' 
                  or linkprim='P')""") 
crsp = conn.raw_sql("""select a.permno, a.permco, a.date,
                    a.prc, b.shrcd, b.exchcd
	                  from crsp.msf as a 
                    left join crsp.msenames as b
	                  on a.permno = b.permno
	                  and b.namedt <= a.date
	                  and a.date <= b.nameendt
	                  where b.exchcd between 1 and 2
                    and b.shrcd between 10 and 11
                    and a.date between '2006-01-01'
                    and '2017-12-31'
                    """)
#Linktable Dates
ccm['linkdt']=pd.to_datetime(ccm['linkdt'])
ccm['linkenddt']=pd.to_datetime(ccm['linkenddt'])
ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

#Merge Compustat and Linktable
ccm1=pd.merge(fundaclean1[['gvkey','datadate','fyear','NegEPS', 'bkvlps','bkvlps_lag1', 'epspx_lead1', "epspx", "tacc",'industry']],ccm,how='left',on=['gvkey'])
ccm1['datadate']=pd.to_datetime(ccm1['datadate'])


#Set dates for CRPS Merge
crsp['date']=pd.to_datetime(crsp['date'])
crsp['jdate']=crsp['date']+pd.offsets.MonthEnd(0) 

#Merge with CRSP
ccm2=ccm1[(ccm1['datadate']>=ccm1['linkdt'])&(ccm1['datadate']<=ccm1['linkenddt'])]
ccm2=ccm2[['gvkey','fyear','datadate','epspx_lead1','epspx','NegEPS','bkvlps','bkvlps_lag1','tacc','industry', 'permno']]
crspcomp=pd.merge(ccm2, crsp,how='left', on=['permno'])
crspcomp=crspcomp[(crspcomp['jdate']==crspcomp['datadate'])]
crspcomp = crspcomp.drop_duplicates(['gvkey','fyear'],keep='last')

#Winsorize
for var in ['epspx_lead1','epspx','bkvlps','bkvlps_lag1','tacc']:    
  crspcomp[var]=np.where(crspcomp[var].isnull(), np.nan, winsorize(crspcomp[var], limits=(0.01,0.01)))
crspcomp.describe()

#Lag Share Price
crspcomplag1=crspcomp.copy()
crspcomplag1['fyear']=crspcomplag1['fyear']+1
crspcomplag1=crspcomplag1.rename(columns={'prc':'prc_lag1'})
crspcomplag1=crspcomplag1[['gvkey','fyear','permno', 'prc_lag1']]
crspcomp=pd.merge(crspcomp,crspcomplag1,on=['gvkey','fyear','permno'])
crspcomp=crspcomp.dropna() 

#Train/Test Split
train = crspcomp[(crspcomp['fyear']<=2015)]
test = crspcomp[(crspcomp['fyear']>=2016)]

#Seperate target
y_train = train['epspx_lead1']
X_train = train.drop(['epspx_lead1','gvkey','fyear','datadate','industry','permno','permco','date','shrcd','exchcd','jdate'], axis=1)
y_test = test['epspx_lead1']
X_test = test.drop(['epspx_lead1','gvkey','fyear','datadate','industry','permno','permco','date','shrcd','exchcd','jdate'], axis=1)

#Create regression formula to paste
all_columns = "+".join(X_train)
pwformula = "epspx_lead1~(NegEPS*epspx)" + all_columns

#Fit Regression
pw = sm.OLS.from_formula(formula=pwformula, data=train).fit()
print(pw.summary())

## Predict future earnings using trained models
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
    ##Filter for same industry
    Table = Table.loc['indsutry'=='28']
        
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
        
## Recommend Future stocks
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

#extract ibes data
ibes_id = conn.raw_sql("""
                        select ticker, oftic
                        from ibes.id
                        """)

ibes_guidance = conn.raw_sql("""
                              select ticker, eefymo, range_desc, prd_yr, prd_mon, measure, mean_at_date
                              from ibes.det_guidance where
                              prd_yr<=2018 and prd_yr>1999
                              """)

ibes = pd.merge(ibes_guidance,ibes_id, on=['ticker'])

## Join IBES Dataset to compare to analysts forecasts
fundaclean1["ibtic"] = fundaclean1["ibtic"].str.replace("@","",regex = False)
ibes["ticker"] = ibes["ticker"].str.replace("@","",regex = False)
ibes["prd_yr"] = pd.to_numeric(ibes["prd_yr"], downcast = "integer")
analysts_forecasts = pd.merge(ibes, fundaclean1, left_on=['ticker','prd_yr'], right_on=['ibtic', 'fyear'])
analysts_forecasts['datadate'] = pd.to_datetime(analysts_forecasts['datadate'])
analysts_forecasts = analysts_forecasts.set_index('datadate')
analysts_forecasts= analysts_forecasts[analysts_forecasts['prd_yr'] == 2017]
analysts_forecasts= analysts_forecasts[analysts_forecasts['measure'] == 'EPS']
analysts_forecasts = analysts_forecasts.groupby(by = "ticker", as_index=False).last()


