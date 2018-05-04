# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 18:48:47 2016

@author: John Newman
Results checked against similar code from PriceDerivatives

This code aims to:
1. Implement the Black Scholes equation to determine Option Value, and the popular Greeks
2. Create a self-replicating portfolio with different Spot Price and Volatility Dynamics to
better understand how sensitivity to different values evolve in different price and vol movements.
3. Delta Hedging Implementation

Next Step:
Implement in JupyterNotebook and create widgets that will enable sliders on inputs in order to
demonstrate the Greeks' sensitivity to input values.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def BlackScholes(tau, S, K, r, sigma, Call=True):
    """
    Inputs:
    tau = Time to Maturity
    S = Spot Price
    K = Strike Price
    r = risk-free rate
    sigma = volatility
    Call = Call/Put flag
    
    Outputs:
    npv = value of the call/put
    delta = d(npv)/d(spot price) (derivative of call value to spot price)
    gamma = d^2(npv)/d(spot price)^2 (second derivative of call value to spot price)
    vega = d(npv)/d(sigma) = (derivative of call value to volatility)
    theta = -d(npv)/d(tau) = (derivative of call value to time to maturity)
    """
    d1=(np.log(S/K)+(r+sigma*sigma/2)*tau)/(sigma*np.sqrt(tau))
    d2=d1-sigma*np.sqrt(tau)
    if Call:    
        npv=(S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d2))
        delta=norm.cdf(d1)
        theta=(-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)) - r*K*np.exp(-r*tau)*norm.cdf(d2)
        rho = K*tau*np.exp(-r*tau)*norm.cdf(d2)
    else:
        npv=(-S*norm.cdf(-d1)+K*np.exp(-r*tau)*norm.cdf(-d2))
        delta=-norm.cdf(-d1)
        theta=(-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)) + r*K*np.exp(-r*tau)*norm.cdf(-d2)
        rho = -K*tau*np.exp(-r*tau)*norm.cdf(-d2)
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    return {'npv':npv,'delta':delta,'gamma':gamma,'vega':vega,'theta':theta,'rho':rho}

class Call(object):
    def __init__(self,start,T,K,r,N):
        """
        T = Maturity in days, based on 250 trading days a year
        K = Strike Price
        start = start time
        N = Number of calls purchased (sold if negative)
        tau = time to maturity (assuming 250 trading days)
        instrinsic = value of option if it were at maturity (max(0, K-S))
        r = risk-free rate
        """
        self.T=T 
        self.K=K 
        self.start=start  #day to sell   option
        self.N=N
        self.r=r 

    def calc(self,today,vol,S):
        if today<self.start:
            return {'delta':0,'npv':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
        if today>self.T:
            return {'delta':0,'npv':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
        if today==self.T:
            return {'delta':0,'npv':0,'vega':0,'gamma':0,'theta':0,'intrinsic':self.N*max(0,S-self.K)}
        tau=(self.T-today)/250.
        call=BlackScholes(tau, S, self.K, self.r, vol)
        return {'delta':self.N*call['delta'],'npv':self.N*call['npv'],'vega':self.N*call['vega'],'gamma':self.N*call['gamma'],'theta':self.N*call['theta'],'rho':self.N*call['rho'],'intrinsic':self.N*max(0,S-self.K)}

def self_replicate(Ndays,N,S,K,r,vol,S_drift,S_rand,vol_drift,vol_rand):
    """
    Inputs:
    Ndays = Starting days
    N = Number of Contracts
    S = Starting Spot Price
    K = Strike Price
    r = Risk-Free Rate
    vol = Starting volatility (should be expressed as a decimal ex:0.3 = 30%) Must be positive
    S_drift = Drift component of Spot Price 
    S_rand = Random component of Spot Price
    vol_drift = Drift component of Spot Price 
    vol_rand = Random component of Spot Price
    """
    #Initiate Output DataFrame
    columns=('spot','vol','shares','bonds','option','delta','vega','gamma','theta', 'rho')
    df = pd.DataFrame([[S,vol,0,0,0,0,0,0,0,0]],columns=columns)
    #Days are normalized to a 250 trading day year
    dt=1/250.
    #Initialize call date
    dayToBuyCall=1
    #Days to the call Maturity
    #Debugging values
#    Ndays=100
#    K=100
#    r =0.02
#    day=5
#    vol=0.3
#    S=80
    maturityCall=Ndays

    #Initialize Call
    call=Call(dayToBuyCall,maturityCall,K,r,N)# sell one call on dayToSellCall day
    for day in range(1,Ndays+1):
        #Define a path for the Spot Price
        S *= (1.0+S_drift*dt+S_rand*np.sqrt(dt)*np.random.randn())
        vol *= (1.0+vol_drift*dt+vol_rand*np.sqrt(dt)*np.random.randn())
        #ex: S*=(1.0 + (drift % per day)*dt + volatility*sqrt(dt)*random sample from normal distribution)
            
        #Do Black-Scholes calculation for option value given day, volatility, and Spot
        callValue=call.calc(day,vol,S)

        #Create Replicated Stock and Bond portfolio
        d1=(np.log(S/K)+(r+vol*vol/2)*((call.T-day)*dt))/(vol*np.sqrt((call.T-day)*dt))
        d2=d1-vol*np.sqrt((call.T-day)*dt)
        currentNumberShares_new=S*norm.cdf(d1)
        currentBonds_new=K*np.exp(-r*(call.T-day)*dt)*norm.cdf(d2)
        if day==maturityCall:
            option=callValue['intrinsic']
        else:
            option=callValue['npv']
        gamma=callValue['gamma']
        theta=callValue['theta']
        #Dictionary only allows rho to be recovered using .get, not sure why
        rho = callValue.get('rho')
        dfnew=pd.DataFrame([[S,vol,currentNumberShares_new, currentBonds_new,option,callValue['delta'],callValue['vega'], gamma,theta, rho ]],columns=columns)
        df=df.append(dfnew,ignore_index=True)
    df.loc[:,['theta']].plot(title='Theta')
    df.loc[:,['delta']].plot(title='Delta')
    df.loc[:,['vega']].plot(title='Vega')
    df.loc[:,['gamma']].plot(title='Gamma')
    df.loc[:,['rho']].plot(title='Rho')

    df.loc[:,['spot', 'shares', 'bonds', 'option']].plot(title='Shares and Bonds')
    plt.xlim(0, Ndays)
    plt.ylim(0, df['spot'].max()*1.1)

    return df
    #print df

result = self_replicate(100, 1,40, 30, 0.01, 0.3, -0.01, 0.3, 0, 0)
#Spot = 40, Strike = 30, r = 1%, vol = 30%, Spot Drift = -1%, Spot Random = Stochastic Based on implied Vol, Homoskedastic

#Next Planned Function: Delta_Hedge with Cash
def delta_hedge(Ndays,N,S,K,r,vol,S_drift,S_rand,vol_drift,vol_rand):
    """
    Inputs:
    Ndays = Starting days
    N = Number of Contracts
    S = Starting Spot Price
    K = Strike Price
    vol = Starting volatility (should be expressed as a decimal ex:0.3 = 30%) Must be positive
    S_drift = Drift component of Spot Price 
    S_rand = Random component of Spot Price
    vol_drift = Drift component of Spot Price 
    vol_rand = Random component of Spot Price
    """
    #Initiate Output DataFrame
    columns=('spot','vol','shares', 'bonds','interest','option','portfolio','PnL','vega','gamma','theta', 'rho')
    df = pd.DataFrame([[S,vol,0,0,0,0,0,0,0,0,0,0]],columns=columns)
    #Days are normalized to a 250 trading day year
    dt=1/250.
    #Initialize optional purchase date
    dayToBuyCall = 1
    #Days to the call Maturity
    maturityCall = Ndays-1
    #Initialize Bond Principal
    new_bonds_value = 0
    #Initialize Interest
    interest = 0
    #Initialize Call
    call=Call(dayToBuyCall,maturityCall,K,r,N)# sell one call on dayToSellCall day
    for day in range(1,Ndays):
        #Define a path for the Spot Price
        S *= (1.0+S_drift*dt+S_rand*np.sqrt(dt)*np.random.randn())
        #ex: S*=(1.0 + (drift % per day)*dt + volatility*sqrt(dt)*random sample from normal distribution)
        vol *= (1.0+vol_drift*dt+vol_rand*np.sqrt(dt)*np.random.randn())

        if day==dayToBuyCall: #buy call
            callValue = call.calc(day,vol,S)
            new_bonds_value -= callValue['npv']
        #delta hedge
        #Do Black-Scholes calculation given day, volatility, and Spot
        callValue=call.calc(day,vol,S)

        #Determine the results of a delta hedge
        old_shares = df.iloc[day-1].shares
        #delta = norm.cdf(n1)
        #Note: shares will be negative delta
        delta=callValue['delta']
        #determine amount of new shares to be bought/sold
        new_shares = old_shares + delta
        #Fund the new shares
        new_bonds_value+=(new_shares)*S
        #Calculate Interest at risk-free rate
        interest += new_bonds_value*(np.exp(r*dt)-1)
        if day == maturityCall:
            #Resolve Call
            new_bonds_value+=call.calc(day,vol,S)['intrinsic']
        #Portfolio Value = Contribution from interest, Bonds, Stocks and the option
        portfolio_npv = interest + new_bonds_value - delta*S + callValue['npv']
        #Realized Profit & Loss
        profit_and_loss = portfolio_npv - df.iloc[day-1].portfolio
        if day==maturityCall:
            option=callValue['intrinsic']
        else:
            option=callValue['npv']
        gamma=callValue['gamma']
        theta=callValue['theta']
        #Dictionary only allows rho to be recovered using .get, not sure why
        rho = callValue.get('rho')
        dfnew=pd.DataFrame([[S,vol,-delta, new_bonds_value, interest,abs(option), portfolio_npv,profit_and_loss,callValue['vega'], gamma,theta*dt, rho ]],columns=columns)
        df=df.append(dfnew,ignore_index=True)
    
    #Plot the Greeks
    df.loc[:,['theta']].plot(title='Theta')
    df['delta'] = df['shares']*-1
    df.loc[:,['delta']].plot(title='Delta')
    df.loc[:,['vega']].plot(title='Vega')
    df.loc[:,['gamma']].plot(title='Gamma')
    df.loc[:,['rho']].plot(title='Rho')
    
    plt.xlim(0, Ndays)
    df.loc[:,['portfolio','spot','option']].plot(title='Spot vs Option Price vs Portfolio Value')
    df.loc[:,['PnL']].hist(bins=20)

    return df
    #print df

delta_hedge_result = delta_hedge(250, 10, 35, 30, 0.01, 0.3, 0, 0, 0, 0)

delta_hedge_constant_vol = delta_hedge(250, 10, 30, 30, 0.01, 0.3, -0.3*-0.3*.5, 0.3, 0, 0)
#Spot = Strike, r = 1%, vol = 30%, S_drift = -1/2 * vol^2, S_rand = vol (stochastic), homoskedastic
    