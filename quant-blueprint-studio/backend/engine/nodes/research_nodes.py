"""Causal data cleaning and descriptive quantitative feature views."""
from typing import Any
import numpy as np
import pandas as pd


def data_quality(frame: pd.DataFrame,p: dict[str,Any]) -> pd.DataFrame:
    if not isinstance(frame.index,pd.DatetimeIndex):raise ValueError('Data Quality requires dated rows.')
    duplicates=str(p.get('duplicates','error')); missing=str(p.get('missing','drop'))
    if duplicates not in ('error','last') or missing not in ('error','drop','ffill'):
        raise ValueError('Choose duplicate error/last and missing error/drop/ffill.')
    if frame.index.has_duplicates and duplicates=='error':raise ValueError('Duplicate timestamps found; choose last explicitly to resolve.')
    result=frame.sort_index().copy()
    if duplicates=='last':result=result.loc[~result.index.duplicated(keep='last')]
    result=result.replace([np.inf,-np.inf],np.nan)
    if missing=='error' and result.isna().any().any():raise ValueError('Missing/non-finite values found.')
    limit=int(p.get('ffill_limit',3))
    if not 1<=limit<=30:raise ValueError('Forward-fill limit must be 1–30 observations.')
    if missing=='ffill':result=result.ffill(limit=limit)
    if missing!='error':result=result.dropna()
    if len(result)<3:raise ValueError('Cleaning leaves fewer than three rows.')
    result.attrs['raw_rows']=frame.attrs.get('raw_rows',len(frame))
    return result


def quant_features(frame: pd.DataFrame,p: dict[str,Any]) -> pd.DataFrame:
    if not isinstance(frame.index,pd.DatetimeIndex) or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Quant features require sorted unique dated rows; use Data Quality first.')
    column=str(p.get('price_column','Close'));window=int(p.get('window',20));annual=int(p.get('periods_per_year',252))
    if column not in frame or not 2<=window<=252 or not 1<=annual<=366:
        raise ValueError('Choose an available price column, window 2–252 and periods/year 1–366.')
    price=frame[column].to_numpy(dtype=float)
    if not np.isfinite(price).all() or np.any(price<=0):raise ValueError('Quant features require positive finite prices.')
    names=['SimpleReturn','LogReturn','CumulativeLogReturn','RollingVolatility','Drawdown','RollingSharpe']
    result=frame.copy();r=result[column].pct_change(fill_method=None);log=np.log(result[column]/result[column].shift(1))
    rf=float(p.get('risk_free_rate',.02))
    if not np.isfinite(rf) or rf<=-1:raise ValueError('Annual risk-free rate must be finite and greater than -1.')
    std=r.rolling(window).std(ddof=1);excess=r-np.expm1(np.log1p(rf)/annual)
    result['SimpleReturn']=r;result['LogReturn']=log
    result['CumulativeLogReturn']=np.log(result[column]/result[column].iloc[0])
    result['RollingVolatility']=std*np.sqrt(annual)
    result['Drawdown']=result[column]/result[column].cummax()-1
    result['RollingSharpe']=excess.rolling(window).mean()/std.replace(0,np.nan)*np.sqrt(annual)
    # Undefined Sharpe is retained as NaN; selecting it requires an explicit cleaning policy.
    result.attrs['raw_rows']=frame.attrs.get('raw_rows',len(frame))
    return result
