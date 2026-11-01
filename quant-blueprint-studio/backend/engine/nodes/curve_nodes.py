"""Curve-fitting notebook workflow. Full-sample approximation is explicitly in-sample."""
from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .study_nodes import chart, metrics


def monthly_curve(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(frame.index,pd.DatetimeIndex) or 'Close' not in frame:
        raise ValueError('Connect daily Market Data with dated Close prices.')
    daily=frame[['Close']].sort_index().dropna()
    monthly=daily.resample('ME').last().dropna()
    # Yahoo's current daily series can include the unfinished current month.
    if params.get('exclude_incomplete_month',True) and len(monthly):
        now=pd.Timestamp.now(tz=daily.index.tz)
        if monthly.index[-1].to_period('M') >= now.to_period('M'):
            monthly=monthly.iloc[:-1]
    if len(monthly)<24:
        raise ValueError(f'Only {len(monthly)} monthly closes; at least 24 required. Increase history.')
    result=monthly.copy()
    result['Coordinate']=np.linspace(-2,2,len(result))
    result.attrs['raw_rows']=len(frame)
    return result


def vectors(actual: np.ndarray,predicted: np.ndarray) -> dict[str,float]:
    arrays=[np.asarray(actual),np.asarray(predicted)]
    if any(v.ndim not in (1,2) or (v.ndim==2 and v.shape[1]!=1) for v in arrays):
        raise ValueError("Require vectors or single-column arrays, never a multi-output matrix.")
    a,b=[v.reshape(-1) for v in arrays]
    if a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Require one finite scalar prediction for each target; broadcasting is prohibited.')
    result=metrics(a,b)
    result['RMSE pips']=result['RMSE']/.0001
    return result


def curve_fit(frame: pd.DataFrame,p: dict[str,Any],progress: Callable[[dict[str,Any]],None] | None = None) -> dict[str,Any]:
    if not {'Close','Coordinate'}.issubset(frame.columns):
        raise ValueError('Connect Monthly Curve Data first; technical indicators are not required.')
    mode=str(p.get('evaluation','in_sample'))
    if mode not in ('in_sample','temporal'):raise ValueError('Evaluation must be in_sample or temporal.')
    variant=str(p.get('variant','ols'))
    if variant not in ('ols','polynomial_net','dense'):raise ValueError('Choose ols, polynomial_net or dense.')
    degree=int(p.get('degree',5)); epochs=int(p.get('epochs',1500)); batch=int(p.get('batch_size',32))
    lr=float(p.get('lr',.001)); l2=float(p.get('l2',0)); patience=int(p.get('patience',100))
    if not 1<=degree<=10 or not 1<=epochs<=5000 or not 2<=batch<=1024 or not 1e-6<=lr<=.1 or not 0<=l2<=1 or not 1<=patience<=1000:
        raise ValueError('Degree 1–10, epochs 1–5000, batch 2–1024, LR .000001–.1, L2 0–1, patience 1–1000 required.')
    n=len(frame); a,b=n,n
    if mode=='temporal':
        tr,va=float(p.get('train_ratio',.65)),float(p.get('val_ratio',.15))
        if tr<.3 or va<.05 or tr+va>.95:raise ValueError('Require TR≥.3, VA≥.05 and TE≥.05.')
        a,b=int(n*tr),int(n*(tr+va))
        if min(a,b-a,n-b)<3:raise ValueError('Each temporal split needs at least 3 monthly observations.')
    x=frame.Coordinate.to_numpy(dtype=np.float64)[:,None]
    raw=frame.Close.to_numpy(dtype=np.float64)
    mean=float(raw[:a].mean()); y=raw-mean
    grid=np.linspace(float(x.min()),float(x.max()),300)[:,None]
    start=perf_counter(); history=[]; parameter_count=degree+1
    label=str(p.get('name','Curve fit'))
    if variant=='ols':
        coefficients=np.polynomial.polynomial.polyfit(x[:a,0],y[:a],degree)
        predicted=np.polynomial.polynomial.polyval(x[:,0],coefficients)
        smooth=np.polynomial.polynomial.polyval(grid[:,0],coefficients)
    else:
        polynomial=PolynomialFeatures(degree=degree,include_bias=False) if variant=='polynomial_net' else None
        inputs=polynomial.fit_transform(x) if polynomial else x
        inputs_grid=polynomial.transform(grid) if polynomial else grid
        normalize=bool(p.get('normalize',True)) or polynomial is not None
        scaler=StandardScaler().fit(inputs[:a]) if normalize else None
        inputs=scaler.transform(inputs) if scaler else inputs
        inputs_grid=scaler.transform(inputs_grid) if scaler else inputs_grid
        activation=str(p.get('activation','ReLU'))
        activations={'ReLU':nn.ReLU,'Swish':nn.SiLU,'Tanh':nn.Tanh,'GELU':nn.GELU,'Sigmoid':nn.Sigmoid}
        if activation not in activations:raise ValueError('Unsupported activation.')
        widths=[] if polynomial else [int(s.strip()) for s in str(p.get('widths','256')).split(',') if s.strip()]
        if not polynomial and (not widths or len(widths)>5 or any(w<1 or w>512 for w in widths)):raise ValueError('Choose 1–5 hidden layers, 1–512 units each.')
        torch.manual_seed(int(p.get('seed',100)))
        layers: list[nn.Module]=[]; dimension=inputs.shape[1]
        for width in widths:
            layers.extend([nn.Linear(dimension,width),activations[activation]()]); dimension=width
        layers.append(nn.Linear(dimension,1)); net=nn.Sequential(*layers)
        parameter_count=sum(v.numel() for v in net.parameters())
        optimizer=str(p.get('optimizer','RMSprop'))
        optimizers={'Adam':torch.optim.Adam,'RMSprop':torch.optim.RMSprop,'SGD':torch.optim.SGD,'AdamW':torch.optim.AdamW}
        if optimizer not in optimizers:raise ValueError('Unsupported optimizer.')
        opt=optimizers[optimizer](net.parameters(),lr=lr)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=30,min_lr=1e-5) if p.get('reduce_lr',False) else None
        xt=torch.tensor(inputs,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
        best=float('inf'); state=deepcopy(net.state_dict()); stale=0
        for epoch in range(epochs):
            net.train()
            for idx in torch.randperm(a).split(batch):
                opt.zero_grad(set_to_none=True)
                prediction=net(xt[idx]).squeeze(1)
                data_loss=nn.functional.mse_loss(prediction,yt[idx])
                penalty=sum(w.square().sum() for name,w in net.named_parameters() if name.endswith('weight'))
                (data_loss+l2*penalty).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                training=float(nn.functional.mse_loss(net(xt[:a]).squeeze(1),yt[:a]))
                validation=float(nn.functional.mse_loss(net(xt[a:b]).squeeze(1),yt[a:b])) if mode=='temporal' else training
            if not np.isfinite(training+validation):raise ValueError('Non-finite loss. Reduce learning rate.')
            row={'epoch':epoch+1,'train':training,'learning_rate':float(opt.param_groups[0]['lr'])}
            if mode=='temporal':row['validation']=validation
            history.append(row)
            if progress and ((epoch+1)%max(1,epochs//100)==0 or epoch==0 or epoch==epochs-1):progress({**row,'epochs':epochs,'detail':label})
            if scheduler:scheduler.step(validation if mode=='temporal' else training+l2*float(penalty.detach()))
            if mode=='temporal':
                if validation<best:best,state,stale=validation,deepcopy(net.state_dict()),0
                else:stale+=1
                if stale>=patience:break
        if mode=='temporal':net.load_state_dict(state)
        with torch.no_grad():
            predicted=net(xt).squeeze(1).numpy().astype(float)
            smooth=net(torch.tensor(inputs_grid,dtype=torch.float32)).squeeze(1).numpy().astype(float)
    seconds=perf_counter()-start
    evaluation=slice(b,None) if mode=='temporal' else slice(None)
    score=vectors(y[evaluation],predicted[evaluation])
    table=[{'model':label,'scope':'test / temporal extrapolation' if mode=='temporal' else 'in-sample curve fit','activation':'—' if variant!='dense' else str(p.get('activation','ReLU')),'optimizer':'OLS' if variant=='ols' else str(p.get('optimizer','RMSprop')),'parameters':parameter_count,'epochs':len(history),'seconds':seconds,**score}]
    dates=[str(d.date()) for d in frame.index]
    rows=[{'date':dates[i],'coordinate':float(x[i,0]),'actual':float(y[i]),'fitted':float(predicted[i])} for i in range(n)]
    charts=[chart('Monthly centered EUR/USD · full curve (fit / extrapolation)',rows,['actual','fitted']),chart('Smooth curve · artificial coordinate', [{'coordinate':float(v),'fitted':float(pred)} for v,pred in zip(grid[:,0],smooth)],['fitted'],'coordinate')]
    if history:charts.append(chart('Data MSE by epoch',history,['train','validation'] if mode=='temporal' else ['train'],'epoch'))
    if variant=='ols':
        learning=[]
        for count in range(max(10,degree+2),a+1,20):
            coef=np.polynomial.polynomial.polyfit(x[:count,0],y[:count],3)
            fit=np.polynomial.polynomial.polyval(x[:,0],coef)
            learning.append({'samples':count,'training_MSE':vectors(y[:count],fit[:count])['MSE'],'remaining_MSE':vectors(y[count:],fit[count:])['MSE'] if count<n else 0.})
        charts.append(chart('Cubic prefix fit · extrapolation, not training-only error',learning,['training_MSE','remaining_MSE'],'samples'))
    return {'study':True,'topic':'curve_fit','selected':label,'scope':table[0]['scope'],'features':['Coordinate'],'splits':{'train':a,'validation':b-a,'test':n-b},'metrics':score,'table':table,'charts':charts,'notes':[f'Target = monthly Close − fitted training mean ({mean:.8f} USD/EUR). Artificial coordinate spans −2…2; it is not an economic predictor.','In-sample errors measure approximation, not forecasting ability.' if mode=='in_sample' else 'Training mean and feature scaler use TR only. VA chooses the neural checkpoint; TE is temporal extrapolation, evaluated afterwards.','PyTorch implementation of the Keras architectures; initialization, optimizer defaults and timings differ. RMSE pips uses 0.0001 USD/EUR per pip.']}
