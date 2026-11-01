"""Executable teaching experiments, with chronological isolation of held-out data.

An experiment pin contains tabular comparisons and named charts; no executable
code is accepted from a graph. Checkpoints stay in memory for the current run.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .scaling_nodes import fit_scaler, affine_audit, disabled_audit
from scipy.stats import wasserstein_distance

FEATURES = ['LogReturn', 'Volatility', 'RSI', 'EMA', 'SMA', 'VWAP', 'MACD', 'MACDSignal', 'BBUpper', 'BBMiddle', 'BBLower']
TOPICS = ('feedforward', 'optimizers', 'regularization', 'hyperparameters', 'cnn', 'recurrent', 'autoencoders', 'generative', 'cdcgan', 'double_dqn')
Progress = Callable[[dict[str, Any]], None]


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    dates: list[str]
    gap: np.ndarray
    intraday: np.ndarray
    train: int
    validation: int
    names: list[str]


def samples(frame: pd.DataFrame, p: dict[str, Any]) -> Samples:
    names = [s.strip() for s in str(p.get('feature_columns', ','.join(FEATURES))).split(',') if s.strip()]
    if not names or len(names) != len(set(names)) or set(names) - set(frame.columns):
        raise ValueError('Select unique available feature columns; enable their Technical Features indicators.')
    window = int(p.get('lookback', 1))
    tr, va = float(p.get('train_ratio', .65)), float(p.get('val_ratio', .15))
    if not 1 <= window <= 120 or not .3 <= tr < .9 or va < .05 or tr + va > .95:
        raise ValueError('Require lookback 1–120, train ≥ .3, validation ≥ .05, test ≥ .05.')
    # Window ends at close t; its label belongs to session t+1. Never shift labels into features.
    raw = frame[names].to_numpy(dtype=np.float64)
    x = np.stack([raw[i-window:i] for i in range(window, len(raw))]) if len(raw) > window else np.empty((0,window,len(names)))
    y = (10000 * np.log(frame.Close / frame.Open)).to_numpy()[window:]
    gap = np.log(frame.Open / frame.Close.shift(1)).to_numpy()[window:]
    n = len(x)
    a, b = int(n*tr), int(n*(tr+va))
    if min(a, b-a, n-b) < 8:
        raise ValueError(f'Insufficient rows: {len(frame)} usable, lookback {window}, TR/VA/TE {a}/{b-a}/{n-b}. Increase history or reduce lookback; each split needs 8.')
    return Samples(x, y, [str(d.date()) for d in frame.index[window:]], gap, y / 10000, a, b, names)


def scaled(x: np.ndarray, fit_end: int, audits: list[dict[str,Any]] | None = None, names: list[str] | None = None, scope: str = 'training') -> np.ndarray:
    # Fit on the union of observed training rows, without overweighting overlap.
    if x.ndim!=3 or not 1<=fit_end<=len(x) or not np.isfinite(x).all():
        raise ValueError('Sequence scaler needs finite samples × time × features and a valid training boundary.')
    fit = np.concatenate([x[0], x[1:fit_end, -1]], axis=0)
    scaler,audit = fit_scaler(fit,names or [f'feature_{i+1}' for i in range(x.shape[-1])],scope=scope)
    audit['note']+=' Fit uses unique observed training bars, not repeated overlapping windows.'
    if audits is not None: audits.append(audit)
    return scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype(np.float32)


class Predictor(nn.Module):
    def __init__(self, n: int, window: int, kind: str, widths: list[int], dropout: float = 0, batchnorm: bool = False) -> None:
        super().__init__()
        self.kind = kind
        self.bidirectional = kind.startswith('Bi')
        if kind == 'CNN':
            self.encoder = nn.Sequential(nn.Conv1d(n,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
            d = 64
        elif kind in ('LSTM','GRU','BiLSTM','BiGRU'):
            cell = nn.LSTM if 'LSTM' in kind else nn.GRU
            self.encoder = cell(n,32,batch_first=True,bidirectional=self.bidirectional)
            d = 64 if self.bidirectional else 32
        else:
            self.encoder = nn.Flatten()
            d = n*window
        layers: list[nn.Module] = []
        for width in widths if kind == 'MLP' else []:
            layers += [nn.Linear(d,width)]
            if batchnorm:
                layers += [nn.BatchNorm1d(width)]
            layers += [nn.ReLU()]
            if dropout:
                layers += [nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d,1)]
        self.head = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == 'CNN':
            h = self.encoder(x.transpose(1,2))
        elif self.kind in ('LSTM','GRU','BiLSTM','BiGRU'):
            _, state = self.encoder(x)
            h = state[0] if isinstance(state,tuple) else state
            h = torch.cat([h[-2],h[-1]],dim=1) if self.bidirectional else h[-1]
        else:
            h = self.encoder(x)
        return self.head(h).squeeze(1)


def fit_predictor(x: np.ndarray, y: np.ndarray, a: int, b: int, cfg: dict[str, Any], p: dict[str, Any], progress: Progress | None) -> tuple[nn.Module, float, list[dict[str, Any]], tuple[float,float]]:
    torch.manual_seed(int(p.get('seed',42)))
    classification = bool(cfg.get('classification',False))
    mean, std = (0.,1.) if classification else (float(y[:a].mean()), max(float(y[:a].std()),1e-8))
    labels = (y>0).astype(np.float32) if classification else ((y-mean)/std).astype(np.float32)
    net = Predictor(x.shape[-1],x.shape[1],str(cfg.get('kind','MLP')),cfg.get('widths',[64,32]),float(cfg.get('dropout',0)),bool(cfg.get('batchnorm',False)))
    lr = float(cfg.get('lr',p.get('lr',.001)))
    opt_class = {'SGD':torch.optim.SGD,'RMSprop':torch.optim.RMSprop,'AdamW':torch.optim.AdamW,'Adam':torch.optim.Adam}[str(cfg.get('optimizer','Adam'))]
    opt = opt_class(net.parameters(), lr=lr, weight_decay=float(cfg.get('l2',0)))
    loss_fn = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    xt, yt = torch.tensor(x), torch.tensor(labels)
    best, state, stale = float('inf'), deepcopy(net.state_dict()), 0
    history: list[dict[str,Any]] = []
    batch = int(p.get('batch_size',64))
    for epoch in range(int(p.get('epochs',12))):
        net.train()
        for idx in torch.randperm(a).split(batch):
            if len(idx)<2 and cfg.get('batchnorm'): continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[idx]),yt[idx])
            penalty = sum(w.abs().sum() for name,w in net.named_parameters() if name.endswith('weight'))
            (loss + float(cfg.get('l1',0))*penalty).backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(xt[a:b]),yt[a:b]))
            tr = float(loss_fn(net(xt[:a]),yt[:a]))
        if not np.isfinite(val): raise ValueError('Non-finite loss. Reduce learning rate.')
        row = {'epoch':epoch+1,'train':tr,'validation':val}
        history.append(row)
        if progress: progress({**row,'epochs':int(p.get('epochs',12)),'detail':str(cfg.get('name','training'))})
        if val < best:
            best,state,stale = val,deepcopy(net.state_dict()),0
        else: stale += 1
        if stale >= int(cfg.get('patience',p.get('patience',5))): break
    net.load_state_dict(state)
    net.eval()
    return net,best,history,(mean,std)


def predict(net: nn.Module, x: np.ndarray, scale: tuple[float,float], classification: bool = False) -> np.ndarray:
    with torch.no_grad():
        values = net(torch.tensor(x))
        return torch.sigmoid(values).numpy() if classification else values.numpy()*scale[1]+scale[0]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str,float]:
    mse = float(mean_squared_error(actual,predicted))
    return {'MSE':mse,'RMSE':float(np.sqrt(mse)),'MAE':float(mean_absolute_error(actual,predicted)),'R²':float(r2_score(actual,predicted))}


def chart(title: str, rows: list[dict[str,Any]], series: list[str], x_key: str = 'date', chart_type: str = 'line') -> dict[str,Any]:
    return {'title':title,'rows':rows,'series':series,'x_key':x_key,'chart_type':chart_type}


def trading(s: Samples, predictions: np.ndarray, p: dict[str,Any], classification: bool = False, signals: np.ndarray | None = None) -> tuple[dict[str,float],dict[str,Any]]:
    signal = signals if signals is not None else np.where(predictions > (.5 if classification else float(p.get('threshold_bps',0))),1,-1)
    intraday = np.expm1(s.intraday[s.validation:])
    # Supervised experiments are flat overnight: enter at open, exit at close.
    costs = 2*np.abs(signal)*float(p.get('cost_bps',5))/10000
    net = signal*intraday-costs
    equity = np.cumprod(1+net)
    market = np.cumprod(1+intraday)
    stats = {'ROI':float(equity[-1]-1),'Sharpe':float(net.mean()/net.std(ddof=1)*np.sqrt(252)) if net.std() else 0.,'Max drawdown':float(np.min(equity/np.maximum.accumulate(np.r_[1.,equity])[1:]-1)),'Turnover':float(2*np.abs(signal).sum())}
    rows = [{'date':d,'strategy':float(v-1),'market':float(m-1),'signal':int(a)} for d,v,m,a in zip(s.dates[s.validation:],equity,market,signal)]
    return stats,chart('Intraday holdout · net of round-trip costs',rows,['strategy','market'])


def supervised(s: Samples, p: dict[str,Any], progress: Progress | None) -> dict[str,Any]:
    topic = str(p['topic'])
    audits: list[dict[str,Any]] = []
    x = scaled(s.x,s.train,audits,s.names)
    base = {'kind':'MLP','widths':[int(v) for v in str(p.get('widths','64,32')).split(',')]}
    if not base['widths'] or len(base['widths'])>4 or any(v<2 or v>512 for v in base['widths']): raise ValueError('Choose 1–4 hidden widths, each 2–512.')
    candidates: list[dict[str,Any]] = [{**base,'name':'MLP'}]
    if topic == 'optimizers': candidates = [{**base,'name':o,'optimizer':o} for o in ('SGD','RMSprop','AdamW')]
    if topic == 'regularization':
        candidates = [{**base,'classification':True,'name':name,'patience':int(p.get('epochs',12)),**cfg} for name,cfg in [('Baseline',{}),('Dropout',{'dropout':.3}),('L1',{'l1':1e-5}),('L2',{'l2':1e-3}),('BatchNorm',{'batchnorm':True}),('Early stopping',{'patience':int(p.get('patience',5))})]]
    if topic == 'cnn': candidates = [{'kind':'CNN','name':'CNN 11→32→64→pool→1'},{**base,'name':'Flattened MLP'}]
    if topic == 'recurrent': candidates = [{'kind':k,'name':k} for k in ('LSTM','GRU','BiLSTM','BiGRU')]
    table: list[dict[str,Any]] = []
    if topic == 'hyperparameters':
        rng = np.random.default_rng(int(p.get('seed',42)))
        trials = int(p.get('trials',6))
        # Expanding folds wholly inside TR. Refit both X/y scales independently.
        for trial in range(trials):
            width, depth = int(rng.choice([32,64,128])),int(rng.choice([1,2,3]))
            cfg = {'kind':'MLP','widths':[width]*depth,'lr':float(10**rng.uniform(-4,-2.5)),'l2':float(rng.choice([0,1e-4,1e-3])),'name':f'Trial {trial+1}'}
            scores=[]
            for fraction in (.5,.75):
                a, b = int(s.train*fraction), min(s.train,int(s.train*(fraction+.25)))
                fold = scaled(s.x[:b],a,audits,s.names,f'TR-only CV trial {trial+1}, fold {fraction}')
                net,_,_,ys = fit_predictor(fold,s.y[:b],a,b,cfg,p,progress)
                audits.append(affine_audit(s.y[:a],ys[0],ys[1],f'CV trial {trial+1}, fold {fraction}: target y',scope='TR-only CV training'))
                scores.append(metrics(s.y[a:b],predict(net,fold[a:b],ys))['RMSE'])
            cfg['cv_RMSE_bps'] = float(np.mean(scores))
            candidates.append(cfg)
        candidates = candidates[1:]
        table = [{'model':c['name'],'widths':','.join(map(str,c['widths'])),'lr':c['lr'],'L2':c['l2'],'CV RMSE (bps)':c['cv_RMSE_bps']} for c in candidates]
        candidates = [min(candidates,key=lambda c:c['cv_RMSE_bps'])]
    winner: tuple[nn.Module,tuple[float,float],dict[str,Any],list[dict[str,Any]]] | None = None
    best = float('inf')
    for cfg in candidates:
        net,val,history,ys = fit_predictor(x,s.y,s.train,s.validation,cfg,p,progress)
        score = val if cfg.get('classification') else val*ys[1]**2
        table.append({'model':cfg['name'],'validation BCE' if cfg.get('classification') else 'validation MSE (bps²)':score,'epochs':len(history),'parameters':sum(v.numel() for v in net.parameters())})
        if score < best: best,winner = score,(net,ys,cfg,history)
    assert winner is not None
    net,ys,cfg,history = winner
    classification = bool(cfg.get('classification'))
    if not classification: audits.append(affine_audit(s.y[:s.train],ys[0],ys[1],'Regression target y'))
    else: audits.append(disabled_audit(s.train,'Binary target y','training','Binary labels stay 0/1. Sigmoid probabilities are not standardized.'))
    predicted = predict(net,x[s.validation:],ys,classification)
    actual = (s.y[s.validation:]>0).astype(float) if classification else s.y[s.validation:]
    summary = metrics(actual,predicted)
    if classification: summary['Accuracy'] = float(np.mean((predicted>=.5)==actual))
    else:
        # Ridge is a baseline, not used for neural architecture selection.
        ridge = Ridge(alpha=1).fit(x[:s.train].reshape(s.train,-1),s.y[:s.train])
        summary['Ridge RMSE (bps)'] = metrics(actual,ridge.predict(x[s.validation:].reshape(len(actual),-1)))['RMSE']
    stats,equity = trading(s,predicted,p,classification)
    summary.update(stats)
    rows = [{'date':d,'actual':float(a),'predicted':float(b)} for d,a,b in zip(s.dates[s.validation:],actual,predicted)]
    return {'selected':cfg['name'],'scaling_audits':audits,'table':table,'metrics':summary,'charts':[chart('Direction probability' if classification else 'Next-session intraday log return · bps',rows,['actual','predicted']),chart('Selected checkpoint · data loss',history,['train','validation'],'epoch'),equity]}


class Autoencoder(nn.Module):
    def __init__(self,d: int,z: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,z))
        self.decoder = nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,d))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def autoencoders(s: Samples,p: dict[str,Any],progress: Progress | None) -> dict[str,Any]:
    audits: list[dict[str,Any]] = []
    x = scaled(s.x[:,-1:,:],s.train,audits,s.names)[:,0,:]
    xt = torch.tensor(x)
    z = min(int(p.get('latent_dim',3)),x.shape[1],s.train)
    pca = PCA(n_components=z).fit(x[:s.train])
    outputs: dict[str,tuple[np.ndarray,np.ndarray]] = {'PCA':(pca.transform(x),pca.inverse_transform(pca.transform(x)))}
    for name,noise in [('AE',0.),('DAE',.1)]:
        torch.manual_seed(int(p.get('seed',42)))
        net = Autoencoder(x.shape[1],z)
        opt = torch.optim.Adam(net.parameters(),lr=float(p.get('lr',.001)))
        best,state,stale = float('inf'),deepcopy(net.state_dict()),0
        for epoch in range(int(p.get('epochs',12))):
            net.train()
            for idx in torch.randperm(s.train).split(int(p.get('batch_size',64))):
                clean = xt[idx]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(clean+noise*torch.randn_like(clean)),clean)
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): val = float(nn.functional.mse_loss(net(xt[s.train:s.validation]),xt[s.train:s.validation]))
            if progress: progress({'epoch':epoch+1,'epochs':int(p.get('epochs',12)),'validation':val,'detail':name})
            if val<best: best,state,stale = val,deepcopy(net.state_dict()),0
            else: stale+=1
            if stale>=int(p.get('patience',5)): break
        net.load_state_dict(state); net.eval()
        with torch.no_grad(): outputs[name] = (net.encoder(xt).numpy(),net(xt).numpy())
    table=[]
    for name,(latent,recon) in outputs.items():
        val=float(np.mean((recon[s.train:s.validation]-x[s.train:s.validation])**2))
        table.append({'model':name,'validation reconstruction MSE':val})
    selected = min((row for row in table if row['model']!='PCA'),key=lambda r:float(r['validation reconstruction MSE']))['model']
    summary: dict[str,float] = {}
    rows: list[dict[str,Any]] = [{'date':d,'actual':float(y)} for d,y in zip(s.dates[s.validation:],s.y[s.validation:])]
    for name,(latent,recon) in outputs.items():
        ridge = Ridge(alpha=1).fit(latent[:s.train],s.y[:s.train])
        forecast = ridge.predict(latent[s.validation:])
        summary[f'{name} test reconstruction MSE'] = float(np.mean((recon[s.validation:]-x[s.validation:])**2))
        summary[f'{name} downstream RMSE (bps)'] = metrics(s.y[s.validation:],forecast)['RMSE']
        for row,v in zip(rows,forecast): row[name] = float(v)
    errors = [{'date':d,**{name:float(np.mean((recon[s.validation+i]-x[s.validation+i])**2)) for name,(_,recon) in outputs.items()}} for i,d in enumerate(s.dates[s.validation:])]
    return {'selected':selected,'scaling_audits':audits,'table':table,'metrics':summary,'charts':[chart('Latent factors → Ridge · intraday bps',rows,['actual','PCA','AE','DAE']),chart('Holdout reconstruction MSE · standardized features',errors,['PCA','AE','DAE'])]}


class ConditionalGenerator(nn.Module):
    def __init__(self,horizon: int,z: int,convolutional: bool,assets: int = 1) -> None:
        super().__init__()
        self.conv = convolutional
        self.regime = nn.Embedding(3,3)
        self.asset = nn.Embedding(assets,2)
        condition_dim = 8 if convolutional else 1
        self.dense = nn.Sequential(nn.Linear(z+condition_dim,64),nn.ReLU(),nn.Linear(64,32*(horizon//4) if convolutional else horizon))
        self.horizon = horizon
        self.upsample = nn.Sequential(nn.ConvTranspose1d(32,16,4,2,1),nn.LeakyReLU(.2),nn.ConvTranspose1d(16,1,4,2,1)) if convolutional else nn.Identity()
    def condition(self,c: torch.Tensor) -> torch.Tensor:
        if not self.conv: return c[:,:1]
        regime = torch.where(c[:,1]<-.5,0,torch.where(c[:,1]>.5,2,1)).long()
        return torch.cat([c[:,:3],self.regime(regime),self.asset(c[:,3].long())],1)
    def forward(self,z: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
        h = self.dense(torch.cat([z,self.condition(c)],1))
        return self.upsample(h.reshape(-1,32,self.horizon//4)).squeeze(1) if self.conv else h


class Discriminator(nn.Module):
    def __init__(self,horizon: int,conv: bool,assets: int = 1) -> None:
        super().__init__()
        self.conv = conv
        self.asset = nn.Embedding(assets,2)
        self.net = nn.Sequential(nn.Conv1d(6,16,3,padding=1),nn.LeakyReLU(.2),nn.Conv1d(16,32,3,padding=1),nn.LeakyReLU(.2),nn.Flatten(),nn.Linear(32*horizon,1)) if conv else nn.Sequential(nn.Linear(horizon+1,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
        condition = torch.cat([c[:,:3],self.asset(c[:,3].long())],1) if self.conv else c[:,:1]
        inp = torch.cat([x[:,None,:],condition[:,:,None].expand(-1,-1,x.shape[1])],1) if self.conv else torch.cat([x,c[:,:1]],1)
        return self.net(inp).squeeze(1)


class CVAE(nn.Module):
    def __init__(self,h: int,z: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(h+1,64),nn.ReLU())
        self.mu,self.logvar = nn.Linear(64,z),nn.Linear(64,z)
        self.decoder = nn.Sequential(nn.Linear(z+1,64),nn.ReLU(),nn.Linear(64,h))
    def generate(self,z: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z,c[:,:1]],1))
    def forward(self,x: torch.Tensor,c: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        h=self.encoder(torch.cat([x,c[:,:1]],1))
        mu,lv=self.mu(h),self.logvar(h).clamp(-8,8)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
        return self.generate(z,c),.5*(mu.square()+lv.exp()-1-lv).sum(1).mean()


def lag1(a: np.ndarray) -> float:
    left,right=a[:,:-1].ravel(),a[:,1:].ravel()
    return float(np.corrcoef(left,right)[0,1]) if min(left.std(),right.std())>1e-12 else 0.


def tail_dependence(a: np.ndarray) -> float:
    threshold=float(np.quantile(a,.05))
    previous=a[:,:-1]<threshold
    return float(np.mean((a[:,1:]<threshold)[previous])) if previous.any() else 0.


def distribution_score(real: np.ndarray,fake: np.ndarray) -> float:
    # Distribution distance + temporal dependence; fixed validation noise.
    return float(wasserstein_distance(real.ravel(),fake.ravel()) + abs(real.std()-fake.std()) + .25*abs(lag1(real)-lag1(fake)))


def generative(frame: pd.DataFrame,p: dict[str,Any],progress: Progress | None, additional: list[pd.DataFrame]) -> dict[str,Any]:
    torch.manual_seed(int(p.get('seed',42)))
    horizon,z = int(p.get('horizon',16)),int(p.get('latent_dim',4))
    if horizon%4 or not 4<=horizon<=64: raise ValueError('Scenario horizon must be a multiple of 4 between 4 and 64.')
    frames=[frame,*additional]
    common=frame.index
    for f in additional: common=common.intersection(f.index)
    if len(common)<100: raise ValueError('Conditional assets need at least 100 common usable dates.')
    a,b = int(len(common)*float(p.get('train_ratio',.65))),int(len(common)*(float(p.get('train_ratio',.65))+float(p.get('val_ratio',.15))))
    windows,conditions,groups = [],[],[]
    for asset, f in enumerate(frames):
        f=f.loc[common]
        r=f.LogReturn.to_numpy(); vol=f.Volume.to_numpy(dtype=float)
        for i in range(20,len(r)-horizon+1):
            group = 0 if i+horizon<=a else 1 if i>=a and i+horizon<=b else 2 if i>=b else -1
            if group<0: continue
            windows.append(r[i:i+horizon])
            conditions.append([np.log(max(r[i-20:i].std(ddof=1),1e-8)),float(r[i-20:i].sum()),float(np.log1p(vol[i-20:i].mean())),float(asset)])
            groups.append(group)
    g=np.array(groups); raw=np.array(windows); c=np.array(conditions)
    if any(np.sum(g==i)<8 for i in range(3)): raise ValueError('Generative windows need ≥8 wholly contained examples per split. Increase history or reduce horizon.')
    mean,std=float(raw[g==0].mean()),max(float(raw[g==0].std()),1e-8)
    x=(raw-mean)/std
    condition_scaler,condition_audit=fit_scaler(c[g==0,:3],['log_volatility','momentum','log_volume'],title='Causal conditions (asset IDs excluded)')
    c[:,:3]=condition_scaler.transform(c[:,:3])
    audits=[condition_audit,affine_audit(raw[g==0],mean,std,'Training return windows',method='Train mean/std (global return scale)',inverse=False)]
    c=c.astype(np.float32)
    xt,ct=torch.tensor(x,dtype=torch.float32),torch.tensor(c)
    train_idx=np.flatnonzero(g==0); vi=np.flatnonzero(g==1); ti=np.flatnonzero(g==2)
    fixed=torch.randn(len(vi),z); test_z=torch.randn(len(ti),z)
    conv=p['topic']=='cdcgan'
    names=['cDCGAN'] if conv else ['CVAE','cGAN']
    table=[]; models: dict[str,nn.Module]={}; histories: dict[str,list[dict[str,Any]]]={}
    for name in names:
        net: nn.Module=CVAE(horizon,z) if name=='CVAE' else ConditionalGenerator(horizon,z,conv,len(frames))
        disc=Discriminator(horizon,conv,len(frames)) if name!='CVAE' else None
        opt=torch.optim.Adam(net.parameters(),lr=float(p.get('lr',.001)))
        dop=torch.optim.Adam(disc.parameters(),lr=float(p.get('lr',.001))) if disc else None
        best,state,stale=float('inf'),deepcopy(net.state_dict()),0
        history=[]
        for epoch in range(int(p.get('epochs',12))):
            net.train()
            for indices in torch.tensor(train_idx)[torch.randperm(len(train_idx))].split(int(p.get('batch_size',64))):
                xb,cb=xt[indices],ct[indices]
                opt.zero_grad(set_to_none=True)
                if isinstance(net,CVAE):
                    recon,kl=net(xb,cb)
                    loss=nn.functional.mse_loss(recon,xb)+.01*kl
                else:
                    assert disc is not None and dop is not None
                    dop.zero_grad(set_to_none=True)
                    fake=net(torch.randn(len(xb),z),cb)
                    dl=nn.functional.binary_cross_entropy_with_logits(disc(xb,cb),torch.ones(len(xb)))+nn.functional.binary_cross_entropy_with_logits(disc(fake.detach(),cb),torch.zeros(len(xb)))
                    dl.backward(); dop.step()
                    for param in disc.parameters(): param.requires_grad_(False)
                    loss=nn.functional.binary_cross_entropy_with_logits(disc(fake,cb),torch.ones(len(xb)))
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                if disc:
                    for param in disc.parameters(): param.requires_grad_(True)
            net.eval()
            with torch.no_grad(): fake=(net.generate(fixed,ct[vi]) if isinstance(net,CVAE) else net(fixed,ct[vi])).numpy()
            score=distribution_score(x[vi],fake)
            history.append({'epoch':epoch+1,'validation':score})
            if progress: progress({'epoch':epoch+1,'epochs':int(p.get('epochs',12)),'validation':score,'detail':name})
            if score<best: best,state,stale=score,deepcopy(net.state_dict()),0
            else: stale+=1
            if stale>=int(p.get('patience',5)): break
        net.load_state_dict(state); net.eval()
        models[name]=net; histories[name]=history
        table.append({'model':name,'validation distribution score (standardized)':best})
    selected=str(min(table,key=lambda v:float(v['validation distribution score (standardized)']))['model'])
    net=models[selected]
    with torch.no_grad(): fake=(net.generate(test_z,ct[ti]) if isinstance(net,CVAE) else net(test_z,ct[ti])).numpy()*std+mean
    real=raw[ti]
    summary={'Test Wasserstein':float(wasserstein_distance(real.ravel(),fake.ravel())),'Real mean':float(real.mean()),'Generated mean':float(fake.mean()),'Real volatility':float(real.std()),'Generated volatility':float(fake.std()),'Real 5% quantile':float(np.quantile(real,.05)),'Generated 5% quantile':float(np.quantile(fake,.05)),'Generated diversity':float(np.mean(np.std(fake,axis=0)))}
    summary.update({'Real lag-1 ACF':lag1(real),'Generated lag-1 ACF':lag1(fake),'Real lower-tail dependence':tail_dependence(real),'Generated lower-tail dependence':tail_dependence(fake)})
    if conv:
        for asset in range(len(frames)):
            for regime,label in [(-1,'Bear'),(0,'Sideways'),(1,'Bull')]:
                momentum=c[ti,1]
                mask=(c[ti,3]==asset) & (np.where(momentum<-.5,-1,np.where(momentum>.5,1,0))==regime)
                if mask.any():
                    table.append({'model':f'Asset {asset+1} / {label}','test windows':int(mask.sum()),'real std':float(real[mask].std()),'generated std':float(fake[mask].std()),'real 5% quantile':float(np.quantile(real[mask],.05)),'generated 5% quantile':float(np.quantile(fake[mask],.05))})
    rows=[{'step':i+1,'real':float(np.cumsum(real[0])[i]),'generated':float(np.cumsum(fake[0])[i])} for i in range(horizon)]
    counts,edges=np.histogram(real.ravel(),bins=24,density=True)
    generated,_=np.histogram(fake.ravel(),bins=edges,density=True)
    histogram=[{'return':float((edges[i]+edges[i+1])/2),'real':float(counts[i]),'generated':float(generated[i])} for i in range(len(counts))]
    charts=[chart('Conditional scenario · cumulative log returns',rows,['real','generated'],'step'),chart('Holdout distribution · daily log returns',histogram,['real','generated'],'return','bar'),chart('Selection score · fixed validation noise',histories[selected],['validation'],'epoch')]
    if conv:
        # Identical latent noise, same asset; only causal momentum regime changes.
        same_z=torch.randn(1,z).repeat(3,1); cc=ct[ti[:1]].repeat(3,1); cc[:,1]=torch.tensor([-1.,0.,1.])
        with torch.no_grad(): regimes=net(same_z,cc).numpy()*std+mean
        charts.append(chart('Same noise · first asset regime conditions', [{'step':i+1,**{name:float(np.cumsum(regimes[j])[i]) for j,name in enumerate(['Bear','Sideways','Bull'])}} for i in range(horizon)],['Bear','Sideways','Bull'],'step'))
    return {'selected':selected,'table':table,'metrics':summary,'charts':charts,'scaling_audits':audits,'splits':{'train':int(np.sum(g==0)),'validation':int(np.sum(g==1)),'test':int(np.sum(g==2))},'notes':['Complete windows are purged at split boundaries. Conditions use the 20 preceding sessions.','Scenarios are hypothetical; no calibrated prediction intervals. Asset embeddings are fitted jointly to connected asset frames.']}


class QNetwork(nn.Module):
    def __init__(self,d: int) -> None:
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d+1,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x: torch.Tensor) -> torch.Tensor: return self.net(x)


def double_dqn(s: Samples,p: dict[str,Any],progress: Progress | None) -> dict[str,Any]:
    seed=int(p.get('seed',42)); random.seed(seed); torch.manual_seed(seed)
    audits: list[dict[str,Any]] = []
    x=scaled(s.x[:,-1:,:],s.train,audits,s.names)[:,0,:]
    cost=float(p.get('cost_bps',5))/10000
    actions=np.array([-1,0,1])
    online=QNetwork(x.shape[1]); target=deepcopy(online).eval()
    opt=torch.optim.Adam(online.parameters(),lr=float(p.get('lr',.001)))
    replay: list[tuple[np.ndarray,int,float,np.ndarray,bool]]=[]
    updates=steps=0; best=-float('inf'); state=deepcopy(online.state_dict()); history=[]
    def step_return(t: int,prev: int,action: int,terminal: bool) -> float:
        # Existing position earns overnight gap; new position earns intraday.
        gross=(1+prev*np.expm1(s.gap[t]))*(1+action*np.expm1(s.intraday[t]))-1
        return float(gross-cost*abs(action-prev)-(cost*abs(action) if terminal else 0))
    def rollout(start: int,end: int,policy: Callable[[np.ndarray],int]) -> tuple[np.ndarray,np.ndarray]:
        prev=0; returns=[]; signals=[]
        for t in range(start,end):
            action=int(actions[policy(np.r_[x[t],prev].astype(np.float32))])
            returns.append(step_return(t,prev,action,t==end-1)); signals.append(action); prev=action
        return np.array(returns),np.array(signals)
    for epoch in range(int(p.get('epochs',12))):
        prev=0
        for t in range(s.train):
            obs=np.r_[x[t],prev].astype(np.float32)
            epsilon=max(.05,1-steps/5000)
            with torch.no_grad(): ai=random.randrange(3) if random.random()<epsilon else int(online(torch.tensor(obs)).argmax())
            action=int(actions[ai]); done=t==s.train-1
            reward=100*step_return(t,prev,action,done)
            nxt=np.r_[x[t+1] if not done else np.zeros_like(x[t]),action].astype(np.float32)
            replay.append((obs,ai,reward,nxt,done)); replay=replay[-10000:]
            steps+=1; prev=action
            if len(replay)>=128 and steps%4==0:
                batch=random.sample(replay,64)
                ob=torch.tensor(np.stack([r[0] for r in batch])); ac=torch.tensor([r[1] for r in batch]); re=torch.tensor([r[2] for r in batch],dtype=torch.float32)
                ns=torch.tensor(np.stack([r[3] for r in batch])); terminal=torch.tensor([r[4] for r in batch],dtype=torch.float32)
                with torch.no_grad():
                    chosen=online(ns).argmax(1)
                    expected=re+.95*(1-terminal)*target(ns).gather(1,chosen[:,None]).squeeze(1)
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.smooth_l1_loss(online(ob).gather(1,ac[:,None]).squeeze(1),expected)
                loss.backward(); nn.utils.clip_grad_norm_(online.parameters(),1.); opt.step(); updates+=1
                if updates%100==0: target.load_state_dict(online.state_dict())
        def greedy(obs: np.ndarray) -> int:
            with torch.no_grad(): return int(online(torch.tensor(obs)).argmax())
        val,_=rollout(s.train,s.validation,greedy)
        score=float(np.prod(1+val)-1)
        history.append({'epoch':epoch+1,'validation':score})
        if progress: progress({'epoch':epoch+1,'epochs':int(p.get('epochs',12)),'validation':score,'detail':'Double DQN · validation net ROI'})
        if score>best: best,state=score,deepcopy(online.state_dict())
    online.load_state_dict(state); online.eval()
    def greedy(obs: np.ndarray) -> int:
        with torch.no_grad(): return int(online(torch.tensor(obs)).argmax())
    # Tabular Q-learning on train only, discretized train-scaled return and position.
    q=np.zeros((5,3,3)); bins=np.array([-1.,-.25,.25,1.])
    for _ in range(int(p.get('epochs',12))):
        prev=0
        for t in range(s.train):
            bucket=int(np.digitize(x[t,0],bins)); pi=prev+1
            ai=random.randrange(3) if random.random()<.1 else int(q[bucket,pi].argmax())
            action=int(actions[ai]); done=t==s.train-1
            future=0 if done else float(q[int(np.digitize(x[t+1,0],bins)),action+1].max())
            q[bucket,pi,ai]+=.1*(100*step_return(t,prev,action,done)+.95*future-q[bucket,pi,ai]); prev=action
    policies={'Double DQN':greedy,'Q-learning':lambda obs:int(q[int(np.digitize(obs[0],bins)),int(obs[-1])+1].argmax()),'Always long':lambda _:2,'Cash':lambda _:1}
    table=[]; rows=[{'date':d} for d in s.dates[s.validation:]]; summary={}
    for name,policy in policies.items():
        daily,signals=rollout(s.validation,len(x),policy)
        equity=np.cumprod(1+daily); vol=float(daily.std(ddof=1))
        score={'model':name,'test net ROI':float(equity[-1]-1),'Sharpe':float(daily.mean()/vol*np.sqrt(252)) if vol else 0.,'max drawdown':float(np.min(equity/np.maximum.accumulate(np.r_[1.,equity])[1:]-1)),'turnover':float(np.abs(np.diff(np.r_[0,signals,0])).sum())}
        table.append(score)
        if name=='Double DQN': summary={k:v for k,v in score.items() if isinstance(v,(float,int))}
        for row,v in zip(rows,equity): row[name]=float(v-1)
    return {'selected':'Double DQN','scaling_audits':audits,'table':table,'metrics':summary,'charts':[chart('Greedy holdout · gap + intraday − turnover costs',rows,list(policies)),chart('Checkpoint selection · validation net ROI',history,['validation'],'epoch')],'notes':['Replay contains training transitions only. Terminal targets omit bootstrap value. Validation and test use greedy actions without updates.']}


def study(frame: pd.DataFrame,p: dict[str,Any],progress: Progress | None = None, additional: list[pd.DataFrame] | None = None) -> dict[str,Any]:
    topic=str(p.get('topic','feedforward'))
    if topic not in TOPICS: raise ValueError('Unknown experiment topic.')
    if additional and topic!='cdcgan': raise ValueError('Optional asset_b/asset_c pins are supported by cdcgan only.')
    for key,lo,hi,default in [('epochs',1,300,12),('batch_size',2,1024,64),('patience',1,100,5),('trials',1,30,6),('latent_dim',1,32,3)]:
        if not lo<=int(p.get(key,default))<=hi: raise ValueError(f'{key} must be {lo}–{hi}.')
    if not 1e-6<=float(p.get('lr',.001))<=.1 or not 0<=float(p.get('cost_bps',5))<=100: raise ValueError('Learning rate .000001–.1; costs 0–100 bps.')
    if topic in ('generative','cdcgan'):
        # Validate shared split controls before constructing complete future windows.
        s=samples(frame,p)
        result=generative(frame,p,progress,additional or [])
    else:
        s=samples(frame,p)
        if topic=='autoencoders': result=autoencoders(s,p,progress)
        elif topic=='double_dqn': result=double_dqn(s,p,progress)
        else: result=supervised(s,p,progress)
    notes=['Inputs end at close t. Supervised labels: 10000 × log(Close[t+1]/Open[t+1]). Scalers fit training only.',*result.pop('notes',[])]
    return {'study':True,'topic':topic,'features':s.names,'splits':{'train':s.train,'validation':s.validation-s.train,'test':len(s.y)-s.validation},'notes':notes,**result}
