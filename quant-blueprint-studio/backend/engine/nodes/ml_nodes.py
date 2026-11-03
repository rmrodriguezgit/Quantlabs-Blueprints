"""Classical regressors with one chronological TR/VA/TE selection protocol."""
from typing import Any, Callable
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .stats_nodes import PreparedData
from .dl_nodes import ModelResult

ALGORITHMS = ('Mean baseline','Ridge','ElasticNet','Random Forest','Histogram Boosting','SVR','KNN')


def train_ml(data: PreparedData, p: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None) -> tuple[ModelResult, dict[str, Any]]:
    algorithm = str(p.get('algorithm','Compare'))
    mode = str(p.get('window_mode','last'))
    if algorithm not in (*ALGORITHMS,'Compare') or mode not in ('last','flatten'):
        raise ValueError('Choose a supported ML algorithm and last/flatten window mode.')
    trees, depth = int(p.get('estimators',100)), int(p.get('max_depth',8))
    alpha, seed = float(p.get('alpha',1)), int(p.get('seed',42))
    neighbors = int(p.get('neighbors',5))
    if not 10<=trees<=500 or not 1<=depth<=32 or not np.isfinite(alpha) or not 0<alpha<=1e6 or not 1<=neighbors<=100:
        raise ValueError('ML controls: trees 10–500, depth 1–32, alpha >0–1e6, neighbors 1–100.')
    if max(len(data.x_train),len(data.x_val),len(data.x_test))>20000:
        raise ValueError('ML node supports at most 20000 windows per split; shorten history.')
    def matrix(x: np.ndarray) -> np.ndarray:
        result=x[:,-1,:] if mode=='last' else x.reshape(len(x),-1)
        if not np.isfinite(result).all():raise ValueError('ML inputs must be finite.')
        return result
    x,va,te=[matrix(v) for v in (data.x_train,data.x_val,data.x_test)]
    if x.shape[1]>4096:raise ValueError('ML supports at most 4096 flattened features; reduce lookback or columns.')
    if algorithm in ('SVR','Compare') and len(x)>5000:raise ValueError('SVR/Compare supports at most 5000 training windows; shorten history or select a different algorithm.')
    if neighbors>len(x):raise ValueError('KNN neighbors must not exceed training rows.')
    center,scale=float(data.y_train.mean()),max(float(data.y_train.std()),1e-8)
    y=(data.y_train-center)/scale
    factories={
        'Mean baseline':lambda:DummyRegressor(strategy='mean'),
        'Ridge':lambda:Ridge(alpha=alpha),
        'ElasticNet':lambda:ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=5000,random_state=seed),
        'Random Forest':lambda:RandomForestRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=3,random_state=seed,n_jobs=1),
        'Histogram Boosting':lambda:HistGradientBoostingRegressor(max_iter=trees,max_depth=depth,early_stopping=False,random_state=seed),
        'SVR':lambda:SVR(C=alpha,epsilon=.1),
        'KNN':lambda:KNeighborsRegressor(n_neighbors=neighbors,weights='distance'),
    }
    candidates=ALGORITHMS if algorithm=='Compare' else (algorithm,)
    rows=[];best=float('inf');winner=None;selected=''
    for i,name in enumerate(candidates):
        model=factories[name]();model.fit(x,y)
        prediction=model.predict(va)*scale+center
        error=float(mean_squared_error(data.y_val,prediction))
        if not np.isfinite(error):raise ValueError('ML validation produced non-finite error.')
        rows.append({'algorithm':name,'validation MSE':error,'validation RMSE':float(np.sqrt(error))})
        if error<best:best,winner,selected=error,model,name
        if progress:progress({'detail':f'{name}: validation MSE {error:.6g} · {i+1}/{len(candidates)}'})
    assert winner is not None
    prediction=np.asarray(winner.predict(te)*scale+center,dtype=np.float64)
    mse=float(mean_squared_error(data.y_test,prediction))
    metrics={'MSE':mse,'RMSE':float(np.sqrt(mse)),'MAE':float(mean_absolute_error(data.y_test,prediction)),'R2':float(r2_score(data.y_test,prediction))}
    result=ModelResult(data,prediction,metrics,[])
    report={'study':True,'topic':'classical_ml','selected':selected,'scope':'holdout / out of sample','features':data.feature_names,'splits':{'train':len(x),'validation':len(va),'test':len(te)},'metrics':metrics,'table':rows,'charts':[{'title':'Model selection · validation only','x_key':'algorithm','series':['validation RMSE'],'rows':rows,'chart_type':'bar'}],'notes':['Candidates fit TR only. VA selects one winner; test is evaluated once after selection. No refit on VA.','X scaling comes from Series Preparation fitted on TR; target standardization also fits TR only. Scores and predictions return to original target units.',f'Window mode: {mode}. All candidates share the same input windows and temporal cuts.']}
    return result,report
