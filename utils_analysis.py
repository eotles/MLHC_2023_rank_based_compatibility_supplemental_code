'''
EXO
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn import base, linear_model, model_selection
from sklearn import metrics
from tqdm import notebook



percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]


cmap = plt.cm.rainbow
norm = mpl.colors.Normalize(vmin=0, vmax=1)

def ci_plot(df, ax=None, x='alpha', y_center='center', y_lb='lb', y_ub='ub',
            y_dash=None,
            x_eps=0,
            color_value=None,
            color_cmap=plt.cm.rainbow,
            color_norm=mpl.colors.Normalize(vmin=0, vmax=1)):
    
    if ax is None:
        ax = plt.gca()
        
    x = df[x].values + x_eps
    y = df[y_center].values
    y_le = y - df[y_lb].values
    y_ue = df[y_ub].values - y
    
    if color_value is not None:
        if color_value in df.columns:
            cv = df[color_value].values
            c = [color_cmap(color_norm(_cv)) for _cv in cv]
        else:
            c = [color_value for _ in x]
    else:
        c = ['lightgray' for _ in x]
    
    for _x, _y, _y_le, _y_ue, _c in zip(x, y, y_le, y_ue, c):
        ax.errorbar(_x, _y, yerr=[[_y_le], [_y_ue]], color=_c, fmt='o')
    
    if y_dash is not None:
        y = df[y_dash].values
        for _x, _y, _c in zip(x, y, c):
            ax.scatter(_x, _y, marker='_', s=100, color=_c)
        
    return ax


def selection_tradeoff(res_df, sx_name='ΔAUROC(ue)', sy_name='RBC(ue)',
                       alphas = np.expand_dims(np.linspace(0,1, 11), axis=1).T):

    _ = res_df.copy()
    _ = _.reset_index(drop=True)
    o = _[[sx_name]].values @ alphas + _[[sy_name]].values @ (1-alphas)
    alpha_columns = ['alpha={:0.1f}'.format(a) for a in alphas[0]]
    o = pd.DataFrame(o, columns=alpha_columns)
    _ = _.join(o)
    
    _idxmax = _.groupby(by='f_o_rep')[alpha_columns].idxmax(axis=0)
    _idxmax = np.unique(_idxmax)
    _ = _.loc[_idxmax]
    
    return _
    
    
def get_ci_df(comparison_res_df, column='ΔRBC(e)', ax=None, lb=0.025, center=0.5, ub=0.975, x_eps=0, color_value='f_u_alpha',
             display_df=True):
    percentiles = [lb, center, ub]
    _ = comparison_res_df.reset_index()
    _ = _[['f_u_alpha', column]]
    _['f_u_alpha'] = _['f_u_alpha'].astype(float)
    _ = _.groupby(by='f_u_alpha')
    _ = _.describe(percentiles=percentiles)
    _ = _.reset_index()
    
    if display_df:
        display(_.set_index('f_u_alpha').T)
    
    str_lb, str_center, str_ub = pd.io.formats.format.format_percentiles(percentiles)
    
    ci_plot(_, ax=ax, x='f_u_alpha', y_center=(column, str_center),
           y_lb=(column, str_lb),
           y_ub=(column, str_ub),
           y_dash=(column, 'mean'),
           color_value=color_value,
               x_eps=x_eps
          )

