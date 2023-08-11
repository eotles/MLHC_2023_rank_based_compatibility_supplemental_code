import numpy as np
from sklearn import utils



def btc_score(y, y_hat_o, y_hat_u):
    '''
    Backwards Trust Compatibility
    
    Determines the impact of going from an original model (f_o) to an updated
    model (f_u) in terms of correct classifications. Uses their respective
    labels (y_hat_o and y_hat_o).
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels.
    
    y_hat_o : array-like of shape (n_samples,)
        Labels produced by original model.


    y_hat_u : array-like of shape (n_samples,)
        Labels produced by updated model.
        
    Returns
    -------
    btc : float
    
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
         
    '''
    
    utils.check_consistent_length(y, y_hat_o, y_hat_u)
    
    g = np.vstack([y, y_hat_o, y_hat_u]).T
    trust_candidates = g[g[:,0] == g[:,1]]
    
    if trust_candidates.shape[0] <= 0:
        raise ValueError("No y_hat_o correct predictions present. Backwards "\
                         "Trust Compatibility is not defined in that case.")

    trust_matches = trust_candidates[trust_candidates[:,1] == trust_candidates[:,2]]

    btc = trust_matches.shape[0]/trust_candidates.shape[0]
    return(btc)



def bec_score(y, y_hat_o, y_hat_u):
    '''
    Backwards Error Compatibility
    
    Determines the impact of going from an original model (f_o) to an updated
    model (f_u) in terms of incorrect classifications (i.e., errors). Uses
    their respective labels (y_hat_o and y_hat_o).
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels.
    
    y_hat_o : array-like of shape (n_samples,)
        Labels produced by original model.


    y_hat_u : array-like of shape (n_samples,)
        Labels produced by updated model.
        
    Returns
    -------
    bec : float
    
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
         
    '''
    
    utils.check_consistent_length(y, y_hat_o, y_hat_u)
    
    g = np.vstack([y, y_hat_o, y_hat_u]).T
    error_candidates = g[g[:,0] != g[:,1]]
    
    if error_candidates.shape[0] <= 0:
        raise ValueError("No y_hat_o errors present. Backwards Error "\
                          "Compatibility is not defined in that case.")
    
    error_matches = error_candidates[error_candidates[:,1] == error_candidates[:,2]]

    bec = error_matches.shape[0]/error_candidates.shape[0]
    return(bec)


_check_array_kwargs = {'ensure_2d':False, 'ensure_min_samples':2}


def _make_pair_matrix(y, p_hat, op=np.greater):
    row_instances = p_hat[y==1]
    col_instances = p_hat[y==0][:, None]
    pm = op(row_instances, col_instances)
    return(pm)



def rbc_score(y, p_hat_o, p_hat_u):
    '''
    Rank-based Compatibility
    
    Determines the impact of going from an original model (f_o) to an updated
    model (f_u) in terms of correct instance rankings. Compares instance-pairs
    (consisting of a 0-labelled instance and 1-labelled instance) in terms of
    the ranking produced by risk estiamates from the original model (p_hat_o)
    and the risk estimates from the updated model (p_hat_u).
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels.
    
    p_hat_o : array-like of shape (n_samples,)
        Risk estimates produced by the original model.


    p_hat_u : array-like of shape (n_samples,)
        Risk estimates produced by the updated model.
        
    Returns
    -------
    rbc : float
    
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
         
    '''

    y = utils.check_array(y, **_check_array_kwargs)
    p_hat_o = utils.check_array(p_hat_o, **_check_array_kwargs)
    p_hat_u = utils.check_array(p_hat_u, **_check_array_kwargs)

    utils.check_consistent_length(y, p_hat_o, p_hat_u)
    
    pm_o = _make_pair_matrix(y, p_hat_o)
    pm_u = _make_pair_matrix(y, p_hat_u)
    
    numerator = np.sum(pm_o*pm_u)
    denominator = np.sum(pm_o)
    
    if denominator == 0:
        raise ValueError("No p_hat_o correct rankings present. Rank-Based "\
                         "Compatibility is not defined in that case.")
    
    return numerator/denominator




def irbc_score(y, p_hat_o, p_hat_u):
    '''
    Incorrect Rank-based Compatibility
    
    Determines the impact of going from an original model (f_o) to an updated
    model (f_u) in terms of incorrect instance rankings. Compares instance-pairs
    (consisting of a 0-labelled instance and 1-labelled instance) in terms of
    the ranking produced by risk estiamates from the original model (p_hat_o)
    and the risk estimates from the updated model (p_hat_u).
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels.
    
    p_hat_o : array-like of shape (n_samples,)
        Risk estimates produced by the original model.


    p_hat_u : array-like of shape (n_samples,)
        Risk estimates produced by the updated model.
        
    Returns
    -------
    irbc : float
    
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
         
    '''


    y = utils.check_array(y, **_check_array_kwargs)
    p_hat_o = utils.check_array(p_hat_o, **_check_array_kwargs)
    p_hat_u = utils.check_array(p_hat_u, **_check_array_kwargs)

    utils.check_consistent_length(y, p_hat_o, p_hat_u)

    pm_o = _make_pair_matrix(y, p_hat_o, op=np.less)
    pm_u = _make_pair_matrix(y, p_hat_u, op=np.less)

    numerator = np.sum(pm_o*pm_u)
    denominator = np.sum(pm_o)
    
    if denominator == 0:
        raise ValueError("No p_hat_o incorrect rankings present. Incorrect " \
                         "Rank-Based Compatibility is not defined in that case.")
    
    return numerator/denominator
