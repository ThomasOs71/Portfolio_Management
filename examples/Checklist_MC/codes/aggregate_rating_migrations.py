#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def aggregate_rating_migrations(db, ratings_param, t_start, t_end):
    
    """This function aggregates rating migrations over a specified period, calculating the number of obligors 
    and transitions between ratings at each time point, and providing summaries of these transitions 
    and the final ratings for each obligor within the given analysis period.
        
    Parameters
    ----------
        db : dataframe
        ratings_param : array, shape (c_bar,)
        t_start : date
        t_end : date

    Returns
    -------
        dates : array, shape (t_bar,)
        n : array, shape (t_bar, c_bar)
        n_trans : array, shape (t_bar, c_bar, c_bar)
        m_trans : array, shape (t_bar, c_bar, c_bar)
        n_tot : array, shape (t_bar,)
        rating_tend : dict, size n_bar
    """

    # extract list of obligors
    obligors = db.issuer.unique()

    # extract transition dates
    dates_tend = db.sort_values(["date"]).date.unique().astype('datetime64[ns]')
    dates_tend = dates_tend[dates_tend <= t_end]

    # extract transition dates within analysis period [t_start, t_end]
    dates = dates_tend[dates_tend >= t_start]

    # determine number of obligors and transitions at each time

    # initialize point in time (PIT) rating dataframe and counter dictionaries
    pit_rating = pd.DataFrame(index=obligors, columns=['rating', 'prv_rating'])
    n_dict = {}
    m_trans_dict = {}
    # for each transition date
    for date in dates:
        # get rating changes
        idx = (db.date == date)
        rating_changes = db.loc[idx, ('issuer', 'rating')].copy()
        rating_changes.set_index('issuer', inplace=True)
        
        # update previous rating column in PIT dataframe
        pit_rating.prv_rating = pit_rating.rating
        # update current rating column with any changes
        pit_rating.update(rating_changes)
        
        # get counts for each date within period of interest
        if date >= t_start:
            # count number of obligors in each rating class
            n_dict[date] = pit_rating.rating.value_counts().to_dict()
            # select obligors with transition
            idx_change = (pit_rating.prv_rating != pit_rating.rating)
            # count transitions between classes
            m_trans_dict[date] = pit_rating.loc[idx_change].groupby(['prv_rating', 'rating']).size().to_dict()

    # number of obligors in each rating at transition dates
    n_df = pd.DataFrame.from_dict(n_dict, orient='index')
    n_df = n_df.reindex(index=dates, columns=ratings_param).fillna(0)
    # final output as numpy array
    n = n_df.values

    # number of transitions between ratings at transition dates
    m_trans_df = pd.DataFrame.from_dict(m_trans_dict, orient='index')

    from_to_index = pd.MultiIndex.from_product([ratings_param, ratings_param], names=['from', 'to'])
    m_trans_df = m_trans_df.reindex(index=dates, columns=from_to_index).fillna(0)
    # final output as numpy array
    m_trans_list = []
    for from_rating in ratings_param:
        m_trans_list.append(m_trans_df.loc[:, from_rating].values)
    m_trans = np.array(m_trans_list)
    m_trans = np.moveaxis(m_trans, 0, 1)

    # calculate cumulative number of transitions between ratings
    n_trans_df = pd.DataFrame(index=dates, columns=from_to_index)
    for col in n_trans_df:
        n_trans_df.loc[:, col] = m_trans_df.loc[:, col].cumsum()
    n_trans_df = n_trans_df.reindex(index=dates, columns=from_to_index)
    # final output as numpy array
    n_trans_list = []
    for from_rating in ratings_param:
        n_trans_list.append(n_trans_df.loc[:, from_rating].values)
    n_trans = np.array(n_trans_list)
    n_trans = np.moveaxis(n_trans, 0, 1)

    # total number of transitions up to time t
    n_tot = n_trans_df.sum(axis=1).values

    # final rating for each obligor in analysis period [t_start, t_end]
    rating_tend = pit_rating.rating.to_dict()

    return dates, n, n_trans, m_trans, n_tot, rating_tend
