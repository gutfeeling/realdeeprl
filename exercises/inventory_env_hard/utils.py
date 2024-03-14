import numpy as np
from scipy.stats import poisson


def classical_baseline_action(obs):
    lead_time = obs.shape[0] - 5
    mean_daily_demand = obs[lead_time]
    selling_price = obs[lead_time + 1]
    buying_price = obs[lead_time + 2]
    daily_holding_cost_per_unit = obs[lead_time + 3]
    goodwill_penalty_per_unit = obs[lead_time + 4]
    critical_ratio = (
            (selling_price - buying_price + goodwill_penalty_per_unit) /
            (selling_price - buying_price + daily_holding_cost_per_unit + goodwill_penalty_per_unit)
    )
    z_star = poisson.ppf(critical_ratio, lead_time * mean_daily_demand)
    return np.array([max(0, z_star - obs[:lead_time].sum())])
