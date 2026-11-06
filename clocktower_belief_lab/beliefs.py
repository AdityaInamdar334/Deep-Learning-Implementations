"""Synthetic mechanics exercise. Probabilities are teaching assumptions, not measured game rates."""
from dataclasses import dataclass

@dataclass(frozen=True)
class World:
    target_is_demon: bool
    slayer_poisoned: bool
    prior: float


def after_survival(worlds):
    if not worlds or any(w.prior < 0 for w in worlds):
        raise ValueError('World weights must be nonnegative and nonempty')
    # Assumes a real, living Slayer with an unused ability and a living target.
    # Poisoning is the only impairment represented in this deliberately tiny model.
    survivors=[w for w in worlds if w.slayer_poisoned or not w.target_is_demon]
    mass=sum(w.prior for w in survivors)
    if mass <= 0: raise ValueError('Observation impossible under these worlds')
    return sum(w.prior for w in survivors if w.target_is_demon)/mass


def worlds(prior_demon, poison_if_demon, poison_if_not):
    if any(not 0 <= p <= 1 for p in (prior_demon,poison_if_demon,poison_if_not)):
        raise ValueError('Probabilities must be in [0,1]')
    return [World(True,False,prior_demon*(1-poison_if_demon)),
            World(True,True,prior_demon*poison_if_demon),
            World(False,False,(1-prior_demon)*(1-poison_if_not)),
            World(False,True,(1-prior_demon)*poison_if_not)]


def categorical_brier(probabilities, current_demon):
    """Sum across candidates, not per-candidate average. Lower is better."""
    if current_demon not in probabilities:raise ValueError('Missing truth label')
    if any(not 0 <= p <= 1 for p in probabilities.values()):raise ValueError('Invalid probability')
    if abs(sum(probabilities.values())-1)>1e-8:raise ValueError('Probabilities must sum to one')
    return sum((p-int(name==current_demon))**2 for name,p in probabilities.items())

if __name__=='__main__':
    for label,a,b in [('No poison',0,0),('Independent poison',.1,.1),('Targeted poison',.8,.1)]:
        print(f'{label}: Demon probability after survival = {after_survival(worlds(.3,a,b)):.1%}')
    print('All scenarios begin at 30%; all poison probabilities are hypothetical.')
