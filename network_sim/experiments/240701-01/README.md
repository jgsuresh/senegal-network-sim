Testing improved transmission pipeline:

- infectiousness = 1-np.prod(1-infectiousness_of_strains).
- infectiousness_of_strains is diluted for coinfected strains to give the correct total infectiousness.
- When vector is picking up strains, back out gametocyte NUMBERS in bloodmeal, then do poisson draw based on blood meal of size 1. Use these numbers without replacement for oocyst formation.