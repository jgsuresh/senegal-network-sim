# senegal-network-sim
Simplified generative model for malaria transmission networks in Senegal

# Work setup
1. Learn simple functional forms to describe EMOD-like transmission
   1. Set up Maka-like sims (without interventions), and sweep these over transmission intensity
   2. Identify the transmission points to focus on (Wes: 300 cases/1000/yr, 100, 10, <10)
   3. For each of these, bin the population into age bins
   4. Look at age distributions of infection duration, infectiousness, and susceptibility in these bins
   5. Try to derive simple functional forms for these distributions
2. Create new, simple transmission model that has EMOD-like descriptions
   1. Human population with infection duration, infectiousness, and susceptibility drawn from distributions
   2. Simplified vector layer that is parametrized version of EMOD vector populations (only need to keep track of vectors that actually transmit)
   3. No limit on complexity of infection in humans
3. Identify scale above which we can use model to trust population-level genetics signals and not be too concerned about the individual level heterogeneity
4. Use model to explore potential networks that have similar incidence but different genetic signals