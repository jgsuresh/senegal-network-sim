Trying emod-like biting.

Since emod distribution has mean of 3.7 bites per infectious mosquito, I put in a survival probability of 1/3.7 to get back to a mean of 1. This is a bit of a hack, but it should work for now.


Notes:
- Increasing number of mean bites per mosquito and compensating by decreasing survival probability actually decreases the total number of 

scenario a: 100 bites in 100 days, 1 gets infected, this one has 100% chance to infect anyone --> expected 1 onward
scenario b: 100 bites in 100 days, 1 gets infected, this has a 1/3.7 chance to infect anyone, but then delivers an average of 3.7 bites --> expected 1 onward.

However, scenario b is a bit slower since the mosquito has to deliver its ~3.7 bites which takes time (~11 days). In other words, the human population is seeing a slightly lower rate of infectious bite challenges. In scenario a, it sees ~1 infectious bite in ~50+12 days. In scenario b, it sees ~1 infectious bite in ~50+12+11 days.
The biting rate in scenario b should be a bit higher than in scenario a to compensate for this. I will try 1*(50+12+11)/(50+12) = 1.18 bites per day

