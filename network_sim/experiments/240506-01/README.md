Experimenting with adding in human ages. These affect:
	- biting risk - surface area, so lower for kids
	- average infectiousness: higher in kids

Since mean of surface-area-based-risk is 0.7528, we need to increase biting by this factor to keep the same overall biting risk for comparison to older runs. This means going from 1 daily bite to 1.328 daily bites.

For now, keep age_modifies_infectiousness = False and just see what modified biting rate does.