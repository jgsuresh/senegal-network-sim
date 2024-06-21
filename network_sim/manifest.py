import os

# emod_infection_summary_filepath = os.path.join(r"C:\Users\joshsu\OneDrive - Bill & Melinda Gates Foundation\Code\senegal-network-sim\network_sim", "emod_infection_summary.csv")
# emod_infection_summary_filepath = os.path.join(".", "emod_infection_summary.csv")

# Join current directory with the filename
# current_directory = os.path.dirname(os.path.realpath(__file__))
script_directory = os.path.dirname(os.path.abspath(__file__))
emod_infection_summary_filepath = os.path.join(script_directory, "emod_infection_summary.csv")