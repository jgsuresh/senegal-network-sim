import numpy as np

from jsuresh_helpers.interventions.importations import import_infections_through_outbreak
from jsuresh_helpers.running_emodpy import build_standard_campaign_object
from emod_sims import manifest


def constant_annual_importation(campaign, total_importations_per_year):

    if total_importations_per_year <= 365:
        days_between_importations = int(np.round(365/total_importations_per_year))
        num_infections = 1
    else:
        days_between_importations = 1
        num_infections = int(np.round(total_importations_per_year/365))

    import_infections_through_outbreak(campaign,
                                       days_between_outbreaks=days_between_importations,
                                       start_day=1,
                                       num_infections=num_infections)
    return campaign

def add_custom_events(campaign):
    # Add to custom events (used to do this by directly editing config.parameters.Custom_Individual_Events
    campaign.get_send_trigger("InfectionDropped", old=True)


def build_full_campaign(total_importations_per_year=25):
    campaign = build_standard_campaign_object(manifest=manifest)
    constant_annual_importation(campaign, total_importations_per_year=total_importations_per_year)
    add_custom_events(campaign)
    return campaign
