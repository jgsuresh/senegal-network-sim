from emodpy_malaria.reporters.builtin import add_report_event_counter, add_malaria_summary_report, \
    add_report_infection_stats_malaria, add_spatial_report_malaria_filtered, add_report_intervention_pop_avg, \
    add_report_node_demographics_malaria_genetics, add_malaria_cotransmission_report

from emod_sims import manifest


def add_scenario_reports(emod_task, include_inset=True, include_bednet_events_in_counter=False):
    # add_malaria_summary_report(emod_task, manifest=manifest, age_bins=summary_report_age_bins, reporting_interval=365)

    events_to_count = [
        "Received_Treatment",
        "Received_Test",
        "Received_Campaign_Drugs",
        "Received_SMC",
        "Received_Ivermectin",
        "Received_Primaquine"
    ]
    if include_bednet_events_in_counter:
        events_to_count += ["Bednet_Discarded", "Bednet_Got_New_One", "Bednet_Using"]

    add_report_event_counter(emod_task, manifest=manifest, event_trigger_list=events_to_count)
    # add_report_event_counter(emod_task, manifest=manifest)

    if include_inset:
        emod_task.config.parameters.Enable_Default_Reporting = 1
    else:
        emod_task.config.parameters.Enable_Default_Reporting = 0

    # Limit stdout
    emod_task.config.parameters["logLevel_default"] = "WARNING"
    emod_task.config.parameters["logLevel_JsonConfigurable"] = "WARNING"
    emod_task.config.parameters["Enable_Log_Throttling"] = 1

def add_standard_cotransmission_reports(emod_task):
    emod_task.config.parameters.Enable_Demographics_Reporting = 1

    add_malaria_cotransmission_report(emod_task, manifest, max_age_years=10000)
    add_report_infection_stats_malaria(emod_task, manifest,
                                       reporting_interval=30,
                                       include_hepatocyte=False)


def add_standard_fpg_reports(emod_task):
    emod_task.config.parameters.Enable_Demographics_Reporting = 1

    # add_report_node_demographics_malaria_genetics(emod_task, manifest, drug_resistant_strings=["A","C","G","T"], stratify_by_gender=0)
    # add_report_node_demographics_malaria_genetics(emod_task, manifest, drug_resistant_strings=["AA","AC","CA","CC"], stratify_by_gender=0)
    add_report_node_demographics_malaria_genetics(emod_task, manifest,
                                                  drug_resistant_strings=["AA","AC","CA","CC"],
                                                  drug_resistant_and_hrp_statistic_type="NUM_INFECTIONS",
                                                  stratify_by_gender=0)




def add_burnin_reports(emod_task):
    emod_task.config.parameters.Enable_Demographics_Reporting = 0

    # Limit stdout
    emod_task.config.parameters["logLevel_default"] = "WARNING"
    emod_task.config.parameters["logLevel_JsonConfigurable"] = "WARNING"
    emod_task.config.parameters["Enable_Log_Throttling"] = 1

def add_testing_reports(emod_task):
    add_standard_cotransmission_reports(emod_task)

    # Limit stdout
    emod_task.config.parameters["logLevel_default"] = "WARNING"
    emod_task.config.parameters["logLevel_JsonConfigurable"] = "WARNING"
    emod_task.config.parameters["Enable_Log_Throttling"] = 1

    # events_to_count = [
    #     "Received_Treatment",
    #     "Received_Test",
    #     "Received_Campaign_Drugs",
    #     "Received_SMC",
    #     "Received_Ivermectin",
    #     "Received_Primaquine"
    # ]
    # if include_bednet_events_in_counter:
    #     events_to_count += ["Bednet_Discarded", "Bednet_Got_New_One", "Bednet_Using"]

    events_to_count = ["Screened_For_Fever",
                       "Has_Fever",
                       "Screened_by_RDT",
                       "Received_PECADOM_drugs",
                       "Received_Treatment",
                       "InfectionDropped"]
                       # "Received_PECADOM_drugs",
                       # "Did_Not_Receive_PECADOM_drugs"]
                       # "Febrile_and_RDT_positive",
                       # "Febrile_and_RDT_negative"]

    add_report_event_counter(emod_task, manifest=manifest, event_trigger_list=events_to_count)
    # add_scenario_reports(emod_task, include_summary=True, include_inset=True, include_bednet_events_in_counter=True)

    # for campaign visualizer
    add_spatial_report_malaria_filtered(emod_task, manifest=manifest, spatial_output_channels=["Population"])
    add_report_intervention_pop_avg(emod_task, manifest=manifest)

