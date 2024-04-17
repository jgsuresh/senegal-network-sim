from emodpy_malaria.reporters.builtin import add_malaria_cotransmission_report, add_malaria_summary_report, \
    add_report_infection_stats_malaria

from emod_sims import manifest


def add_testing_reports(emod_task):
    start_day = 365*55
    add_malaria_summary_report(emod_task, manifest=manifest,
                               start_day=start_day, age_bins=[0, 5, 15, 100])
    add_report_infection_stats_malaria(emod_task, manifest, start_day=start_day)

    # Limit stdout
    emod_task.config.parameters["logLevel_default"] = "WARNING"
    emod_task.config.parameters["logLevel_JsonConfigurable"] = "WARNING"
    emod_task.config.parameters["Enable_Log_Throttling"] = 1
