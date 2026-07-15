import matplotlib.pyplot as plt
import optuna

from optuna.storages import JournalStorage, JournalFileStorage

from optuna.visualization import plot_pareto_front, plot_param_importances

storage = JournalStorage(JournalFileStorage("/export/data/vgiusepp/HydraBFlow/data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow/tuning/stream_ibata_grid_study.log"))

study = optuna.load_study(
    study_name='stream_ibata_grid_study',
    storage=storage,
)
fig = plot_pareto_front(study, target_names=["RMSE", "Calibration Error"])
fig.show()

fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="RMSE")
    # plt.savefig('plots_optuna/param_importances_rmse.png')
fig.show()

fig = plot_param_importances(study, target=lambda t: t.values[1], target_name="Calibration Error")
# plt.savefig('plots_optuna/param_importances_calibration_error.png')
fig.show()