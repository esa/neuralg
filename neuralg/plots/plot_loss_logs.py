import numpy as np
import matplotlib.pyplot as plt


def plot_loss_logs(run_cfg, save_path=None):
    """ Plot dynamics from training
    Args:
        run_cfg (DotMap): Run configuration post model training
        save_path (_type_, optional): Folder to save plots in. If not specified, they will not be saved.
    """
    ms = run_cfg.matrix_sizes
    no_plots = len(ms)
    for j in range(no_plots):
        d = ms[j]
        fig = plt.figure(figsize=(14, 6), dpi=150)
        fig.patch.set_facecolor("white")
        k = run_cfg[d].run_params.iterations
        epoch = run_cfg[d].run_params.epoch
        ax = fig.add_subplot()
        p1 = ax.plot(run_cfg[d].results.loss_log, label="Training loss")
        p2 = ax.plot(
            run_cfg[d].results.weighted_average_log,
            label="Weighted average",
            color="pink",
        )
        p3 = ax.plot(
            np.linspace(0, k * epoch, len(run_cfg[d].results.eval_loss_log)),
            run_cfg[d].results.eval_loss_log,
            label="Evaluation loss",
            color="red",
        )
        ax.set_yscale("log")
        ax.set_title("Loss logs, {}x{}".format(d, d), fontsize=18)
        ax.legend(fontsize=14)
        ax.set_xlabel("Iteration", fontsize=18)
        ax.set_ylabel("Loss", fontsize=16)
        if save_path is not None:
            plt.savefig(
                save_path + "/losses{}.png".format(run_cfg.matrix_sizes[j]), dpi=150
            )
