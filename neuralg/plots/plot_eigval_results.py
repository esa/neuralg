import matplotlib.pyplot as plt


def plot_eigval_results(run_cfg, save_path=None):
    """ Plot accuracy results as a function of tolerance for evaluated post-training run configuration

    Args:
        run_cfg (DotMap): Evaluated post-training configuration
        save_path (str, optional): Path to folder for saving plot. Defaults to None
    """
    ms = run_cfg.matrix_sizes
    no_plots = len(ms)
    for j in range(no_plots):
        d = ms[j]
        fig = plt.figure(figsize=(14, 6), dpi=150)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot()
        for i, k in enumerate(run_cfg[d].test_results):
            ax.plot(
                run_cfg[d].test_cfg.tolerances,
                run_cfg[d].test_results[k],
                label="{}".format(k),
            )

        ax.set_title("Test accuracy versus tolerance, {}x{}".format(d, d), fontsize=18)
        ax.legend(fontsize=14)
        ax.set_xlabel("$\\tau $", fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=16)
        if save_path is not None:
            plt.savefig(save_path + "/test_accuracy{}.png".format(d), dpi=150)
