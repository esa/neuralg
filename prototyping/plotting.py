import numpy as np
import matplotlib as plt
import torch
def plot_loss_logs(ax,loss_log,weighted_average_log,eval_loss_log,k):
    ax.plot(loss_log, label = "Training loss")
    ax.plot(weighted_average_log, label = "Weighted average")
    ax.plot(np.linspace(0,k,len(eval_loss_log)), eval_loss_log, label= "Evaluation loss")
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_title('Loss logs', fontsize = 18)
    ax.legend(fontsize = 14)
    ax.set_xlabel('Iteration', fontsize=18)
    ax.set_ylabel('Loss', fontsize=16)

def error_histogram(ax,test_errors):
    """
    Args:
        test_errors (dict): dict from model evaluation function.
    """
    m = test_errors
    ax.hist(np.log10(m["errors"]), bins = 40, edgecolor='black', alpha=0.65)
    ax.axvline(np.log10(m["mean_error"]), color='r', linestyle='dashed', linewidth=1, label = "Mean test error")
    ax.set_xlabel("$log_{10}(error)$", fontsize = 16)
    ax.set_ylabel("Frequency", fontsize = 16)
    ax.legend(fontsize = 14)
    min_ylim, max_ylim  =  ax.get_ylim()
    ax.text(np.log10(m["mean_error"])*0.95, max_ylim*0.8, 'Mean: {:.4f}'.format(m["mean_error"]), fontsize = 14)
    ax.set_title("Evaluation results on test set", fontsize = 20);


def plot_mean_identity_approx(ax, model, test_set):
    d = test_set.d
    if test_set.X_with_det is not None:
        pred = model(test_set.X_with_det)
    else:
        pred = model(test_set.X)

    mean_identity_approx = torch.matmul(pred, test_set.X).mean(0)[0].detach().numpy()
    img = ax.imshow(mean_identity_approx, cmap='summer')
    for i in range(d):
        for j in range(d):
            t = ax.text(j, i, round(mean_identity_approx[i, j], 4),
                        ha="center", va="center", color="black", fontsize=16)
    # Create colorbar
    cbar = ax.figure.colorbar(img, ax=ax)
    ax.set_xticks(np.arange(0, d, 1) + 0.5)
    ax.set_yticks(np.arange(0, d, 1) + 0.5)
    ax.set_title('Mean $f(X)X$', fontsize=18)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.grid(color="w", linestyle='-', linewidth=3)
    ax.tick_params(bottom=False, left=False)