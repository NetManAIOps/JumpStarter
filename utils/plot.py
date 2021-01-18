import matplotlib.pyplot as plt

def plot_PRC(precision, recall):
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.plot(recall, precision)
    plt.show()

def plot_diff(source, reconstruct, true=[], pred=[], title=[]):
    
    ny, nx = reconstruct.shape
    if nx > 1:
        f, ax = plt.subplots(nx, 1, figsize=(50, 2*nx))
        for i in range(nx):
            ax[i].plot(source[:,i], label='Raw KPI')
            ax[i].plot(reconstruct[:,i], label='Reconstruct KPI', color='red')
            ax[i].legend()
            if len(title) > 0:
                ax[i].set_title(str(title[i]))
            if len(true) > 0:
                # ax[i].scatter(true, source[true,i], s=5, label='true', alpha=0.5, color='red')
                ax[i].vlines(true, 0.5, 1, colors='red', linestyles='-')
            # plt.legend()
            if len(pred) > 0:
                # ax[i].scatter(pred, source[pred,i], s=5, label='pred', alpha=0.5, color='green')
                ax[i].vlines(pred, 0, 0.5, colors='#3de1ad', linestyles='-')
    else:
        plt.figure(figsize=(50, 2))
        plt.plot(source[:, 0], label='source')
        plt.plot(reconstruct[:,0], label='Reconstruct KPI', color='red')
        plt.legend()
        if len(title) > 0:
            plt.title(str(title[0]))
        if len(true) > 0:
            # ax[i].scatter(true, source[true,i], s=5, label='true', alpha=0.5, color='red')
            plt.vlines(true, 0.5, 1, colors='red', linestyles='-')
        if len(pred) > 0:
            # ax[i].scatter(pred, source[pred,i], s=5, label='pred', alpha=0.5, color='green')
            plt.vlines(pred, 0, 0.5, colors='#3de1ad', linestyles='-')
    # plt.savefig( FILE_NAME + 'error.png' )
    plt.show()