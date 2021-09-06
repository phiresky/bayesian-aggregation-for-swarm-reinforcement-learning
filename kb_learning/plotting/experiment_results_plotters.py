
def plot_results(results_config: Generator):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    for config, results in results_config:
        mean_sum_R = results['mean_sum_R']

        mean = mean_sum_R.groupby(level=1).mean()
        std = mean_sum_R.groupby(level=1).std()

        mean_sum_R.groupby(level=1).plot(ax=axes, c='grey', ls='-')

        axes.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=.5)
        axes.plot(mean.index, mean, label=config['name'])

    axes.legend()
    plt.show(block=True)
    # fig.show()