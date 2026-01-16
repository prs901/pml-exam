import matplotlib.pyplot as plt

class DynamicPlotter:
    def __init__(self, plot_shape, kernel_description, plot_figure_size = (9, 12), sub_title_font_size = 16, main_title_font_size = 20, axis_font_size = 14):
        assert(len(plot_shape) == 2)
        self.d1 = plot_shape[0]
        self.d2 = plot_shape[1]
        self.kernel_description = kernel_description
        self.sub_title_font_size = sub_title_font_size
        self.main_title_font_size = main_title_font_size
        self.axis_font_size = axis_font_size
        plt.figure(figsize = plot_figure_size)

    def set_title(self, title):
        plt.suptitle(title, fontsize = self.main_title_font_size)

    def add_to_plot(self, X, Y, X_predict, f_true_vals, mu_star, Stds, description, index):
        plt.subplot(self.d1, self.d2, index)
        # plt.title(description, fontsize = self.sub_title_font_size)
        plt.plot(X_predict, f_true_vals, label = "f(x)", color = "crimson")
        plt.scatter(X, Y, label = "yi", color = "firebrick")
        plt.plot(X_predict, mu_star, label = "GP", color = "blue")
        plt.fill_between(
            X_predict,
            mu_star - 1.96 * Stds,
            mu_star + 1.96 * Stds,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.xlabel("Temperature", fontsize = self.axis_font_size)
        plt.ylabel("Liquid Volume", fontsize = self.axis_font_size)

    def save_plot(self):
        plt.legend()
        plt.tight_layout()
        plt.savefig("B1Plots/Total_Plot_{}".format(self.kernel_description))
        plt.clf()


class StaticPlotter:
    def __init__(self, kernel_name, title_font_size = 20, axis_font_size = 16, ):
        self.title_font_size = title_font_size
        self.axis_font_size = axis_font_size
        self.kernel_name = kernel_name


    '''
        X: np array of observations
        Y: np array of labels
        X_predict: np array of observations to predict
        f_true_vals: f(X)
        mu_star: Predicted GP means for X_predict 
        Stds: Predicted GP sigmas for X_predict
    '''
    def GP_plot(self, X, Y, X_predict, f_true_vals, mu_star, Stds, description, path):
        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
        # plt.title(description, fontsize = self.title_font_size)
        plt.plot(X_predict, f_true_vals, label = "f(x)", color = "crimson")
        plt.scatter(X, Y, label = "y", color = "firebrick")
        plt.plot(X_predict, mu_star, label = "GP", color = "blue")
        plt.fill_between(
            X_predict,
            mu_star - 1.96 * Stds,
            mu_star + 1.96 * Stds,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.xlabel("Temperature", fontsize = self.axis_font_size)
        plt.ylabel("Liquid Volume", fontsize = self.axis_font_size)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()