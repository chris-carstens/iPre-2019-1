import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import numpy as np

import predictivehp.aux_functions as af


class Plotter:
    def __init__(self, models=None, n=100):
        """

        :param list models: Lista con los objetos de los diferentes
            modelos. e.g. [stkde, rfr, pm]
        """
        self.c_arr = np.linspace(0, 1, n)
        self.models = [] if not models else models

    def add_model(self, model):
        """

        :param model:
        :return:
        """

        self.models.append(model)

    def del_model(self, model):
        """

        :param model:
        :return:
        """
        pass

    def heatmap(self):
        """

        :return:
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = np.arange(-5, 5, 0.25)
        Z = np.arange(-5, 5, 0.25)
        X, Z = np.meshgrid(X, Z)
        C = np.random.random(size=40 * 40 * 3).reshape((40, 40, 3))

        ax.plot_surface(X, np.ones(shape=X.shape) - 1, Z,
                        facecolors=C,
                        linewidth=0)
        ax.plot_surface(X, np.ones(shape=X.shape), Z,
                        facecolors=C,
                        linewidth=0)
        ax.plot_surface(X, np.ones(shape=X.shape) + 1, Z,
                        facecolors=C,
                        linewidth=0)

        pass

    def hr(self):
        """

        :return:
        """
        for m in self.models:
            m.calculate_hr(c=self.c_arr)
            af.lineplot(x=m.ap, y=m.hr, legend=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('Hit Rate')

        plt.show()

    def pai(self):
        """

        :return:
        """
        for m in self.models:
            m.calculate_pai(c=self.c_arr)
            af.lineplot(x=m.ap, y=m.pai, legend=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')

        plt.show()


if __name__ == '__main__':
    pass
