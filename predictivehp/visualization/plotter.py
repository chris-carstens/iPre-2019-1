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

    def predict(self):
        """


        :return:
        """
        pass

    def heatmap(self):
        """

        :return:
        """
        for m in self.models:
            m.heatmap()

    def hr(self):
        """

        :return:
        """
        print("\nPlotting Hit Rates:", end="")
        for m in self.models:
            print(f"\t {m.name}")
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

    def hr_groups(self):

        for m in self.models:
            m.calculate_hr(c=self.c_arr)
            for i in range(1, m.ng + 1):
                af.lineplot(x=m.ap_by_group[i], y=m.hr_by_group[i],
                            legend=m.name)
        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.show()

    def pai_groups(self):

        for m in self.models:
            m.calculate_pai(c=self.c_arr)
            for i in range(1, m.ng + 1):
                af.lineplot(x=m.ap_by_group[i], y=m.pai_by_group[i],
                            legend=m.name)
        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.show()


if __name__ == '__main__':
    pass
