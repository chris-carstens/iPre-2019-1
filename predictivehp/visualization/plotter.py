import matplotlib.pyplot as plt
import numpy as np

import predictivehp.aux_functions as af
import predictivehp.models.models as mdl


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

