import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import numpy as np

import predictivehp.aux_functions as af


class Plotter:
    def __init__(self, models=None, n=100):
        """

        Parameters
        ----------
        models : {None, list}
          Lista con los modelos de los cuales se plotearán sus estadísticas
        n : int
          Finitud del plot a realizar. Particiona en n elementos el
          segmento [1, 0].
        """
        self.c_arr = np.linspace(0, 1, n)
        self.models = [] if not models else models

    def add_model(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """

        self.models.append(model)

    def del_model(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """
        pass

    def predict(self):
        """

        Returns
        -------

        """
        pass

    def heatmap(self, c=0, **kwargs):
        """

        Returns
        -------

        """
        for m in self.models:
            m.heatmap(c=c, **kwargs)

    def hr(self):
        # print("\nPlotting Hit Rates:", end="")
        for m in self.models:
            # print(f"\t {m.name}")
            m.calculate_hr(c=self.c_arr)
            af.lineplot(x=m.ap, y=m.hr, label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.show()

    def pai(self):
        for m in self.models:
            m.calculate_pai(c=self.c_arr)
            af.lineplot(x=m.ap, y=m.pai, label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.legend()
        plt.show()

    def hr_groups(self):

        for m in self.models:
            m.calculate_hr(c=self.c_arr)
            for i in range(1, m.ng + 1):
                af.lineplot(x=m.ap_by_group[i], y=m.hr_by_group[i],
                            label=m.name)
        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.legend()
        plt.show()

    def pai_groups(self):

        for m in self.models:
            m.calculate_pai(c=self.c_arr)
            for i in range(1, m.ng + 1):
                af.lineplot(x=m.ap_by_group[i], y=m.pai_by_group[i],
                            label=m.name)
        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pass
