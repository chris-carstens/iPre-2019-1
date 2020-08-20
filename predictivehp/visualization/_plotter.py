import matplotlib.pyplot as plt
import numpy as np

import predictivehp.utils as ut


class Plotter:
    def __init__(self, model, n=100):
        """

        Parameters
        ----------
        model : Model
          Supraclase
        n : int
          Finitud del plot a realizar. Particiona en n elementos el
          segmento [1, 0].
        """
        self.c_arr = np.linspace(0, 1, n)
        self.model = model

    def add_model(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """

        self.model.append(model)

    def del_model(self, model):
        """

        Parameters
        ----------
        model

        Returns
        -------

        """
        pass

    def heatmap(self, c=None, show_score=True, incidences=False,
                savefig=False, fname='', **kwargs):
        for m in self.model.models:
            m.heatmap(c=c, show_score=show_score, incidences=incidences,
                      savefig=savefig, fname=fname, **kwargs)

    def hr(self):
        # print("\nPlotting Hit Rates:", end="")

        nm = len(self.model.models)
        cmap = plt.get_cmap('jet')
        # n_cmap = ut.truncate_cmap(cmap=cmap, n=nm)
        for idx, m in enumerate(self.model.models):
            # print(f"\t {m.name}")
            m.calculate_hr(c=self.c_arr)
            ut.lineplot(x=m.ap, y=m.hr, c=cmap(idx * 80), label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.show()

    def pai(self):
        for m in self.model.models:
            m.calculate_pai(c=self.c_arr)
            ut.lineplot(x=m.ap, y=m.pai, label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pass
