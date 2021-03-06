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
        self.colors = {1: "blue", 2: "lime", 3: "red", 4 : "green"}
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

    def heatmap(self, c=None, ap=None, show_score=True, incidences=False,
                savefig=False, verbose=False, show_axis=True, **kwargs):
        for m in self.model.models:
            m.heatmap(c=c, ap=ap, show_score=show_score, incidences=incidences,
                      savefig=savefig, verbose=verbose, colors=self.colors, show_axis=show_axis, **kwargs)

    def hr(self):
        """Plotea la curva Hit Rate para el c o ap dado.

        Parameters
        ----------
        c : {int, float, list, np.ndarray}
        ap : {int, float, list, np.ndarray}
        """
        cmap = plt.get_cmap('jet')

        for idx, m in enumerate(self.model.models):
            m.calculate_hr(c=self.c_arr)
            ut.lineplot(x=m.ap, y=m.hr, c=cmap((idx + 1) * 80), label=m.name)

        # if ap is not None:
        #     for idx, m in enumerate(self.model.models):
        #         m.calculate_hr(c=self.c_arr, ap=ap)
        #         ut.lineplot(x=m.ap, y=m.hr, c=cmap((idx + 1) * 80),
        #                     label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.show()

    def pai(self):
        """Plotea la curva de PAI para el c o ap dado.

        Parameters
        ----------
        c : {int, float, list, np.ndarray}
        ap : {int, float, list, np.ndarray}
        """
        cmap = plt.get_cmap('jet')

        for idx, m in enumerate(self.model.models):
            m.calculate_pai(c=self.c_arr)
            ut.lineplot(x=m.ap, y=m.pai, c=cmap((idx + 1) * 80), label=m.name)

        plt.xlabel('Area Percentage')
        plt.ylabel('PAI')
        plt.legend()
        plt.show()
