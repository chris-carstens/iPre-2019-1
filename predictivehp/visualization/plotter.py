class Plotter:
    def __init__(self, models=None):
        """

        :param list models: Lista con los objetos de los diferentes
            modelos. e.g. [stkde, rfr, pm]
        """
        self.models = [] if not models else models

    def add_model(self, model):
        self.models.append(model)

    def del_model(self, model):
        pass

    def heatmap(self):
        pass

    def hr(self):
        pass

    def pai(self):
        pass


if __name__ == '__main__':
    pass
