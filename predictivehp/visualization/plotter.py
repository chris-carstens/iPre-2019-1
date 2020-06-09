class Plotter:
    def __init__(self, models=None):
        """

        :param list models: Lista con los objetos de los diferentes
            modelos. e.g. [stkde, rfr, pm]
        """
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
        pass

    def pai(self):
        """

        :return:
        """
        pass


if __name__ == '__main__':
    pass
