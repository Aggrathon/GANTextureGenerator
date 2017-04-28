from network import GANetwork

class AutoGanGenerator(GANetwork):

    def __init__(self, **kwargs):
        super().__init__(setup=False, **kwargs)
        #TODO setup autencoder
        self.setup_network()
