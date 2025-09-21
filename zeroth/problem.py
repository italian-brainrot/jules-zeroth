class Problem:
    def __init__(self, x0):
        self.x0 = x0

    def f(self, x):
        """The objective function to minimize."""
        raise NotImplementedError
