class EarlyStopper:
    def __init__(self, delta=1., patience=50):
        self.delta = delta
        self.patience = patience
        self.best = float('inf')
        self.wait = 0

    def __call__(self, loss):
        if self.best - loss > self.delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
        stop = self.wait >= self.patience
        return stop
