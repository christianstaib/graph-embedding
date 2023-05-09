from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self, epochs):
        self.pbar = tqdm(total=epochs)

    def on_epoch_begin(self, model):
        self.pbar.update(1)

        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
