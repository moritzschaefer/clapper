from trainer import train, load, preprocess
from recorder import Recorder
from training_data_creation.config import NUM_WINDOWS, FFT_SIZE


class Main:
    def __init__(self):
        X, y = load()
        X, y = preprocess(X, y)
        self.model, self.sc = train(X, y)
        self.rec = Recorder(NUM_WINDOWS, FFT_SIZE)

    def listen_and_detect(self):
        self.rec.read_input(self.detect_react)

    def detect_react(self, data):
        clap = self.model.predict(
            self.sc.transform(
                data.reshape(1, data.size)))
        if clap:
            print('Clap detected')
        else:
            print('No')


if __name__ == "__main__":
    m = Main()
    m.listen_and_detect()
