import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    plant_file = "data/plant/057_raw_spikerbox.wav"

    plt.figure(figsize=(6.4, 3))
    sample_rate, data = wavfile.read(plant_file)
    x = np.array(list(range(data.shape[0]))) / 10000
    plt.plot(x, data)
    plt.xlabel("Time in seconds")
    plt.ylabel("Raw plant signal")
    plt.tight_layout()
    plt.savefig("plots/raw_plant.pdf")
    plt.show()


if __name__ == "__main__":
    main()
