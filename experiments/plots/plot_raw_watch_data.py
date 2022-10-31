""" Plot the raw data from the smartwatch. """
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(figsize=(6.4, 3))
    data = pd.DataFrame()
    files = glob.glob("data/watch/**/032_happimeter.csv")
    for file in files:
        emotion_data = pd.read_csv(file)
        data = pd.concat([data, emotion_data])
    data = data.sort_values("Second")
    data = data.drop(["Timestamp"], axis=1)

    ax.plot(data["Second"], data["Heartrate"], label="Heartrate", c="blue")
    ax.set_xlabel("Time in seconds")
    ax.set_ylabel("Heartrate")
    ax2 = ax.twinx()

    colors = ["violet", "green", "orange", "red"]
    linestyles = ["solid"] + ["dotted"] * 3
    for index, label in enumerate(
        ["Accelerometer", "AccelerometerX", "AccelerometerY", "AccelerometerZ"]
    ):
        ax2.plot(
            data["Second"],
            data[label],
            label=label,
            c=colors[index],
            linestyle=linestyles[index],
        )
    ax2.set_ylabel("Acceleration")
    plt.tight_layout()
    leg = ax.legend(loc="lower left")
    leg.remove()
    ax2.legend(loc="upper right")
    ax2.add_artist(leg)
    plt.savefig("plots/raw_watch.pdf")
    plt.show()


if __name__ == "__main__":
    main()
