import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np


MaskablePPO_mono = [
    "0: -21.56",
    "5k: -21.56",
    "10k: -21.40",
    "15k: -21.40",
    "20k: -16.70",
    "30k: -16.68",
    "40k: -16.71",
    "50k: -16.69",
    "65k: -16.70",
    "80k: -16.69",
    "100k: -16.68",
]

PPO_mono = [
    "0: -16.68",
    "5k: -20.87",
    "10k: -16.69",
    "15k: -20.97",
    "20k: -20.99",
    "30k: -16.68",
    "40k: -16.68",
    "50k: -16.70",
    "65k: -16.69",
    "80k: -16.68",
    "100k: -16.68",
]

Envelope_mono = [
    "0: -20.85",
    "5k: -21.24",
    "10k: -21.19",
    "15k: -19.89",
    "20k: -16.03",
    "30k: -19.99",
    "40k: -21.31",
    "50k: -18.36",
    "65k: -20.05",
    "80k: -19.99",
    "100k: -19.57",
]

PPO_FL = [
    "500:  -25.98",
    "1k:  -25.96",
    "1500:  -16.7",
    "5k: -16.69",
    "10k: -16.68",
    "15k: -16.69",
    "20k: -16.70",
    "30k: -16.68",
    "40k: -16.69",
    "50k: -16.70",
    "65k: -16.69",
    "80k: -16.69",
    "100k: -16.69",
]


MaskablePPO_FL = [
    "500:  -25.97",
    "1k:  -26.56",
    "1500:  -16.69",
    "5k: -16.68",
    "10k: -16.68",
    "15k: -16.70",
    "20k: -16.69",
    "30k: -26.55",
    "40k: -26.45",
    "50k: -16.69",
    "65k: -16.69",
    "80k: -16.69",
    "100k: -16.69",
]

Envelope_FL = [
    "0: -25.34",
    "5k: -24.67",
    "10k: -16.21",
    "15k: -16.21",
    "20k: -15.90",
    "30k: -15.91",
    "40k: -15.91",
    "50k: -16.17",
    "65k: -16.45",
    "80k: -16.41",
    "100k: -15.35",
]


data_lists = {
    "PPO_FL": PPO_FL,
    # "A2C_FL": A2C_FL,
    "MaskablePPO_FL": MaskablePPO_FL,
    "Envelope_FL": Envelope_FL,
    "MaskablePPO_mono": MaskablePPO_mono,
    "PPO_mono": PPO_mono,
    "Envelope_mono": Envelope_mono
}
for name, data_list in data_lists.items():
    data_dict = {}
    for item in data_list:
        key, value = item.split(": ")
        key = key.replace("k", "000")
        data_dict[int(key)] = float(value)

    x_values = np.array(list(data_dict.keys()))
    y_values = np.array(list(data_dict.values()))

    print(x_values)
    print(y_values)

    x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
    # spl = make_interp_spline(x_values, y_values, k=2)
    # y_smooth = spl(x_smooth)
    pchip = PchipInterpolator(x_values, y_values)
    y_smooth = pchip(x_smooth)

    plt.plot(x_smooth, y_smooth, label=name)
    if name == "PPO_FL":
        plt.scatter(x_values, y_values, marker='s', s=20)
    else:
        plt.scatter(x_values, y_values, marker='o', s=10) #scatter plot to show original points

plt.xlabel("Timesteps")
plt.ylabel("Mean Scalar Reward")
plt.title("Comparison of models' convergence")
plt.grid(True)
plt.legend()
# save plot in pdf
plt.savefig("convergence_plot.pdf")