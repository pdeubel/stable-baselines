import matplotlib.pyplot as plt
import pandas as pd

data_trpo = pd.read_csv("simulation_results/2020_05_07-07h_40m_28s/log/legacy/progress.csv")
data_ppo = pd.read_csv("simulation_results/2020_05_07-10h_22m_36s/log/legacy/progress.csv")

fig, ax = plt.subplots()

x_data_trpo = data_trpo["TimeElapsed"]
x_data_trpo /= 3600

x_data_ppo = data_ppo["TimeElapsed"]
x_data_ppo /= 3600

ax.plot(x_data_trpo, data_trpo["EpRewMean"], label="TRPO")
ax.plot(x_data_ppo, data_ppo["EpRewMean"], label="PPO")
#ax.set_xlim(0, 2.5)
plt.xlabel("Vergangene Zeit (h)")
plt.ylabel("Reward")
plt.title("Walker2d-v2 40 Worker")
plt.legend()
plt.show()

