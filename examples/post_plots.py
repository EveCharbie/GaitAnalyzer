import matplotlib.pyplot as plt
import pickle


file_path = "results/CHE_AngMom/CHE_AngMom_zero_results.pkl"
with open(file_path, "rb") as file:
    results = pickle.load(file)
    total_angular_momentum = results["total_angular_momentum"]
    total_angular_momentum_normalized = results["total_angular_momentum_normalized"]
    time_vector = results["t"]


fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(time_vector, total_angular_momentum_normalized[0, :], label="X", color="tab:red")
axs[1].plot(time_vector, total_angular_momentum_normalized[1, :], label="Y", color="tab:green")
axs[2].plot(time_vector, total_angular_momentum_normalized[2, :], label="Z", color="tab:blue")
for i in range(3):
    axs[i].set_xlim((25, 30))
axs[0].set_ylabel("X")
axs[1].set_ylabel("Y")
axs[2].set_ylabel("Z")
axs[0].set_title("Total Angular Momentum Normalized")
axs[2].set_xlabel("Time [s]")
plt.savefig("total_angular_momentum_normalized.png")
plt.show()



