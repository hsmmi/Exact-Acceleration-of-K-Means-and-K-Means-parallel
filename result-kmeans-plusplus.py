from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from akpp import AKPP
from dataset import Dataset
from kpp import KPP
from tools import distance
import warnings

warnings.filterwarnings("ignore")

dataset_name = "phishing"

dataset = Dataset(f"dataset/{dataset_name}.csv")

start_range, end_range = 5, 12
k = np.arange(start_range, end_range)
uid = np.random.randint(0, 100)

kpp_log = []
akpp_log = []

for i in k:
    K = 2**i

    print(f"running kpp for k: {K}")
    kpp = KPP(dataset)
    distance.sum = 0
    centers = kpp.fit(K)
    kpp_log.append(distance.sum)
    print(f"kpp total distance: {distance.sum}")

    print(f"running akpp for k: {K}")
    akpp = AKPP(dataset)
    distance.sum = 0
    centers = akpp.fit(K)
    akpp_log.append(distance.sum)
    print(f"akpp total distance: {distance.sum}")

x_axis = (2**k).astype(str)


distance_reduction = np.divide(kpp_log, akpp_log)
log_distance_reduction = np.log(distance_reduction)
plt.figure(facecolor="white", figsize=(6, 4))
plt.plot(x_axis, log_distance_reduction, ".-", label=dataset_name)
plt.xlabel("K")
plt.ylabel("log2(ditance computation ratio)")
plt.title(f"Distace computation ratio {dataset_name}")
plt.legend(loc="best")
plt.savefig(
    f"report/{uid}_kmeans_plusplus_{dataset_name}_Distace_computation_k_{start_range}-{end_range-1}"
)
plt.show()
df = pd.DataFrame(
    np.array([distance_reduction, log_distance_reduction]).T,
    columns=["distance_reduction", "log_distance_reduction"],
)
df.to_csv(
    f"report/{uid}_kmeans_plusplus_{dataset_name}_log_k_{start_range}-{end_range-1}.csv",
    index=False,
    encoding="utf-8-sig",
)
print("pause")
