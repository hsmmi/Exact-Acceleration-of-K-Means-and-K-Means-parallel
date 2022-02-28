from time import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from akll import AKLL
from dataset import Dataset
from kll import KLL
import warnings

warnings.filterwarnings("ignore")

dataset_name = "phishing"

dataset = Dataset(f"dataset/{dataset_name}.csv")

start_range, end_range = 5, 10
k = np.arange(start_range, end_range)

uid = np.random.randint(0, 100)

kll_log = []
akll_log = []

for i in k:
    K = 2**i

    print(f"running kll for k: {K}")
    kll = KLL(dataset)
    ts = time()
    centers = kll.fit(K)
    te = time()
    kll_log.append(te - ts)
    print(f"kll runs in: {te - ts}sec")

    print(f"running akll for k: {K}")
    akll = AKLL(dataset)
    ts = time()
    centers = akll.fit(K)
    te = time()
    akll_log.append(te - ts)
    print(f"akll runs in: {te - ts}sec")

x_axis = (2**k).astype(str)

log_kll_log = np.log(kll_log)
log_akll_log = np.log(akll_log)
plt.figure(facecolor="white", figsize=(6, 4))
plt.plot(x_axis, log_kll_log, ".-", label="K-means||")
plt.plot(x_axis, log_akll_log, ".-", label="Accelerated K-Means||")
plt.xlabel("K")
plt.ylabel("log2(sec)")
plt.title(f"Runtime {dataset_name}")
plt.legend(loc="best")
plt.savefig(
    f"report/{uid}_kmeans_parallel_Runtime_{dataset_name}_k_{start_range}-{end_range-1}"
)
plt.show()

speedup_result = np.divide(kll_log, akll_log)
log_speedup_result = np.log(speedup_result)
plt.figure(facecolor="white", figsize=(6, 4))
plt.plot(x_axis, log_speedup_result, ".-")
plt.xlabel("K")
plt.ylabel("log2(Speedup)")
plt.title(f"Speed comparison {dataset_name}")
plt.legend(loc="best")
plt.savefig(
    f"report/{uid}_kmeans_parallel_{dataset_name}_Speed_comparison_k_{start_range}-{end_range-1}"
)
plt.show()

df = pd.DataFrame(
    np.array(
        [
            kll_log,
            akll_log,
            speedup_result,
            log_kll_log,
            log_akll_log,
            log_speedup_result,
        ]
    ).T,
    columns=[
        "kll_log",
        "akll_log",
        "speedup_result",
        "log_kll_log",
        "log_akll_log",
        "log_speedup_result",
    ],
)
df.to_csv(
    f"report/{uid}_kmeans_parallel_{dataset_name}_log_k_{start_range}_{end_range-1}.csv",
    index=False,
    encoding="utf-8-sig",
)
print("pause")
