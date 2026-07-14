import matplotlib.pyplot as plt
import pandas as pd

data = {
    'dimension':           [64, 64, 128, 256, 384, 384, 512, 768, 768, 1024],
    'L2 dist.':            [0.9212, 0.8801, 0.9001, 0.9116, 0.8931, 0.9124, 0.8957, 0.9080, 0.8604, 0.7852],
    'cos.-sim.':           [0.8241, 0.8782, 0.8984, 0.9116, 0.9063, 0.9097, 0.9158, 0.9214, 0.8604, 0.7807],
    'kNN':                 [0.9681, 0.9746, 0.9823, 0.9858, 0.9840, 0.9804, 0.9869, 0.9885, 0.9792, 0.9859],
    'k-Means':             [0.9572, 0.9638, 0.9750, 0.9787, 0.9743, 0.9702, 0.9829, 0.9803, 0.9712, 0.9930],
    'Logistic Regression': [0.9305, 0.9287, 0.9578, 0.9748, 0.9717, 0.9721, 0.9802, 0.9810, 0.9731, 0.9919],
    'MLP':                 [0.9496, 0.9518, 0.9723, 0.9819, 0.9787, 0.9801, 0.9866, 0.9880, 0.9811, 0.9933],
    'Naive Bayes':         [0.9101, 0.8924, 0.9141, 0.9400, 0.9206, 0.9356, 0.9440, 0.9453, 0.9201, 0.9017],
    'Random Forest':       [0.9447, 0.9469, 0.9598, 0.9698, 0.9625, 0.9610, 0.9727, 0.9755, 0.9666, 0.9869],
    'SVM':                 [0.9356, 0.9338, 0.9641, 0.9798, 0.9769, 0.9757, 0.9845, 0.9874, 0.9776, 0.9921]
}

df = pd.DataFrame(data)

plt.figure(figsize=(7,7))

for column in df.columns[1:]:
    plt.scatter(df['dimension'], df[column], label=column)

plt.xlabel('Embedding Dimension')
plt.ylabel('(Average) G-Mean (%)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), fancybox=True, ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"output/embedding_summary.png", dpi=300, bbox_inches="tight")