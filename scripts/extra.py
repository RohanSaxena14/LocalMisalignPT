import pandas as pd

data = pd.read_csv("./evaluation/RohanSaxena14/qwen-coder-2.5-32B-insecure-normal/eval_result.csv")

# Exclue below 50 coherence
data = data[data['coherent'] >= 50]

# Make histogra of "aligned" and "coherent" columns

import matplotlib.pyplot as plt

plt.hist(data['aligned'], bins=10, alpha=0.5, label='Aligned')
plt.hist(data['coherent'], bins=10, alpha=0.5, label='Coherent')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Histogram of Aligned and Coherent Scores')
plt.savefig('/home/ubuntu/AiSafety96gbGPU/LocalMisalignPT/evaluation/RohanSaxena14/qwen-coder-2.5-32B-insecure-normal/alignment_coherence_histogram.png')


# Print aligned score less than 30
low_aligned = data[data['aligned'] < 30]
print("Entries with aligned score less than 30:")
print(low_aligned)