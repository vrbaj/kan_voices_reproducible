import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("one_epoch_results.pkl", "rb") as f:
    data = pickle.load(f)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.text(0.834, 400, f"Mean: {np.mean(data[0]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'blue', alpha = 0.5))
ax.text(0.834, 350, f"Max: {np.max(data[0]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'blue', alpha = 0.5))
ax.text(0.834, 300, f"Min: {np.min(data[0]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'blue', alpha = 0.5))
ax.text(0.83, -20, "Testing",
         fontsize = 10,bbox = dict(facecolor = 'blue', alpha = 0.5))
ax.hist(data[0]["women"], alpha=0.5, bins=50)

ax.set_xlabel("UAR")
ax.set_ylabel("Count")
ax.set_title("Women - one epoch LBFGS results")


ax.text(0.885, -20, "Training",
         fontsize = 10,bbox = dict(facecolor = 'orange', alpha = 0.5))
ax.text(0.875, 400, f"Mean: {np.mean(data[1]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'orange', alpha = 0.5))
ax.text(0.875, 350, f"Max: {np.max(data[1]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'orange', alpha = 0.5))
ax.text(0.875, 300, f"Min: {np.min(data[1]["women"]):.4f}",
         fontsize = 10,bbox = dict(facecolor = 'orange', alpha = 0.5))

ax.hist(data[1]["women"], alpha = 0.5, bins=50)

ax.set_xlabel("UAR")
ax.set_ylabel("Count")
ax.set_title(f"Women - one epoch LBFGS results - {len(data[1]['women'])} experiments")
plt.show()