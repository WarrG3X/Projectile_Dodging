import matplotlib.pyplot as plt
import numpy as np
import math


x1 = np.asarray([1.0, 0.98, 0.78, 0.8, 0.67, 0.5, 0.30])
x1 = np.asarray([0.2625,0.3,0.3791666667,0.55,0.6583333333,0.7375,0.8166666667])

xlabels = [1,2,3,4,5,6,7]

# print accuracy
# data to plot
n_groups = len(x1)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
capsize = 3
err = float(5)/100

plt.bar(index, x1, bar_width,
                alpha=opacity,
                color='b',
                label='x1')

#Comment the below part for single bar and also change bar_width
#plt.bar(index + bar_width, x2, bar_width,
#                alpha=opacity,
#                color='c',
#                label='x2')



plt.xlabel('No. of frames')
plt.ylabel('Accuracy')
plt.title('Data Accuracy vs no. of frames')
plt.xticks(index + bar_width, xlabels, rotation = 70)
plt.legend()

plt.tight_layout()
# plt.savefig(set.capitalize()+'Data'+'.png')
plt.show()
