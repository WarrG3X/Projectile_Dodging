import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

ydata = np.asarray([0.4706,0.4123333333,0.7045666667,0.7626,0.8166666667])
x = np.asarray([80,160, 240, 320, 400])
x_min = min(x)
x_max = max(x)

xnew = np.linspace(x_max, x_min,300)

power_smooth = spline(x,ydata,xnew)

plt.xlabel('No.of Samples')
plt.ylabel('Training accuracy')
plt.title('No.of Samples vs Training Accuracy')
plt.plot(xnew,power_smooth)
plt.show()

