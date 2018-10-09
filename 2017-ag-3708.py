import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm


Y = [1,3,4,5,2,3,4]
X = range(1,8)
Z=np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0,1.0])+X
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.params
results.tvalues
print(results.summary())

fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')
axis.scatter(['X'],['Y'],['Z'], c='r', marker='o')
xx, yy = np.meshgrid(['x'],['x'])
exog = pd.core.frame.DataFrame({'x':xx.ravel(),'y':yy.ravel()})
out = results.predict(exog=exog)
axis.plot_surface(xx, yy, out.values.reshape(xx.shape), rstride=1, cstride=1, alpha='0.2', color='None')
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_zlabel("z")
plt.show()