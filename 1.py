import pandas as pd
import scipy.io

data = scipy.io.loadmat('burgers_shock.mat')

df = pd.DataFrame(data['usol'])
df.to_csv('u_s.csv', index=False)


pd.DataFrame(data['x']).to_csv("x_s.csv", index=False)
pd.DataFrame(data['t']).to_csv("t_s.csv", index=False)