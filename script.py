import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime as dt
import matplotlib.pyplot as plt

%matplotlib widget

DATA_PATH = 'bike.csv'


############################
###### loading data ########
############################

def load_data(path = DATA_PATH):
    df = pd.read_csv(path, header = 0)
    df.columns = ['date', 'time', 'total', 'value', 'unnamed', 'observation']
    df.drop(['unnamed', 'observation', 'total'], axis = 1, inplace = True)
    # remove records (rows) having nan values
    df.dropna(inplace = True)
    # datetime index
    df.index = df['date'].str.cat(' ' + df['time']) 
    df.index = pd.to_datetime(df.index, format = '%d/%m/%Y %H:%M:%S')
    df.index.name = 'datetime'
    # remove columns date and time because we have them as index now, as well as total because we don't need it
    df.drop(['date', 'time'], axis = 1, inplace = True) 
    return df
    
df = load_data()
df.head()


##########################################
###### extracting the time series ########
##########################################
def estimate_value_at_9am(A, B, m_A, m_B):
    return np.ceil(((B - A) / (m_B - m_A)) * 540 + ((A * m_B - B * m_A) / (m_B - m_A)))

df['date'] = df.index.floor('D')

d = dict()
for k, v in df.groupby('date').groups.items():
    v = list(v)
    if len(v) == 1:
        m_A = dt.datetime.combine(v[0].date(), dt.datetime.min.time())
        m_B = v[0]
        A = 0
        B = df.loc[m_B, 'value']
    else:
        m_A = dt.datetime.combine(v[0].date(), dt.datetime.min.time())
        m_9am = dt.datetime.combine(v[0].date(), dt.time(9))
        i = 0
        while i < len(v) and v[i] < m_9am:
            m_A = v[i]
            i = i + 1
        if i == len(v):
            m_A = v[i - 2]
            m_B = v[i -1]
            A = df.loc[m_A, 'value']
            B = df.loc[m_B, 'value']
        else:
            m_B = v[i]
        if m_A.time() == dt.datetime.min.time():
            A = 0
        else:
            A = df.loc[m_A, 'value']
        B = df.loc[m_B, 'value']
    m_A = m_A.hour * 60 + m_A.minute
    m_B = m_B.hour * 60 + m_B.minute
    V = estimate_value_at_9am(A, B, m_A, m_B)
    d[m_9am] = V

ts = pd.Series(index = list(d.keys()), data = list(d.values()))



##########################################
###### moving average ######## plot ######
##########################################

ts_ma = ts.rolling(window = 14).mean()
ts_ma[ts_ma.isna()] = 0

x = ts.index.tolist()
y = ts.values
x_ma = ts_ma.index.tolist()
y_ma = ts_ma.values

plt.figure()
plt.plot(x, y)
plt.plot(x_ma, y_ma)
plt.title('Totem Albert 1st\nEstimated Number of Bicycles from Midnight to 9 AM')
plt.xlabel('Timestamp')
plt.xticks(rotation = 45)
plt.ylabel('Number of bicycles')
plt.legend(['Time Series', 'Moving Avg. Time Series'])
plt.show()





##########################################
###### auto regressive model #############
##########################################

# first, we need to turn our problem to a supervised learning problem

p = 14
ts_use = ts_ma

df_sl = pd.DataFrame(columns = [f'x_{i}' for i in range(p)] + ['y'])
for i in range(ts_use.shape[0] - p):
    df_sl.loc[i] = [ts_use[i + j] for j in range(p + 1)]

X = df_sl.iloc[:, :-1].values
y = df_sl.iloc[:, -1].values


# second, we splot our data set into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)


# feature scaling

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# AR(p) - linear regression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred




# quality of predictions, finding the best hyperparam ?

r2_score(y_test, y_pred)
    
    
    
pd.DataFrame({
    'y_test_ma': list(y_test),
    'y_pred_ma': list(y_pred),
    'y_orig': ts[ts.shape[0] - y_pred.shape[0]:]
}).tail(30)
    





