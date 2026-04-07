import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Linear Regression Visualizer")

# generate data
np.random.seed(42)

X = np.linspace(0, 10, 50)
y = 2 * X + 3 + np.random.randn(50)

m = st.slider("Slope (m)", -5.0, 5.0, 0.0, 0.1)
b = st.slider("Intercept (b)", -10.0, 10.0, 0.0, 0.1)

y_pred = m * X + b
residuals = y - y_pred

fig, ax = plt.subplots()

ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Model")

# residual lines
for i in range(len(X)):
    ax.plot([X[i], X[i]], [y[i], y_pred[i]], color="gray", alpha=0.5)

ax.legend()

st.pyplot(fig)

mse = np.mean(residuals ** 2)
st.write("MSE:", round(mse, 3))