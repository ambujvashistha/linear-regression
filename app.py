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

fig, ax = plt.subplots()

ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Model")

ax.legend()

st.pyplot(fig)