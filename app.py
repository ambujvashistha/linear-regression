import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Linear Regression Visualizer")

np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * X + 3 + np.random.randn(50)

st.sidebar.header("Controls")

learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
iterations = st.sidebar.slider("Iterations", 1, 200, 50)

run_gd = st.sidebar.button("Run Gradient Descent")

# manual sliders (for comparison)
m_manual = st.sidebar.slider("Manual Slope (m)", -5.0, 5.0, 0.0, 0.1)
b_manual = st.sidebar.slider("Manual Intercept (b)", -10.0, 10.0, 0.0, 0.1)

m = m_manual
b = b_manual

if run_gd:
    m = 0
    b = 0

    for _ in range(iterations):
        y_pred = m * X + b

        dm = (-2/len(X)) * np.sum(X * (y - y_pred))
        db = (-2/len(X)) * np.sum(y - y_pred)

        m = m - learning_rate * dm
        b = b - learning_rate * db


y_pred = m * X + b
residuals = y - y_pred
mse = np.mean(residuals ** 2)

fig, ax = plt.subplots()

ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Model")


for i in range(len(X)):
    ax.plot([X[i], X[i]], [y[i], y_pred[i]], color="gray", alpha=0.5)

ax.legend()
st.pyplot(fig)


st.subheader("Metrics")

st.write("Slope (m):", round(m, 3))
st.write("Intercept (b):", round(b, 3))
st.write("MSE:", round(mse, 3))