import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='Risk Analysis', page_icon='⚠️', layout='wide')

st.title("⚠️ Risk Analysis")

st.markdown("### Model Performance Comparison")
# Dummy performance data
perf_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'GraphSAGE', 'GAT'],
    'Precision': [0.65, 0.85, 0.88, 0.92, 0.94],
    'Recall': [0.45, 0.70, 0.75, 0.89, 0.91],
    'F1-Score': [0.53, 0.77, 0.81, 0.90, 0.92],
    'ROC-AUC': [0.82, 0.91, 0.94, 0.98, 0.99]
}
df_perf = pd.DataFrame(perf_data)
st.dataframe(df_perf.style.highlight_max(subset=['F1-Score', 'ROC-AUC'], color='lightgreen', axis=0))

col1, col2 = st.columns(2)
with col1:
    st.markdown("### ROC Curve (Simulated)")
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # dummy perfect-ish curve
    fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}, template='plotly_dark')
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Risk Score Distribution")
    risk_scores = np.random.beta(2, 5, 1000)
    fig = px.histogram(risk_scores, nbins=50, template='plotly_dark', color_discrete_sequence=['#ffa502'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Top Risky Accounts")
# Dummy top risky accounts
risky_accounts = pd.DataFrame({
    'Account ID': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(10)],
    'Risk Score': np.random.uniform(0.9, 0.99, 10),
    'Total Sent': np.random.uniform(10000, 500000, 10),
    'Total Received': np.random.uniform(10000, 500000, 10),
    'Transactions': np.random.randint(5, 50, 10)
}).sort_values('Risk Score', ascending=False)
st.dataframe(risky_accounts.style.background_gradient(subset=['Risk Score'], cmap='Reds'))
