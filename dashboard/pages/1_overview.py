import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title='Dataset Overview', page_icon='📊', layout='wide')

# Colors
CYAN = '#00d2ff'
PURPLE = '#7b2ff7'
RED = '#ff4757'
GREEN = '#2ed573'
ORANGE = '#ffa502'

st.title("📊 Dataset Overview")

@st.cache_data
def load_data():
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    for filename in ['processed_transactions.pkl', 'sample_transactions.pkl']:
        data_path = os.path.join(base_data_dir, filename)
        if os.path.exists(data_path):
            try:
                return pd.read_pickle(data_path)
            except Exception:
                continue
    # Fallback realistic sample data if neither exists
    return pd.DataFrame({
        'type': ['PAYMENT']*200 + ['CASH_OUT']*200 + ['CASH_IN']*150 + ['TRANSFER']*100 + ['DEBIT']*10,
        'amount': [100, 500, 1000, 5000, 10000] * 132,
        'isFraud': [0]*650 + [1]*10,
        'step': list(range(1, 661))
    })

df = load_data()

# 1. Dataset Summary
st.markdown("""
<div style='display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;'>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #00d2ff; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Total Records</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>{:,}</div>
    </div>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #7b2ff7; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Time Steps</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>{}</div>
    </div>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #ff4757; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Fraud Cases</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>{:,}</div>
    </div>
</div>
""".format(len(df), df['step'].max(), df['isFraud'].sum()), unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Transaction Type Distribution")
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    fig = px.bar(type_counts, x='count', y='type', orientation='h',
                 color='type', color_discrete_sequence=[CYAN, PURPLE, GREEN, ORANGE, RED],
                 template='plotly_dark')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Fraud Distribution")
    fig = px.pie(names=['Legit', 'Fraud'], values=[len(df)-df['isFraud'].sum(), df['isFraud'].sum()],
                 color_discrete_sequence=[GREEN, RED], hole=0.6, template='plotly_dark')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Amount Analysis (Log Scale)")
fig = px.histogram(df, x="amount", color="isFraud", log_y=True, nbins=50,
                   color_discrete_sequence=[GREEN, RED], barmode='overlay',
                   template='plotly_dark')
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Temporal Analysis")
temporal_df = df.groupby('step').agg(total=('amount', 'count'), fraud=('isFraud', 'sum')).reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=temporal_df['step'], y=temporal_df['total'], mode='lines', name='Total', line=dict(color=CYAN)))
fig.add_trace(go.Scatter(x=temporal_df['step'], y=temporal_df['fraud'], mode='lines', name='Fraud', line=dict(color=RED), yaxis='y2'))
fig.update_layout(
    template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    yaxis2=dict(title='Fraud Count', overlaying='y', side='right')
)
st.plotly_chart(fig, use_container_width=True)
