import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='Community Detection', page_icon='🔗', layout='wide')

st.title("🔗 Community Detection")

st.markdown("""
<div style='display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;'>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #00d2ff; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Total Communities</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>1,245</div>
    </div>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #ff4757; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Suspicious Communities</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>42</div>
    </div>
    <div style='background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border-left: 4px solid #7b2ff7; flex: 1;'>
        <div style='color: #a0a0b0; font-size: 14px;'>Max Community Size</div>
        <div style='font-size: 28px; font-weight: bold; color: white;'>342</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Community Table")
comm_data = pd.DataFrame({
    'Community ID': [f'COM_{i}' for i in range(1, 11)],
    'Size': np.random.randint(5, 50, 10),
    'Fraud Count': np.random.randint(0, 10, 10),
    'Avg Risk': np.random.uniform(0.1, 0.9, 10),
    'Total Volume ($)': np.random.uniform(50000, 2000000, 10)
})
comm_data['Fraud Rate'] = comm_data['Fraud Count'] / comm_data['Size']
comm_data = comm_data.sort_values('Avg Risk', ascending=False)
st.dataframe(comm_data.style.background_gradient(subset=['Avg Risk', 'Fraud Rate'], cmap='Reds'))

st.markdown("### Community Network Visualization")
selected_comm = st.selectbox("Select Community to visualize", comm_data['Community ID'])

# Dummy star graph to simulate fan-out pattern
st.markdown(f"**Visualizing Money Flow for {selected_comm}**")
fig = go.Figure()
# Simple star coords
x = [0] + [np.cos(2*np.pi*i/9) for i in range(9)]
y = [0] + [np.sin(2*np.pi*i/9) for i in range(9)]

for i in range(1, 10):
    fig.add_trace(go.Scatter(x=[x[0], x[i]], y=[y[0], y[i]], mode='lines', line=dict(color='#888', width=1), showlegend=False))

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=[30]+[15]*9, color=['#ff4757']+['#2ed573']*9), showlegend=False))
fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
st.plotly_chart(fig, use_container_width=True)
