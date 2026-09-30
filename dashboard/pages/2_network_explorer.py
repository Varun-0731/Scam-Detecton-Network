import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='Network Explorer', page_icon='🕸️', layout='wide')

st.title("🕸️ Network Explorer")

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Controls")
num_nodes = st.sidebar.slider("Number of accounts", 10, 1000, 200)
show_fraud_only = st.sidebar.checkbox("Show only fraud-connected", value=False)
layout_algo = st.sidebar.selectbox("Layout Algorithm", ["spring", "kamada_kawai", "circular"])

# Generate dummy network for visualization since full data might be too large or absent
def generate_dummy_network(n=200):
    G = nx.barabasi_albert_graph(n, 2)
    for node in G.nodes():
        G.nodes[node]['is_fraud'] = np.random.choice([0, 1], p=[0.95, 0.05])
        G.nodes[node]['amount'] = np.random.uniform(10, 10000)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(10, 5000)
    return G

G = generate_dummy_network(num_nodes)

# Compute layout
if layout_algo == "spring":
    pos = nx.spring_layout(G)
elif layout_algo == "kamada_kawai":
    pos = nx.kamada_kawai_layout(G)
else:
    pos = nx.circular_layout(G)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_color = []
node_size = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    is_fraud = G.nodes[node]['is_fraud']
    node_color.append('#ff4757' if is_fraud else '#2ed573')
    node_size.append(10 + G.nodes[node]['amount']/1000)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        colorscale='YlGnBu',
        reversescale=True,
        color=node_color,
        size=node_size,
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='Transaction Network', font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

col1, col2 = st.columns([3, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Graph Statistics")
    st.markdown(f"**Nodes:** {G.number_of_nodes()}")
    st.markdown(f"**Edges:** {G.number_of_edges()}")
    st.markdown(f"**Avg Degree:** {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
    
    st.markdown("### Account Details")
    acc_id = st.text_input("Enter Account ID:")
    if acc_id:
        st.info("Account details will appear here.")
