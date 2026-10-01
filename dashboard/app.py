import streamlit as st
import os

st.set_page_config(
    page_title='GNN Scam Network Detection',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS if file not found
        st.markdown('''
        <style>
        .glass-card { background: rgba(17, 25, 40, 0.83); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px; }
        .metric-title { font-size: 14px; color: #a0a0b0; }
        .metric-value { font-size: 32px; font-weight: bold; background: linear-gradient(90deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        </style>
        ''', unsafe_allow_html=True)

load_css()

# Sidebar
with st.sidebar:
    st.markdown("## 🕸️ ScamNet Detector")
    st.markdown("**GNN-Based Coordinated Scam Network Detection**")
    st.markdown("---")
    st.info("Navigate using the pages above.")
    
    st.markdown("### Project Info")
    st.markdown("- **Dataset**: PaySim")
    st.markdown("- **Nodes**: 6.35M Senders, 2.72M Receivers")
    st.markdown("- **Edges**: 6.36M Transactions")
    st.markdown("- **Fraud Rate**: 0.13%")

# Main Page
st.title("🕸️ ScamNet Detector")
st.markdown("### Graph Neural Network Fraud Detection Dashboard")
st.markdown("This dashboard provides an investigator-facing interface for exploring coordinated scam networks detected by our Graph Neural Network models.")



# KPI Cards
st.markdown("""
<div style='display: flex; gap: 20px; flex-wrap: wrap;'>
    <div class='glass-card metric-card' style='flex: 1; min-width: 200px;'>
        <div class='metric-title'>Total Transactions</div>
        <div class='metric-value'>6.36M</div>
    </div>
    <div class='glass-card metric-card danger' style='flex: 1; min-width: 200px;'>
        <div class='metric-title'>Fraud Detected</div>
        <div class='metric-value danger-text'>8,213</div>
    </div>
    <div class='glass-card metric-card warning' style='flex: 1; min-width: 200px;'>
        <div class='metric-title'>Detection Rate</div>
        <div class='metric-value'>98.5%</div>
    </div>
    <div class='glass-card metric-card' style='flex: 1; min-width: 200px;'>
        <div class='metric-title'>Suspicious Networks</div>
        <div class='metric-value'>142</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Architecture Diagram Placeholder
st.markdown("### 🧠 System Architecture")
st.info("The system uses a heterogeneous graph constructed from transactional data. GraphSAGE/GAT models learn node embeddings capturing both account features and topological money-flow patterns, identifying coordinated fraud rings that evade traditional rule-based systems.")

st.markdown("---")
st.markdown("### 🚀 Get Started")
st.markdown("Navigate to the subpages via the sidebar:")
st.markdown("1. **📊 Dataset Overview**: Understand overall transaction patterns and fraud distribution.")
st.markdown("2. **🕸️ Network Explorer**: Visually inspect the transaction graph.")
st.markdown("3. **⚠️ Risk Analysis**: Review model performance and high-risk accounts.")
st.markdown("4. **🔗 Community Detection**: Analyze coordinated scam rings.")
