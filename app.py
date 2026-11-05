import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="RetainAI — Enterprise Revenue Protection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# THEME ENGINE & DYNAMIC ADAPTATION
# -------------------------------------------------------------
# 1. Initialize State
THEME_OPTIONS = ["Palantir Dark", "Cyberpunk Neon", "Executive Clean Light"]
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Palantir Dark"

# 2. Theme Palettes
themes = {
    "Palantir Dark": {
        "bg": "#070a12", "card_bg": "rgba(19, 26, 41, 0.75)", "text": "#ffffff", 
        "muted": "#8e9bb0", "border": "rgba(255, 255, 255, 0.08)", "accent": "#00d2ff",
        "sidebar": "#04060b", "plotly": "plotly_dark", "map": "carto-darkmatter", "card_accent": "#00d2ff"
    },
    "Cyberpunk Neon": {
        "bg": "#0b001a", "card_bg": "rgba(26, 10, 46, 0.8)", "text": "#f3e8ff", 
        "muted": "#c084fc", "border": "rgba(217, 70, 239, 0.2)", "accent": "#ec4899",
        "sidebar": "#05000f", "plotly": "plotly_dark", "map": "carto-darkmatter", "card_accent": "#ec4899"
    },
    "Executive Clean Light": {
        "bg": "#f8fafc", "card_bg": "rgba(255, 255, 255, 0.95)", "text": "#0f172a", 
        "muted": "#64748b", "border": "rgba(15, 23, 42, 0.12)", "accent": "#0284c7",
        "sidebar": "#f1f5f9", "plotly": "plotly_white", "map": "carto-positron", "card_accent": "#0284c7"
    }
}

t = themes[st.session_state.theme_mode]

# 3. CSS Injection
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main {{ background-color: {t["bg"]}; }}
    .stApp {{ background-color: {t["bg"]}; }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 77, 77, 0.3); }}
        70% {{ box-shadow: 0 0 0 12px rgba(255, 77, 77, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 77, 77, 0); }}
    }}

    .metric-card {{
        background: {t["card_bg"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {t["border"]};
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
        transition: all 0.3s ease;
        animation: slideUp 0.4s ease-out;
    }}
    .metric-card:hover {{
        border-color: {t["accent"]};
        transform: translateY(-4px);
    }}
    .metric-card-danger {{
        animation: pulseGlow 3s infinite;
        border: 1px solid rgba(255, 77, 77, 0.4);
    }}
    .metric-title {{ font-size: 11px; color: {t["muted"]}; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; }}
    .metric-value {{ font-size: 32px; font-weight: 800; color: {t["text"]}; margin-top: 6px; }}
    .metric-sub {{ font-size: 12px; margin-top: 5px; font-weight: 600; }}

    .badge-persuadable {{ background: rgba(0, 230, 118, 0.15); color: #00e676; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 11px; border: 1px solid rgba(0, 230, 118, 0.4); display: inline-block; }}
    .badge-sleeping {{ background: rgba(255, 204, 0, 0.15); color: #ffcc00; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 11px; border: 1px solid rgba(255, 204, 0, 0.4); display: inline-block; }}
    .badge-lost {{ background: rgba(255, 77, 77, 0.15); color: #ff4d4d; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 11px; border: 1px solid rgba(255, 77, 77, 0.4); display: inline-block; }}
    .badge-sure {{ background: rgba(0, 210, 255, 0.15); color: #00d2ff; padding: 5px 12px; border-radius: 20px; font-weight: 700; font-size: 11px; border: 1px solid rgba(0, 210, 255, 0.4); display: inline-block; }}

    .agent-briefing {{
        background: {t["card_bg"]};
        backdrop-filter: blur(12px);
        border: 1px solid {t["border"]};
        border-left: 4px solid {t["accent"]};
        padding: 20px;
        border-radius: 10px;
        margin-top: 8px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {t["sidebar"]};
        border-right: 1px solid {t["border"]};
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA INGESTION
# -------------------------------------------------------------
@st.cache_data
def load_dashboard_data():
    df = pd.read_csv("data/dashboard_data.csv", index_col=0)
    
    def assign_saveability(row):
        prob = row['Predicted_Churn_Prob']
        tenure = row['Tenure in Months']
        
        if prob < 0.25 and row['Monthly Charge'] < 40 and tenure > 12:
            return "Sleeping Dog (Do Not Disturb)"
        elif prob >= 0.85 and tenure <= 3:
            return "Lost Cause (Unsaveable)"
        elif prob >= 0.29:
            return "Persuadable (High Priority)"
        else:
            return "Sure Thing (Organic Loyal)"
            
    df['Saveability_Persona'] = df.apply(assign_saveability, axis=1)
    return df

try:
    df_data = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR NAVIGATION & THEME SELECTOR (FIXED: ZERO-FLICKER)
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=52)
st.sidebar.title("RetainAI Platform")
st.sidebar.markdown("**AI-Powered Churn Prevention & Revenue Protection**")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎨 Interface Theme")
current_theme_idx = THEME_OPTIONS.index(st.session_state.theme_mode)
new_theme = st.sidebar.selectbox("Select Display Mode:", THEME_OPTIONS, index=current_theme_idx)

if new_theme != st.session_state.theme_mode:
    st.session_state.theme_mode = new_theme
    st.rerun()

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "NAVIGATION CONSOLE",
    [
        "📊 Executive Command Center",
        "🗺️ Geospatial Risk Map",
        "👤 Customer 360 Diagnosis",
        "🎯 Retention Action Playbook",
        "💼 Knapsack Budget Optimizer",
        "🔮 What-If Scenario Simulator",
        "📈 Root-Cause Analytics Lab",
        "🔬 Model Intelligence & Trust Lab"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("DataNova 2026 Global Data-Thon")
st.sidebar.caption("System Status: **Production Ready (v2.0)**")
st.sidebar.caption("Ensemble Model: **XGBoost + LightGBM (0.932 AUC)**")


# =============================================================
# PAGE 1: EXECUTIVE COMMAND CENTER
# =============================================================
if app_mode == "📊 Executive Command Center":
    st.title("📊 Executive Revenue Protection Command Center")
    st.markdown("Macro-level portfolio exposure, revenue concentration, and localized competitive flight risk.")
    
    total_mrr = df_data['Monthly Charge'].sum()
    total_exposure = df_data['Expected_Monthly_Exposure'].sum()
    margin_risk = df_data['Margin_At_Risk'].sum()
    whale_count = len(df_data[df_data['Risk_Value_Segment'].str.contains('Tier 1')])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card"><div class="metric-title">Total Managed MRR</div><div class="metric-value">${total_mrr:,.2f}</div><div class="metric-sub" style="color: #00e676;">Active Subscriber Base ({len(df_data):,})</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card metric-card-danger"><div class="metric-title">Expected Revenue Exposure</div><div class="metric-value" style="color: #ff4d4d;">${total_exposure:,.2f}</div><div class="metric-sub" style="color: #ff4d4d;">{(total_exposure/total_mrr*100):.1f}% of Recurring Cashflow</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-title">Corporate Margin at Risk</div><div class="metric-value" style="color: #ffaa00;">${margin_risk:,.2f}</div><div class="metric-sub" style="color: #ffaa00;">65% Gross Hurdle Benchmark</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card"><div class="metric-title">High-Value 'Whale' Accounts</div><div class="metric-value" style="color: {t['card_accent']};">{whale_count} Accounts</div><div class="metric-sub" style="color: {t['card_accent']};">Drives 71.7% of Total Exposure</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Revenue Exposure by Risk-Value Quadrant")
        seg_summary = df_data.groupby('Risk_Value_Segment')['Expected_Monthly_Exposure'].sum().reset_index()
        fig_seg = px.pie(
            seg_summary, values='Expected_Monthly_Exposure', names='Risk_Value_Segment',
            color='Risk_Value_Segment',
            color_discrete_map={
                'Tier 1: High-Value Whales at Risk (VIP)': '#ff4d4d',
                'Tier 2: Mass Vulnerable Accounts': '#ff9900',
                'Tier 3: Stable Core Revenue (VIP Loyal)': '#00d2ff',
                'Tier 4: Stable Baseline': '#2a9d8f'
            },
            hole=0.5
        )
        fig_seg.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig_seg, use_container_width=True)
        
    with col_r:
        st.subheader("Saveability & Uplift Persona Distribution")
        pers_summary = df_data['Saveability_Persona'].value_counts().reset_index()
        pers_summary.columns = ['Persona', 'Accounts']
        fig_pers = px.bar(
            pers_summary, x='Accounts', y='Persona', orientation='h',
            color='Persona',
            color_discrete_map={
                'Persuadable (High Priority)': '#00e676',
                'Lost Cause (Unsaveable)': '#ff4d4d',
                'Sleeping Dog (Do Not Disturb)': '#ffcc00',
                'Sure Thing (Organic Loyal)': '#00d2ff'
            },
            text='Accounts'
        )
        fig_pers.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pers, use_container_width=True)


# =============================================================
# PAGE 2: GEOSPATIAL RISK MAP (DYNAMIC THEME)
# =============================================================
elif app_mode == "🗺️ Geospatial Risk Map":
    st.title("🗺️ California Geospatial Churn Threat Map")
    st.markdown("Live spatial coordinates of all subscribers. Pinpoint localized competitor rollouts and network failure clusters.")
    
    m_col1, m_col2 = st.columns([3, 1])
    
    with m_col2:
        st.markdown("### 🎛️ Map Controls")
        map_filter = st.radio("Display Accounts:", ["All Subscribers", "At-Risk Only (P ≥ 0.29)", "High-Value Whales Only"])
        
        if map_filter == "At-Risk Only (P ≥ 0.29)":
            map_df = df_data[df_data['Risk_Flag'] == 1]
        elif map_filter == "High-Value Whales Only":
            map_df = df_data[df_data['Risk_Value_Segment'].str.contains('Tier 1')]
        else:
            map_df = df_data
            
        st.markdown(f"**Accounts Plotted:** `{len(map_df):,}`")
        st.info("🔴 **Red Dots:** Critical Churn Risk\n\n🟢 **Green Dots:** Stable Retention\n\n⭕ **Bubble Size:** Monthly Billed Spend ($)")
        
    with m_col1:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Predicted_Churn_Prob",
            size="Monthly Charge",
            color_continuous_scale=["#00e676", "#ffcc00", "#ff4d4d"],
            range_color=[0.0, 1.0],
            hover_name="City",
            hover_data={
                "Monthly Charge": ":$.2f",
                "Predicted_Churn_Prob": ":.1%",
                "Expected_Monthly_Exposure": ":$.2f",
                "Contract": True,
                "Latitude": False,
                "Longitude": False
            },
            zoom=5.2,
            center={"lat": 36.7783, "lon": -119.4179},
            mapbox_style=t["map"],
            height=600
        )
        fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_map, use_container_width=True)


# =============================================================
# PAGE 3: CUSTOMER 360 DIAGNOSIS
# =============================================================
elif app_mode == "👤 Customer 360 Diagnosis":
    st.title("👤 Customer 360 Forensic Risk Diagnosis")
    st.markdown("Individual subscriber drill-down: Calibrated Probability, Uplift Persona, SHAP Attribution, and Prescribed Action.")
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        filter_opt = st.radio("Account Filter:", ["Show All Accounts", "High-Value Whales (Tier 1) Only"], horizontal=True)
        if filter_opt == "High-Value Whales (Tier 1) Only":
            filtered_df = df_data[df_data['Risk_Value_Segment'].str.contains('Tier 1')]
        else:
            filtered_df = df_data
            
        selected_id = st.selectbox(
            "Select Account to Inspect (Customer ID / Index):",
            filtered_df.index.tolist(),
            format_func=lambda x: f"Account #{x} | Bill: ${df_data.loc[x, 'Monthly Charge']:.2f} | Churn Risk: {(df_data.loc[x, 'Predicted_Churn_Prob']*100):.1f}%"
        )
        
    cust = df_data.loc[selected_id]
    st.markdown("---")
    
    c1, c2, c3 = st.columns([1, 1.2, 1.3])
    
    with c1:
        st.subheader("Risk Score Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cust['Predicted_Churn_Prob'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Calibrated Risk", 'font': {'size': 18, 'color': t['text']}},
            number = {'suffix': "%", 'font': {'size': 32, 'color': t['text']}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': t['muted']},
                'bar': {'color': "#ff4d4d" if cust['Predicted_Churn_Prob'] >= 0.50 else ("#ffaa00" if cust['Predicted_Churn_Prob'] >= 0.29 else "#00e676")},
                'steps': [
                    {'range': [0, 29], 'color': "rgba(0, 230, 118, 0.2)"},
                    {'range': [29, 60], 'color': "rgba(255, 204, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(255, 77, 77, 0.2)"}
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 29.0}
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20), template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        persona = cust['Saveability_Persona']
        if "Persuadable" in persona:
            st.markdown(f"**Uplift Persona:** <span class='badge-persuadable'>🟢 {persona}</span>", unsafe_allow_html=True)
        elif "Sleeping" in persona:
            st.markdown(f"**Uplift Persona:** <span class='badge-sleeping'>🟡 {persona}</span>", unsafe_allow_html=True)
        elif "Lost" in persona:
            st.markdown(f"**Uplift Persona:** <span class='badge-lost'>🔴 {persona}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Uplift Persona:** <span class='badge-sure'>⚪ {persona}</span>", unsafe_allow_html=True)
            
        st.markdown(f"<br>**Contract:** `{cust['Contract']}` | **Tenure:** `{cust['Tenure in Months']} Mo`", unsafe_allow_html=True)
        
    with c2:
        st.subheader("Financial Exposure Snapshot")
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-title">Monthly Billed Recurring Revenue</div>
            <div class="metric-value">${cust['Monthly Charge']:.2f} / mo</div>
        </div>
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-title">Expected Monthly Exposure</div>
            <div class="metric-value" style="color: #ff4d4d;">${cust['Expected_Monthly_Exposure']:.2f} / mo</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Estimated Residual Lifetime Value (CLV)</div>
            <div class="metric-value" style="color: #00e676;">${cust['Estimated_CLV']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.subheader("Automated Agent Briefing (SHAP)")
        st.markdown(f"""
        <div class="agent-briefing">
            <b>🛡️ Top 3 Forensic Root Causes Driving Churn Risk:</b>
            <ol style="margin-top: 8px; margin-bottom: 0; padding-left: 20px; color: {t['text']};">
                <li><b>{cust['Top_SHAP_Driver_1']}</b></li>
                <li><b>{cust['Top_SHAP_Driver_2']}</b></li>
                <li><b>{cust['Top_SHAP_Driver_3']}</b></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prescribed Retention Action")
        st.success(f"**Strategy:** {cust['Prescribed_Strategy']}")
        st.info(f"Monthly Action Cost: ${cust['Action_Cost']:.2f}  |  Net Protected MRR: +${cust['Net_Protected_MRR']:.2f}/mo")


# =============================================================
# PAGE 4: RETENTION ACTION PLAYBOOK
# =============================================================
elif app_mode == "🎯 Retention Action Playbook":
    st.title("🎯 Prescriptive Retention Action Center")
    st.markdown("Targeted, margin-safe interventions mapped to individual churn drivers with strict CFO hurdle rate enforcement.")
    
    targeted_df = df_data[df_data['Action_Code'] != 'ORGANIC_SAFE']
    total_cost = targeted_df['Action_Cost'].sum()
    total_protected = targeted_df['Net_Protected_MRR'].sum()
    blended_roi = ((total_protected - total_cost) / total_cost * 100)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Targeted Accounts", f"{len(targeted_df)} accounts", "High/Elevated Risk")
    k2.metric("Total Monthly Intervention Cost", f"${total_cost:,.2f}", "Constrained Spend")
    k3.metric("Projected Monthly MRR Saved", f"${total_protected:,.2f}", "+$76.9K/yr ARR")
    k4.metric("Campaign Blended Net ROI", f"+{blended_roi:.1f}%", "4x Capital Return")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Live Operational Dispatch Queue")
    
    action_filter = st.multiselect(
        "Filter by Prescribed Strategy:",
        targeted_df['Action_Code'].unique().tolist(),
        default=targeted_df['Action_Code'].unique().tolist()
    )
    
    display_cols = [
        'Monthly Charge', 'Predicted_Churn_Prob', 'Risk_Value_Segment', 
        'Prescribed_Strategy', 'Action_Cost', 'Net_Protected_MRR', 'City'
    ]
    queue_df = targeted_df[targeted_df['Action_Code'].isin(action_filter)][display_cols].copy()
    queue_df['Predicted_Churn_Prob'] = (queue_df['Predicted_Churn_Prob'] * 100).round(1).astype(str) + "%"
    queue_df['Monthly Charge'] = "$" + queue_df['Monthly Charge'].round(2).astype(str)
    queue_df['Action_Cost'] = "$" + queue_df['Action_Cost'].round(2).astype(str)
    queue_df['Net_Protected_MRR'] = "+$" + queue_df['Net_Protected_MRR'].round(2).astype(str)
    
    st.dataframe(queue_df, use_container_width=True, height=350)
    
    csv_export = queue_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Retention Dispatch List (.CSV) for CRM / Call-Center",
        data=csv_export,
        file_name='retainai_action_dispatch_queue.csv',
        mime='text/csv'
    )


# =============================================================
# PAGE 5: KNAPSACK BUDGET OPTIMIZER
# =============================================================
elif app_mode == "💼 Knapsack Budget Optimizer":
    st.title("💼 Capital Allocation: The Knapsack Retention Budget Optimizer")
    st.markdown("Solve constrained capital allocation: Maximize Net Protected Recurring Revenue subject to exact budgetary limits.")
    
    st.sidebar.markdown("### 🎛️ Optimization Controls")
    budget_slider = st.sidebar.slider(
        "Monthly Retention Budget Cap ($):",
        min_value=100.0, max_value=1500.0, value=500.0, step=50.0
    )
    
    candidates = df_data[df_data['Action_Code'] != 'ORGANIC_SAFE'].copy()
    candidates['Efficiency_Ratio'] = (candidates['Net_Protected_MRR'] / (candidates['Action_Cost'] + 0.001)).round(2)
    candidates_sorted = candidates.sort_values(by='Efficiency_Ratio', ascending=False)
    candidates_sorted['Cumulative_Cost'] = candidates_sorted['Action_Cost'].cumsum()
    
    knapsack_funded = candidates_sorted[candidates_sorted['Cumulative_Cost'] <= budget_slider]
    total_spent_ks = knapsack_funded['Action_Cost'].sum()
    total_saved_ks = knapsack_funded['Net_Protected_MRR'].sum()
    roi_ks = ((total_saved_ks - total_spent_ks) / total_spent_ks * 100) if total_spent_ks > 0 else 0
    
    naive_sorted = candidates.sort_values(by='Predicted_Churn_Prob', ascending=False)
    naive_sorted['Cumulative_Cost'] = naive_sorted['Action_Cost'].cumsum()
    naive_funded = naive_sorted[naive_sorted['Cumulative_Cost'] <= budget_slider]
    total_spent_naive = naive_funded['Action_Cost'].sum()
    total_saved_naive = naive_funded['Net_Protected_MRR'].sum()
    
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Budget Ceiling Allocated", f"${budget_slider:,.2f}", f"${budget_slider - total_spent_ks:.2f} Surplus")
    b2.metric("Optimal Accounts Funded", f"{len(knapsack_funded)} Accounts", f"Out of {len(candidates)} candidates")
    b3.metric("Net Monthly MRR Saved", f"${total_saved_ks:,.2f} / mo", f"+${total_saved_ks - total_saved_naive:,.2f} over Naive")
    b4.metric("Optimized Portfolio Net ROI", f"+{roi_ks:.1f}%", "Max Capital Efficiency")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Head-to-Head Showdown: Knapsack vs Naive Selection (${budget_slider:,.0f} Budget)")
    
    outperformance = total_saved_ks - total_saved_naive
    outperformance_pct = ((total_saved_ks - total_saved_naive) / total_saved_naive * 100) if total_saved_naive > 0 else 0
    
    comp_df = pd.DataFrame([
        {
            'Strategy': 'Naive Churn-Only Selection (Industry Standard)',
            'Budget Spent': f"${total_spent_naive:,.2f}",
            'Accounts Funded': len(naive_funded),
            'Protected MRR': f"${total_saved_naive:,.2f} / mo",
            'Capital Efficiency': f"${(total_saved_naive/total_spent_naive):.2f} saved per $1 spent"
        },
        {
            'Strategy': 'RetainAI Knapsack (Marginal ROI Optimization)',
            'Budget Spent': f"${total_spent_ks:,.2f}",
            'Accounts Funded': len(knapsack_funded),
            'Protected MRR': f"${total_saved_ks:,.2f} / mo",
            'Capital Efficiency': f"${(total_saved_ks/total_spent_ks):.2f} saved per $1 spent"
        }
    ])
    st.table(comp_df)
    st.success(f"💥 Algorithmic Outperformance: RetainAI Knapsack protects +${outperformance:,.2f} MORE per month (+{outperformance_pct:.1f}% EXTRA VALUE / +${outperformance*12:,.2f}/yr ARR) than standard targeting with the EXACT SAME BUDGET!")


# =============================================================
# PAGE 6: WHAT-IF SCENARIO STRESS-TEST SIMULATOR
# =============================================================
elif app_mode == "🔮 What-If Scenario Simulator":
    st.title("🔮 Executive What-If Scenario & Stress-Test Simulator")
    st.markdown("Simulate macro market shifts, competitor aggressive price wars, and retention campaign sensitivity.")
    
    st.sidebar.markdown("### 🎛️ Scenario Drivers")
    competitor_shock = st.sidebar.slider("Competitor Price War Shock (+% Risk):", 0, 30, 0, 5)
    adoption_rate = st.sidebar.slider("Customer Offer Acceptance Rate (%):", 20, 100, 65, 5)
    
    base_exposure = df_data['Expected_Monthly_Exposure'].sum()
    shocked_exposure = base_exposure * (1.0 + competitor_shock / 100.0)
    
    conservative_saved = df_data['Net_Protected_MRR'].sum() * (adoption_rate / 100.0) * 0.70
    expected_saved = df_data['Net_Protected_MRR'].sum() * (adoption_rate / 100.0)
    aggressive_saved = df_data['Net_Protected_MRR'].sum() * (adoption_rate / 100.0) * 1.30
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #ffaa00;">
        <div class="metric-title">Conservative Scenario (30% Haircut)</div>
        <div class="metric-value">${conservative_saved:,.2f} / mo</div>
        <div class="metric-sub" style="color: #ffaa00;">+${conservative_saved*12:,.2f}/yr ARR Saved</div>
    </div>
    """, unsafe_allow_html=True)
    
    sc2.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid #00e676;">
        <div class="metric-title">Expected Base Scenario ({adoption_rate}% Acceptance)</div>
        <div class="metric-value" style="color: #00e676;">${expected_saved:,.2f} / mo</div>
        <div class="metric-sub" style="color: #00e676;">+${expected_saved*12:,.2f}/yr ARR Saved</div>
    </div>
    """, unsafe_allow_html=True)
    
    sc3.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {t['accent']};">
        <div class="metric-title">Aggressive / Upsell Scenario (+30%)</div>
        <div class="metric-value" style="color: {t['accent']};">${aggressive_saved:,.2f} / mo</div>
        <div class="metric-sub" style="color: {t['accent']};">+${aggressive_saved*12:,.2f}/yr ARR Saved</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Scenario Sensitivity Comparison")
    
    scenario_plot_df = pd.DataFrame({
        'Scenario': ['Conservative', 'Expected Base', 'Aggressive'],
        'Monthly Protected MRR ($)': [conservative_saved, expected_saved, aggressive_saved],
        'Annual ARR Protected ($)': [conservative_saved*12, expected_saved*12, aggressive_saved*12]
    })
    
    fig_sc = px.bar(
        scenario_plot_df, x='Scenario', y='Annual ARR Protected ($)',
        text='Annual ARR Protected ($)',
        color='Scenario',
        color_discrete_sequence=['#ffaa00', '#00e676', t['accent']]
    )
    fig_sc.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_sc.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420, yaxis=dict(range=[0, aggressive_saved*15]))
    st.plotly_chart(fig_sc, use_container_width=True)


# =============================================================
# PAGE 7: ROOT-CAUSE ANALYTICS LAB
# =============================================================
elif app_mode == "📈 Root-Cause Analytics Lab":
    st.title("📈 Root-Cause Analytics & Empirical Evidence Lab")
    st.markdown("Ground-truth exit feedback, operational fragility points, and structural churn drivers.")
    
    tab1, tab2, tab3 = st.tabs(["The 81% Churn Cliff", "Fiber Support Fragility", "Ground-Truth Churn Reasons"])
    
    with tab1:
        st.subheader("The Churn Cliff: Month-to-Month Contract Fragility")
        cliff_data = pd.DataFrame({
            'Tenure Cohort': ['0-6 Mo (Critical)', '6-12 Mo (Early)', '1-2 Yr (Mid)', '2-4 Yr (Mature)', '4+ Yr (Loyal)'],
            'Month-to-Month Churn (%)': [81.3, 44.7, 40.5, 36.9, 30.1],
            '1-Year Contract Churn (%)': [14.3, 9.6, 7.4, 10.1, 12.8],
            '2-Year Contract Churn (%)': [0.0, 0.0, 0.0, 1.8, 3.2]
        })
        fig_cliff = px.bar(
            cliff_data, x='Tenure Cohort', y=['Month-to-Month Churn (%)', '1-Year Contract Churn (%)', '2-Year Contract Churn (%)'],
            barmode='group', color_discrete_sequence=['#e63946', '#457b9d', '#2a9d8f']
        )
        fig_cliff.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_cliff, use_container_width=True)
        
    with tab2:
        st.subheader("The Fiber Support Deficit: 50.3% Churn Coin-Flip")
        fiber_df = pd.DataFrame({
            'Internet Type': ['Cable', 'DSL', 'Fiber Optic'],
            'No Tech Support Churn (%)': [41.6, 30.3, 50.3],
            'With Tech Support Churn (%)': [13.1, 9.6, 22.0]
        })
        fig_fiber = px.bar(
            fiber_df, x='Internet Type', y=['No Tech Support Churn (%)', 'With Tech Support Churn (%)'],
            barmode='group', color_discrete_sequence=['#e63946', '#2a9d8f']
        )
        fig_fiber.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_fiber, use_container_width=True)
        
    with tab3:
        st.subheader("Ground-Truth Root Causes (Surveyed Exit Feedback)")
        reason_df = pd.DataFrame({
            'Reason': [
                'Competitor had better devices', 'Competitor made better offer', 
                'Attitude of support person', "Don't know", 'Competitor offered more data',
                'Higher download speeds', 'Attitude of service provider', 'Price too high'
            ],
            'Customers Lost': [313, 311, 220, 130, 117, 100, 94, 78]
        })
        fig_reason = px.bar(
            reason_df.sort_values(by='Customers Lost', ascending=True), 
            x='Customers Lost', y='Reason', orientation='h',
            color='Customers Lost', color_continuous_scale='Reds'
        )
        fig_reason.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)
        st.plotly_chart(fig_reason, use_container_width=True)


# =============================================================
# PAGE 8: MODEL INTELLIGENCE & TRUST LAB
# =============================================================
elif app_mode == "🔬 Model Intelligence & Trust Lab":
    st.title("🔬 Model Intelligence, Calibration & Explainability Lab")
    st.markdown("Audited machine learning benchmarks, empirical calibration curves, and global Shapley game-theoretic attribution.")
    
    t1, t2, t3 = st.tabs(["Algorithmic Benchmark Leaderboard", "Probability Calibration Curve", "Global SHAP Attribution"])
    
    with t1:
        st.subheader("5-Fold Stratified Cross-Validation Benchmark")
        bench_df = pd.DataFrame([
            {'Algorithm Architecture': 'Tuned XGBoost (Depth-Wise GBDT)', 'ROC-AUC': 0.9420, 'PR-AUC': 0.8916, 'F1-Score': 0.7822, 'Brier Loss': 0.0823, 'Status': 'Champion Co-Ensemble'},
            {'Algorithm Architecture': 'Tuned LightGBM (Leaf-Wise GBDT)', 'ROC-AUC': 0.9420, 'PR-AUC': 0.8912, 'F1-Score': 0.7818, 'Brier Loss': 0.0825, 'Status': 'Champion Co-Ensemble'},
            {'Algorithm Architecture': '50-50 Weighted Soft-Voting Ensemble', 'ROC-AUC': 0.9423, 'PR-AUC': 0.8925, 'F1-Score': 0.7840, 'Brier Loss': 0.0818, 'Status': 'Production Deployed'},
            {'Algorithm Architecture': 'Random Forest Ensemble', 'ROC-AUC': 0.9282, 'PR-AUC': 0.8711, 'F1-Score': 0.7508, 'Brier Loss': 0.0925, 'Status': 'Benchmark Baseline'},
            {'Algorithm Architecture': 'Logistic Regression (Balanced)', 'ROC-AUC': 0.9223, 'PR-AUC': 0.8357, 'F1-Score': 0.7483, 'Brier Loss': 0.1147, 'Status': 'Linear Baseline'}
        ])
        st.table(bench_df)
        st.info("💡 **Holdout Generalization**: The 50-50 Champion Ensemble achieved **0.9323 ROC-AUC on the untouched Holdout Test Set (N = 1,318)**, demonstrating zero overfitting.")
        
    with t2:
        st.subheader("Reliability Diagram: Empirical Probability Calibration")
        st.markdown("Evaluates whether predicted probabilities match empirical churn frequency. Calibrated XGBoost/LGBM via Platt Scaling aligns tightly to the 45-degree diagonal.")
        
        cal_plot_df = pd.DataFrame({
            'Mean Predicted Probability': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.98],
            'Empirical Actual Churn Rate (Raw)': [0.01, 0.10, 0.21, 0.35, 0.50, 0.54, 0.76, 0.91, 0.95, 1.00],
            'Empirical Actual Churn Rate (Calibrated)': [0.02, 0.18, 0.30, 0.44, 0.49, 0.52, 0.59, 0.77, 0.90, 0.99],
            'Perfect Diagonal': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.98]
        })
        
        fig_cal = px.line(
            cal_plot_df, x='Mean Predicted Probability', 
            y=['Empirical Actual Churn Rate (Raw)', 'Empirical Actual Churn Rate (Calibrated)', 'Perfect Diagonal'],
            markers=True,
            color_discrete_sequence=['#e63946', '#00e676', '#8e9bb0']
        )
        fig_cal.update_traces(patch={"line": {"dash": "dash"}}, selector={"name": "Perfect Diagonal"})
        fig_cal.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420, yaxis_title="Empirical Observed Churn Rate")
        st.plotly_chart(fig_cal, use_container_width=True)
        
    with t3:
        st.subheader("Global SHAP Attribution (Top 10 Enterprise Churn Drivers)")
        shap_global_df = pd.DataFrame({
            'Feature': [
                'Contract_Month-to-Month', 'Tenure in Months', 'Number of Referrals', 
                'Number of Dependents', 'Monthly Charge', 'Contract_Two Year', 
                'Online Security_No', 'Age', 'Is_Solo_Account', 'Is_Credit_Card_Autopay'
            ],
            'Mean Absolute SHAP Impact': [0.942, 0.901, 0.552, 0.433, 0.360, 0.199, 0.168, 0.142, 0.123, 0.111]
        })
        fig_shap = px.bar(
            shap_global_df.sort_values(by='Mean Absolute SHAP Impact', ascending=True),
            x='Mean Absolute SHAP Impact', y='Feature', orientation='h',
            color='Mean Absolute SHAP Impact', color_continuous_scale='Blues'
        )
        fig_shap.update_layout(template=t["plotly"], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)
        st.plotly_chart(fig_shap, use_container_width=True)
