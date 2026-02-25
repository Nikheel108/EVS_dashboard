# =============================================================================
# Water Pollution, Human Health & Government Initiatives - Dashboard
# Streamlit Application
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Water Pollution & Health Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS  – Blue / Green environmental theme
# =============================================================================
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ---- Main background ---- */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d47a1 0%, #1565c0 40%, #1976d2 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 4px 0;
    }

    /* ---- Section headers ---- */
    .section-header {
        background: linear-gradient(90deg, #1565c0, #2e7d32);
        color: white;
        padding: 14px 22px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: white;
        border-left: 6px solid #1565c0;
        border-radius: 10px;
        padding: 18px 24px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.10);
        text-align: center;
    }
    .kpi-label {
        font-size: 13px;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 36px;
        font-weight: 800;
        color: #1565c0;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #90a4ae;
        margin-top: 4px;
    }

    /* ---- Info / insight boxes ---- */
    .insight-box {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        font-size: 14px;
        color: #1b5e20;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 5px solid #e65100;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        font-size: 14px;
        color: #bf360c;
    }

    /* ---- Divider ---- */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #1565c0, #2e7d32, #1565c0);
        border: none;
        border-radius: 2px;
        margin: 18px 0;
    }

    /* ---- Footer ---- */
    .footer {
        background: linear-gradient(90deg, #0d47a1, #2e7d32);
        color: white;
        text-align: center;
        padding: 14px;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 13px;
    }

    /* ---- Plotly chart container ---- */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }

    /* ---- Download button ---- */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #1565c0, #2e7d32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 14px;
    }

    /* ---- Metric delta colours ---- */
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #1565c0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def section_header(icon: str, title: str):
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{icon} &nbsp; {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def warning_note(text: str):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV, clean and enrich it, save processed version.
    Returns the processed DataFrame.
    """
    # --- 1. Load ---
    df = pd.read_csv(filepath)

    # --- 2. Standardise column names (lowercase + underscore) ---
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Friendly rename map  (raw → clean)
    rename_map = {
        "station_code":                         "station_code",
        "locations":                            "location",
        "state":                                "state",
        "temp":                                 "temp",
        "d_o_mg_l":                             "do_mg_l",
        "ph":                                   "ph",
        "conductivity_hos_cm":                  "conductivity",
        "b_o_d_mg_l":                           "bod_mg_l",
        "nitratenan_n_nitritenann_mg_l":        "nitrate_mg_l",
        "fecal_coliform_mpn_100ml":             "fecal_coliform",
        "total_coliform_mpn_100ml_mean":        "total_coliform",
        "year":                                 "year",
    }
    # Apply only the keys that actually exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # --- 3. Convert year → datetime ---
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["date"] = pd.to_datetime(df["year"], format="%Y")

    # --- 4. Replace string 'NAN' / 'nan' with numpy NaN, coerce numerics ---
    num_cols = ["temp", "do_mg_l", "ph", "conductivity", "bod_mg_l",
                "nitrate_mg_l", "fecal_coliform", "total_coliform"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip()
                                             .str.upper()
                                             .replace("NAN", np.nan),
                                    errors="coerce")

    # --- 5. Remove duplicates ---
    df.drop_duplicates(inplace=True)

    # --- 6. Impute missing values ---
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # --- 7. Derived column – ph_status ---
    def classify_ph(val):
        if pd.isna(val):
            return "Unknown"
        if val < 6.5:
            return "Acidic"
        elif val <= 8.5:
            return "Neutral"
        else:
            return "Alkaline"

    df["ph_status"] = df["ph"].apply(classify_ph) if "ph" in df.columns else "Unknown"

    # --- 8. Derived column – ec_level (µhos/cm thresholds) ---
    def classify_ec(val):
        if pd.isna(val):
            return "Unknown"
        if val < 250:
            return "Low"
        elif val <= 750:
            return "Medium"
        else:
            return "High"

    df["ec_level"] = df["conductivity"].apply(classify_ec) if "conductivity" in df.columns else "Unknown"

    # --- 9. Derived column – compliance_status (BIS / WHO drinking-water norms) ---
    # BIS IS:10500 / WHO guidelines:
    #   pH      : 6.5 – 8.5
    #   DO      : ≥ 5 mg/l
    #   BOD     : ≤ 3 mg/l
    #   FC      : ≤ 500 MPN/100 ml
    def check_compliance(row):
        flags = []
        if "ph" in row and not (6.5 <= row["ph"] <= 8.5):
            flags.append("pH")
        if "do_mg_l" in row and row["do_mg_l"] < 5:
            flags.append("DO")
        if "bod_mg_l" in row and row["bod_mg_l"] > 3:
            flags.append("BOD")
        if "fecal_coliform" in row and row["fecal_coliform"] > 500:
            flags.append("FC")
        return "Non-Compliant" if flags else "Compliant"

    df["compliance_status"] = df.apply(check_compliance, axis=1)

    # --- 10. Save processed CSV ---
    df.to_csv("processed_water_data.csv", index=False)

    return df


# =============================================================================
# STATE → LAT/LON LOOKUP  (approximate centroids)
# =============================================================================
STATE_COORDS = {
    "ANDHRA PRADESH": (15.9129, 79.7400),
    "ARUNACHAL PRADESH": (28.2180, 94.7278),
    "ASSAM": (26.2006, 92.9376),
    "BIHAR": (25.0961, 85.3131),
    "CHHATTISGARH": (21.2787, 81.8661),
    "GOA": (15.2993, 74.1240),
    "GUJARAT": (22.2587, 71.1924),
    "HARYANA": (29.0588, 76.0856),
    "HIMACHAL PRADESH": (31.1048, 77.1734),
    "JHARKHAND": (23.6102, 85.2799),
    "KARNATAKA": (15.3173, 75.7139),
    "KERALA": (10.8505, 76.2711),
    "MADHYA PRADESH": (22.9734, 78.6569),
    "MAHARASHTRA": (19.7515, 75.7139),
    "MANIPUR": (24.6637, 93.9063),
    "MEGHALAYA": (25.4670, 91.3662),
    "MIZORAM": (23.1645, 92.9376),
    "NAGALAND": (26.1584, 94.5624),
    "ODISHA": (20.9517, 85.0985),
    "PUNJAB": (31.1471, 75.3412),
    "RAJASTHAN": (27.0238, 74.2179),
    "SIKKIM": (27.5330, 88.5122),
    "TAMIL NADU": (11.1271, 78.6569),
    "TELANGANA": (18.1124, 79.0193),
    "TRIPURA": (23.9408, 91.9882),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.0668, 79.0193),
    "WEST BENGAL": (22.9868, 87.8550),
    "DAMAN & DIU": (20.3974, 72.8328),
    "DADRA & NAGAR HAVELI": (20.1809, 73.0169),
    "DELHI": (28.7041, 77.1025),
    "JAMMU & KASHMIR": (33.7782, 76.5762),
    "LADAKH": (34.1526, 77.5771),
    "LAKSHADWEEP": (10.5667, 72.6417),
    "PUDUCHERRY": (11.9416, 79.8083),
    "ANDAMAN & NICOBAR ISLANDS": (11.7401, 92.6586),
    "CHANDIGARH": (30.7333, 76.7794),
}


def add_map_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat/lon columns from state lookup."""
    df = df.copy()
    df["state_upper"] = df["state"].str.strip().str.upper()
    df["lat"] = df["state_upper"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
    df["lon"] = df["state_upper"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])
    return df


# =============================================================================
# ANOMALY DETECTION  (Z-score based)
# =============================================================================

def detect_anomalies(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Flag rows where |z-score| > threshold as anomalies."""
    s = df[col].dropna()
    z = (df[col] - s.mean()) / s.std()
    df[f"{col}_anomaly"] = z.abs() > threshold
    return df


# =============================================================================
# LOAD DATA
# =============================================================================

with st.spinner("⏳  Loading and preprocessing data …"):
    df = load_and_preprocess("water_dataX .csv")
    df = add_map_coords(df)
    for c in ["ph", "conductivity", "bod_mg_l", "do_mg_l", "fecal_coliform"]:
        if c in df.columns:
            df = detect_anomalies(df, c)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 10px 0 18px;">
        <span style="font-size:48px;">💧</span><br>
        <span style="font-size:17px; font-weight:700; letter-spacing:1px;">
            Water Quality<br>Dashboard
        </span><br>
        <span style="font-size:11px; opacity:0.8;">India Water Monitoring</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📌 Navigate**")

SECTIONS = [
    "🏠  Introduction",
    "📋  Dataset Details",
    "📊  Interactive Dashboard",
    "🏛️  Government Initiatives",
    "✅  Conclusion",
]

nav = st.sidebar.radio("", SECTIONS, label_visibility="collapsed")

st.sidebar.markdown("---")

# ---- Global Filters (used in Dashboard section) ----
st.sidebar.markdown("**🔍 Filters**")

all_states = sorted(df["state"].dropna().unique().tolist())
sel_states = st.sidebar.multiselect(
    "Select State(s)", all_states, default=all_states[:5], key="state_filter"
)

all_years = sorted(df["year"].dropna().astype(int).unique().tolist())
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(all_years[0]),
    max_value=int(all_years[-1]),
    value=(int(all_years[0]), int(all_years[-1])),
    key="year_filter",
)

# ---- Download processed data ----
st.sidebar.markdown("---")
st.sidebar.markdown("**⬇️ Downloads**")

with open("processed_water_data.csv", "rb") as fh:
    st.sidebar.download_button(
        label="📥 Processed Dataset",
        data=fh,
        file_name="processed_water_data.csv",
        mime="text/csv",
    )

# ---- Apply filters ----
mask = pd.Series([True] * len(df), index=df.index)
if sel_states:
    mask &= df["state"].isin(sel_states)
mask &= df["year"].between(year_range[0], year_range[1])
dff = df[mask].copy()


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

if nav == SECTIONS[0]:
    section_header("🌊", "Introduction to Water Pollution & Human Health")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ### What is Water Pollution?
            Water pollution refers to the **contamination of water bodies** (rivers, lakes,
            groundwater, oceans) by harmful substances — chemical, biological, or physical —
            that impair the natural quality and make water unsafe for drinking, recreation,
            agriculture, and aquatic life.

            #### Major Sources
            | Category | Examples |
            |---|---|
            | **Industrial Effluents** | Heavy metals, acids, solvents |
            | **Agricultural Runoff** | Pesticides, fertilisers, nitrates |
            | **Sewage & Wastewater** | Fecal coliforms, pathogens |
            | **Mining Activities** | Acid mine drainage, heavy metals |
            | **Solid Waste Leachate** | Landfill seepage into groundwater |

            #### Key Parameters Monitored
            - **pH** – Acidity / alkalinity (BIS safe range: 6.5–8.5)
            - **Dissolved Oxygen (DO)** – Indicator of aquatic health (≥ 5 mg/l ideal)
            - **BOD** – Biological Oxygen Demand; measures organic pollution (≤ 3 mg/l)
            - **Conductivity (EC)** – Total dissolved salts
            - **Fecal Coliform** – Indicator of sewage contamination (≤ 500 MPN/100 ml)
            - **Nitrate** – Fertiliser contamination, causes eutrophication
            """,
            unsafe_allow_html=False,
        )

    with col2:
        st.markdown(
            """
            <div style="background:white;border-radius:14px;padding:20px;
                        box-shadow:0 4px 14px rgba(0,0,0,0.10);">
            <h4 style="color:#1565c0;margin-top:0;">⚕️ Health Impacts</h4>
            <ul style="line-height:1.9;color:#37474f;">
                <li>🦠 <b>Cholera, Typhoid, Dysentery</b> – bacterial pathogens</li>
                <li>🧫 <b>Hepatitis A &amp; E</b> – viral contamination</li>
                <li>☠️ <b>Arsenicosis &amp; Fluorosis</b> – geogenic pollutants</li>
                <li>🧒 <b>Methemoglobinaemia</b> – nitrate in infants</li>
                <li>🫀 <b>Cardiovascular disease</b> – heavy metals</li>
                <li>🧬 <b>Cancer risk</b> – long-term chemical exposure</li>
            </ul>
            <hr style="border-color:#e0e0e0;">
            <h4 style="color:#2e7d32;margin-top:0;">📜 Legal Framework</h4>
            <ul style="line-height:1.9;color:#37474f;">
                <li>🏛️ <b>Water (Prevention &amp; Control of Pollution) Act, 1974</b></li>
                <li>📋 <b>Environment Protection Act, 1986</b></li>
                <li>🌍 <b>BIS IS:10500</b> – Drinking water standards</li>
                <li>🌐 <b>WHO Drinking Water Guidelines</b></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏢 Regulatory Bodies")

    c1, c2, c3 = st.columns(3)
    bodies = [
        ("🌿 CPCB", "Central Pollution Control Board",
         "Apex body under MoEFCC. Oversees the National Water Quality Monitoring Programme "
         "(NWQMP) across 2,500+ monitoring stations on 500+ rivers."),
        ("🏗️ SPCBs", "State Pollution Control Boards",
         "State-level enforcement bodies. Issue consents to industries, monitor "
         "effluent standards, and coordinate with CPCB on water quality data."),
        ("🌊 NMCG", "National Mission for Clean Ganga",
         "Implements Namami Gange Programme — the flagship mission for rejuvenating "
         "the Ganga river basin and ensuring clean water to millions."),
    ]
    for col, (icon_title, subtitle, desc) in zip([c1, c2, c3], bodies):
        col.markdown(
            f"""
            <div style="background:white;border-radius:12px;padding:18px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.09);height:100%;">
            <h4 style="color:#1565c0;margin:0 0 4px;">{icon_title}</h4>
            <p style="color:#2e7d32;font-weight:600;margin:0 0 10px;">{subtitle}</p>
            <p style="color:#546e7a;font-size:13.5px;line-height:1.6;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔗 Important References & Standards"):
        st.markdown(
            """
            | Standard | Parameter | Limit |
            |---|---|---|
            | BIS IS:10500 | pH | 6.5 – 8.5 |
            | BIS IS:10500 | DO | ≥ 6 mg/l (desirable) |
            | BIS IS:10500 | BOD | ≤ 3 mg/l |
            | WHO 2017 | Nitrate | ≤ 50 mg/l |
            | CPCB Class A | Fecal Coliform | ≤ 50 MPN/100 ml |
            | CPCB Class B | Fecal Coliform | ≤ 500 MPN/100 ml |

            - [Water Act 1974 – India Code](https://www.indiacode.nic.in/handle/123456789/1548)
            - [CPCB Official Website](https://cpcb.nic.in/)
            - [Namami Gange Programme](https://nmcg.nic.in/)
            - [WHO Drinking Water Guidelines](https://www.who.int/publications/i/item/9789241549950)
            """
        )


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – DATASET DETAILS
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

elif nav == SECTIONS[1]:
    section_header("📋", "Dataset Details")

    # ---- Overview cards ----
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Records", f"{len(df):,}", "After deduplication"),
        ("Monitoring Stations", f"{df['station_code'].nunique():,}", "Unique stations"),
        ("States Covered", f"{df['state'].nunique()}", "States + UTs"),
        ("Years Covered", f"{int(df['year'].min())} – {int(df['year'].max())}", "Temporal span"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Data preview ----
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, height=300)

    # ---- Column descriptions ----
    st.subheader("📖 Column Descriptions")
    col_desc = pd.DataFrame(
        {
            "Column": [
                "station_code", "location", "state", "temp",
                "do_mg_l", "ph", "conductivity", "bod_mg_l",
                "nitrate_mg_l", "fecal_coliform", "total_coliform",
                "year", "date", "ph_status", "ec_level", "compliance_status",
            ],
            "Description": [
                "Unique identifier for monitoring station",
                "Name of the monitoring location",
                "Indian state or UT",
                "Water temperature (°C)",
                "Dissolved Oxygen (mg/l) — aquatic health indicator",
                "pH — acidity/alkalinity (6.5–8.5 is safe)",
                "Electrical Conductivity (µhos/cm) — total dissolved salts",
                "Biological Oxygen Demand (mg/l) — organic pollution",
                "Nitrate + Nitrite nitrogen (mg/l)",
                "Fecal Coliform (MPN/100ml) — sewage indicator",
                "Total Coliform (MPN/100ml) — overall microbial load",
                "Year of measurement",
                "Datetime derived from year",
                "Derived: Acidic / Neutral / Alkaline",
                "Derived: Low / Medium / High conductivity",
                "Derived: Compliant / Non-Compliant (BIS/WHO)",
            ],
            "Unit / Type": [
                "Integer", "Text", "Text", "°C", "mg/l", "–",
                "µhos/cm", "mg/l", "mg/l",
                "MPN/100ml", "MPN/100ml",
                "Year", "Datetime",
                "Category", "Category", "Category",
            ],
        }
    )
    st.dataframe(col_desc, use_container_width=True, hide_index=True)

    # ---- Summary statistics ----
    st.subheader("📊 Summary Statistics")
    num_df = df.select_dtypes(include=np.number)
    st.dataframe(num_df.describe().T.round(3), use_container_width=True)

    # ---- Missing value analysis ----
    st.subheader("🕳️ Missing Value Analysis (Original Dataset)")
    raw_df = pd.read_csv("water_dataX .csv")
    raw_df.columns = (
        raw_df.columns.str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    # Replace string NANs  
    raw_df.replace("NAN", np.nan, inplace=True)
    raw_df.replace("nan", np.nan, inplace=True)

    missing = (
        raw_df.isnull().sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Count"})
    )
    missing["Missing %"] = (missing["Missing Count"] / len(raw_df) * 100).round(2)
    missing = missing[missing["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    if missing.empty:
        st.success("✅ No missing values found in the dataset after initial load.")
    else:
        fig_miss = px.bar(
            missing,
            x="Column",
            y="Missing %",
            color="Missing %",
            color_continuous_scale="Reds",
            title="Missing Value Percentage by Column",
            labels={"Missing %": "Missing (%)"},
            text="Missing %",
        )
        fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_miss.update_layout(showlegend=False, height=380,
                               plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_miss, use_container_width=True)
        st.dataframe(missing, use_container_width=True, hide_index=True)

    # ---- Download processed data ----
    st.subheader("⬇️ Download Processed Data")
    with open("processed_water_data.csv", "rb") as fh:
        st.download_button(
            label="📥 Download processed_water_data.csv",
            data=fh,
            file_name="processed_water_data.csv",
            mime="text/csv",
        )


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – INTERACTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

elif nav == SECTIONS[2]:
    section_header("📊", "Interactive Dashboard")

    if dff.empty:
        st.warning("No data matches the selected filters. Please adjust sidebar filters.")
        st.stop()

    # ==== KPI METRICS ====
    st.markdown("#### 📌 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)

    avg_ph = dff["ph"].mean() if "ph" in dff.columns else 0
    avg_ec = dff["conductivity"].mean() if "conductivity" in dff.columns else 0
    avg_do = dff["do_mg_l"].mean() if "do_mg_l" in dff.columns else 0
    pct_nc = (
        (dff["compliance_status"] == "Non-Compliant").sum() / len(dff) * 100
        if "compliance_status" in dff.columns
        else 0
    )

    for col, lbl, val, sub, color in zip(
        [k1, k2, k3, k4],
        ["Avg. pH", "Avg. EC (µhos/cm)", "Avg. DO (mg/l)", "% Non-Compliant"],
        [f"{avg_ph:.2f}", f"{avg_ec:.0f}", f"{avg_do:.2f}", f"{pct_nc:.1f}%"],
        ["Safe range: 6.5–8.5", "Low<250 / Med 250-750 / Hi>750",
         "Healthy ≥ 5 mg/l", "BIS/WHO standards"],
        ["#1565c0", "#2e7d32", "#00838f", "#c62828"],
    ):
        col.markdown(
            f"""
            <div style="background:white;border-left:6px solid {color};
                        border-radius:10px;padding:16px 20px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.09);text-align:center;">
                <div style="font-size:12px;color:#546e7a;text-transform:uppercase;
                            letter-spacing:1px;font-weight:600;">{lbl}</div>
                <div style="font-size:34px;font-weight:800;color:{color};line-height:1.1;">{val}</div>
                <div style="font-size:11px;color:#90a4ae;margin-top:4px;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 1 – pH Trend over Years ====
    st.subheader("📈 Chart 1: pH Trend Over Years")
    ph_trend = (
        dff.groupby("year")["ph"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "Avg pH", "min": "Min pH", "max": "Max pH"})
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ph_trend["year"], y=ph_trend["Max pH"],
        fill=None, mode="lines", line=dict(color="rgba(21,101,192,0.2)"),
        name="Max pH", showlegend=True,
    ))
    fig1.add_trace(go.Scatter(
        x=ph_trend["year"], y=ph_trend["Min pH"],
        fill="tonexty", mode="lines", line=dict(color="rgba(21,101,192,0.2)"),
        name="Min pH", fillcolor="rgba(21,101,192,0.12)",
    ))
    fig1.add_trace(go.Scatter(
        x=ph_trend["year"], y=ph_trend["Avg pH"],
        mode="lines+markers", line=dict(color="#1565c0", width=3),
        marker=dict(size=8, color="#1565c0"), name="Avg pH",
    ))
    fig1.add_hline(y=6.5, line_dash="dash", line_color="#e53935",
                   annotation_text="BIS Lower (6.5)", annotation_position="bottom right")
    fig1.add_hline(y=8.5, line_dash="dash", line_color="#e53935",
                   annotation_text="BIS Upper (8.5)", annotation_position="top right")
    fig1.update_layout(
        title="Average pH Trend Over Years (with Range Band)",
        xaxis_title="Year", yaxis_title="pH",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)
    insight(
        "pH remained mostly within the BIS safe range of 6.5–8.5 across years. "
        "Spikes indicate localised acid/alkaline contamination events."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 2 – EC Bar Chart (location-wise) ====
    st.subheader("📊 Chart 2: Average Conductivity (EC) by State")
    ec_state = (
        dff.groupby("state")["conductivity"]
        .mean()
        .reset_index()
        .rename(columns={"conductivity": "Avg EC"})
        .sort_values("Avg EC", ascending=False)
        .head(20)
    )
    fig2 = px.bar(
        ec_state, x="Avg EC", y="state", orientation="h",
        color="Avg EC", color_continuous_scale="Teal",
        labels={"Avg EC": "Avg EC (µhos/cm)", "state": "State"},
        title="Top States by Average Electrical Conductivity",
        text="Avg EC",
    )
    fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig2.add_vline(x=250, line_dash="dot", line_color="#43a047",
                   annotation_text="Low/Medium boundary (250)")
    fig2.add_vline(x=750, line_dash="dot", line_color="#e53935",
                   annotation_text="Medium/High boundary (750)")
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(autorange="reversed"), height=520,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
    insight(
        "States with high EC indicate elevated dissolved salts — possibly from industrial "
        "discharge or natural geology. Values above 750 µhos/cm warrant further investigation."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 3 – Compliance Pie Chart ====
    st.subheader("🥧 Chart 3: Compliance vs Non-Compliance (BIS/WHO)")
    comp_counts = dff["compliance_status"].value_counts().reset_index()
    comp_counts.columns = ["Status", "Count"]
    fig3 = px.pie(
        comp_counts, names="Status", values="Count",
        color="Status",
        color_discrete_map={"Compliant": "#2e7d32", "Non-Compliant": "#c62828"},
        title="Water Sample Compliance with BIS/WHO Standards",
        hole=0.42,
    )
    fig3.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",
        pull=[0.03, 0.03],
    )
    fig3.update_layout(
        height=420,
        legend=dict(orientation="h", y=-0.1),
        annotations=[dict(text="Compliance", x=0.5, y=0.5,
                          font_size=14, showarrow=False)],
    )
    st.plotly_chart(fig3, use_container_width=True)
    warning_note(
        f"{pct_nc:.1f}% of samples are Non-Compliant. High BOD, low DO, or "
        "elevated fecal coliforms are the most common reasons for non-compliance."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 4 – Correlation Heatmap ====
    st.subheader("🌡️ Chart 4: Parameter Correlation Heatmap")
    heat_cols = [c for c in ["ph", "do_mg_l", "bod_mg_l", "conductivity",
                              "nitrate_mg_l", "fecal_coliform", "total_coliform", "temp"]
                 if c in dff.columns]
    corr_df = dff[heat_cols].corr().round(2)

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f", ax=ax4,
        cmap="RdYlGn", center=0, linewidths=0.6,
        linecolor="#eeeeee",
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8},
    )
    ax4.set_title("Pearson Correlation Between Water Quality Parameters",
                  fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    insight(
        "BOD and Fecal Coliform are typically strongly correlated — both are driven by "
        "organic / sewage inputs. Negative DO–BOD correlation indicates oxygen depletion."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 5 – India Map ====
    st.subheader("🗺️ Chart 5: India Map — State-wise Average Water Quality")

    map_param = st.selectbox(
        "Select parameter to visualise on map",
        ["ph", "conductivity", "do_mg_l", "bod_mg_l", "fecal_coliform"],
        format_func=lambda x: {
            "ph": "pH",
            "conductivity": "Conductivity (EC)",
            "do_mg_l": "Dissolved Oxygen",
            "bod_mg_l": "BOD",
            "fecal_coliform": "Fecal Coliform",
        }.get(x, x),
        key="map_param",
    )

    map_data = (
        dff.groupby("state")[[map_param, "lat", "lon"]]
        .agg({map_param: "mean", "lat": "first", "lon": "first"})
        .reset_index()
        .dropna(subset=["lat", "lon"])
    )
    map_data.rename(columns={map_param: "value"}, inplace=True)
    map_data["value_fmt"] = map_data["value"].round(2)

    fig5 = px.scatter_mapbox(
        map_data, lat="lat", lon="lon", size="value",
        color="value",
        color_continuous_scale="RdYlGn_r" if map_param in ["bod_mg_l", "fecal_coliform"] else "RdYlGn",
        hover_name="state",
        hover_data={"value_fmt": True, "lat": False, "lon": False},
        size_max=45,
        zoom=4,
        center={"lat": 22.5, "lon": 82.0},
        title=f"State-wise Average {map_param.upper()} Across India",
        labels={"value": map_param, "value_fmt": "Avg Value"},
        mapbox_style="carto-positron",
        height=560,
    )
    fig5.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig5, use_container_width=True)
    insight(
        "Bubble size and colour indicate the severity of the selected parameter. "
        "Darker / larger bubbles represent worse water quality for that indicator."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 6 – BOD Distribution by pH Status ====
    st.subheader("🎻 Chart 6: BOD Distribution by pH Status")
    fig6 = px.violin(
        dff, x="ph_status", y="bod_mg_l",
        color="ph_status",
        box=True, points="outliers",
        color_discrete_map={"Acidic": "#ef5350", "Neutral": "#42a5f5", "Alkaline": "#66bb6a"},
        title="BOD Distribution Across pH Status Categories",
        labels={"bod_mg_l": "BOD (mg/l)", "ph_status": "pH Status"},
        category_orders={"ph_status": ["Acidic", "Neutral", "Alkaline"]},
        height=440,
    )
    fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)
    insight(
        "Acidic waters tend to have higher BOD — indicating greater organic pollution "
        "load in acidic environments, often from industrial discharge."
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==== CHART 7 – Anomaly Detection ====
    st.subheader("🔴 Chart 7: Anomaly Detection (Z-score ≥ 3σ)")
    anomaly_param = st.selectbox(
        "Select parameter for anomaly view",
        [c for c in ["ph", "conductivity", "bod_mg_l", "do_mg_l", "fecal_coliform"]
         if c in dff.columns],
        key="anomaly_param",
    )
    anom_col = f"{anomaly_param}_anomaly"
    if anom_col not in dff.columns:
        dff = detect_anomalies(dff, anomaly_param)

    fig7 = px.scatter(
        dff, x="year", y=anomaly_param,
        color=anom_col,
        color_discrete_map={True: "#c62828", False: "#1565c0"},
        symbol=anom_col,
        symbol_map={True: "x", False: "circle"},
        opacity=0.7,
        title=f"Anomaly Detection — {anomaly_param} (red × = anomaly, |z| > 3)",
        labels={anomaly_param: anomaly_param.upper(),
                "year": "Year", anom_col: "Anomaly"},
        hover_data=["state", "location"],
        height=440,
    )
    fig7.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    n_anomalies = dff[anom_col].sum()
    st.plotly_chart(fig7, use_container_width=True)
    warning_note(
        f"{n_anomalies} anomalous data points detected for {anomaly_param} "
        f"({n_anomalies/len(dff)*100:.1f}% of filtered records). "
        "These represent values more than 3 standard deviations from the mean."
    )

    # ---- Download filtered data ----
    st.markdown("<br>", unsafe_allow_html=True)
    csv_buf = io.StringIO()
    dff.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="filtered_water_data.csv",
        mime="text/csv",
    )


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – GOVERNMENT INITIATIVES
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

elif nav == SECTIONS[3]:
    section_header("🏛️", "Government Initiatives & Organisations")

    initiatives = [
        {
            "icon": "🌊",
            "title": "Namami Gange Programme (NMCG)",
            "ministry": "Ministry of Jal Shakti | Budget: ₹20,000 Cr | Launched: 2015",
            "color": "#1565c0",
            "points": [
                "Flagship mission for comprehensive conservation and rejuvenation of River Ganga.",
                "Covers sewage treatment plant (STP) construction with total capacity of 5,000+ MLD.",
                "Afforestation of 30,000 hectares along the Ganga basin.",
                "Ghat development and beautification across 97 towns.",
                "Real-time water quality monitoring via online continuous monitoring systems (OCMS).",
                "Industrial units prohibited from discharging untreated effluents.",
                "Biodiversity conservation — Gangetic Dolphin as flagship species.",
            ],
        },
        {
            "icon": "🚰",
            "title": "Jal Jeevan Mission (JJM)",
            "ministry": "Ministry of Jal Shakti | Budget: ₹3.60 Lakh Cr | Launched: 2019",
            "color": "#2e7d32",
            "points": [
                "Har Ghar Jal — tap water connection to every rural household by 2024.",
                "As of 2025, over 14 crore (140 million) households connected.",
                "Establishes 5-member Village Water & Sanitation Committees (VWSCs).",
                "Water quality testing labs at district and sub-district levels.",
                "Field testing kits (FTKs) distributed to grassroots workers (Swajal Sahayikas).",
                "Source sustainability through aquifer management and rainwater harvesting.",
                "Monitoring through JJM Dashboard and IoT-based sensors.",
            ],
        },
        {
            "icon": "🔬",
            "title": "National Water Quality Monitoring Programme (NWQMP – CPCB)",
            "ministry": "CPCB, MoEFCC | Est: 1978 | Stations: 2,500+",
            "color": "#00838f",
            "points": [
                "Monitors 507 rivers, 154 lakes, 25 tanks, 17 ponds, 45 reservoirs and 10 creeks.",
                "2,500+ monitoring stations across 28 states and 7 UTs.",
                "Parameters monitored: temperature, pH, DO, BOD, coliforms, heavy metals, pesticides.",
                "Monthly / quarterly sampling by State Pollution Control Boards (SPCBs).",
                "Data published annually through CPCB reports and India-WRIS portal.",
                "Basis for river classification (Class A to E) under BIS IS:2296.",
            ],
        },
        {
            "icon": "🏗️",
            "title": "State Pollution Control Boards (SPCBs)",
            "ministry": "Constituted under Water Act, 1974",
            "color": "#6a1b9a",
            "points": [
                "Each state has its own SPCB / Pollution Control Committee (PCC).",
                "Issue Consents to Establish (CTE) and Consents to Operate (CTO) to industries.",
                "Enforce effluent discharge standards (ZLD norms for water-intensive sectors).",
                "Monitor common effluent treatment plants (CETPs) in industrial clusters.",
                "Coordinate with CPCB on national programs and data submission.",
                "Impose closure orders and penalties for violations under EP Act, 1986.",
            ],
        },
        {
            "icon": "⚖️",
            "title": "Water (Prevention & Control of Pollution) Act, 1974",
            "ministry": "Parliament of India | Amended: 1988",
            "color": "#c62828",
            "points": [
                "First comprehensive legislation to prevent and control water pollution in India.",
                "Established CPCB at the central level and SPCBs at state level.",
                "Prohibits discharge of polluting matter into streams, wells, sewers or land.",
                "Section 25/26: Any new/existing industrial discharge requires SPCB consent.",
                "Penalties: imprisonment up to 6 years + fine for violations.",
                "Amended in 1988 to enhance penalty provisions.",
                "Supplemented by the Environment Protection Act, 1986 for hazardous substances.",
            ],
        },
        {
            "icon": "🌿",
            "title": "Atal Bhujal Yojana (ABY)",
            "ministry": "Ministry of Jal Shakti | Budget: ₹6,000 Cr | Launched: 2020",
            "color": "#558b2f",
            "points": [
                "Community-led groundwater management in 7 water-stressed states.",
                "States: Gujarat, Haryana, Karnataka, Madhya Pradesh, Maharashtra, Rajasthan, UP.",
                "Focuses on water budgeting at Gram Panchayat level.",
                "Promotes demand-side management and groundwater recharge.",
                "Incentive-based framework — states rewarded for reducing extraction.",
            ],
        },
    ]

    for idx, item in enumerate(initiatives):
        with st.expander(f"{item['icon']}  {item['title']}", expanded=(idx == 0)):
            st.markdown(
                f"""
                <div style="background:#f5f5f5;border-radius:8px;padding:8px 14px;
                            margin-bottom:10px;font-size:13px;color:#546e7a;">
                    📌 <b>{item['ministry']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for point in item["points"]:
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:flex-start;margin:6px 0;">
                        <span style="color:{item['color']};font-size:18px;margin-right:10px;
                                     line-height:1.4;">▸</span>
                        <span style="font-size:14.5px;color:#37474f;line-height:1.6;">{point}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---- Timeline ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📅 Policy Timeline")
    timeline_data = pd.DataFrame(
        {
            "Year": [1974, 1978, 1986, 1991, 2011, 2015, 2019, 2020, 2022],
            "Event": [
                "Water Act enacted",
                "NWQMP launched by CPCB",
                "Environment Protection Act",
                "Ganga Action Plan II",
                "National Water Policy revised",
                "Namami Gange Programme",
                "Jal Jeevan Mission launched",
                "Atal Bhujal Yojana",
                "JJM — Urban extended",
            ],
            "Category": [
                "Legislation", "Monitoring", "Legislation",
                "River Cleaning", "Policy",
                "River Cleaning", "Rural Water Supply",
                "Groundwater", "Urban Water Supply",
            ],
        }
    )
    color_map = {
        "Legislation": "#c62828",
        "Monitoring": "#1565c0",
        "River Cleaning": "#2e7d32",
        "Policy": "#6a1b9a",
        "Rural Water Supply": "#00838f",
        "Groundwater": "#558b2f",
        "Urban Water Supply": "#0277bd",
    }
    fig_tl = px.scatter(
        timeline_data, x="Year", y="Category",
        color="Category", size=[15] * len(timeline_data),
        hover_data=["Event"],
        color_discrete_map=color_map,
        title="Indian Water Policy & Initiative Timeline",
        height=380,
    )
    fig_tl.update_traces(marker=dict(symbol="diamond", size=16))
    fig_tl.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=5),
    )
    for _, row in timeline_data.iterrows():
        fig_tl.add_annotation(
            x=row["Year"], y=row["Category"],
            text=row["Event"], showarrow=True,
            arrowhead=2, ax=0, ay=-30,
            font=dict(size=10, color="#37474f"),
            bgcolor="white", bordercolor="#bdbdbd", borderwidth=1,
        )
    st.plotly_chart(fig_tl, use_container_width=True)


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

elif nav == SECTIONS[4]:
    section_header("✅", "Conclusion & Key Insights")

    # ---- Data insights ----
    st.subheader("🔍 Data Insights")
    ci1, ci2 = st.columns(2)

    with ci1:
        st.markdown(
            """
            <div style="background:white;border-radius:12px;padding:20px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.09);">
            <h4 style="color:#1565c0;">📊 Key Statistical Findings</h4>
            <ul style="line-height:2.0;color:#37474f;font-size:14px;">
                <li><b>pH</b> values generally within BIS safe range, but acidic pockets exist in industrial belts.</li>
                <li><b>BOD</b> exceeds the 3 mg/l limit in a significant portion of samples — indicating untreated sewage outflow.</li>
                <li><b>Fecal Coliform</b> is the #1 cause of non-compliance, often 100× above safe limits near urban centres.</li>
                <li><b>Dissolved Oxygen</b> shows a declining trend in densely populated river stretches.</li>
                <li><b>Conductivity</b> is highest in Rajasthan and Gujarat — linked to saline geology and industrial effluents.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ci2:
        st.markdown(
            """
            <div style="background:white;border-radius:12px;padding:20px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.09);">
            <h4 style="color:#c62828;">⚕️ Health Implications</h4>
            <ul style="line-height:2.0;color:#37474f;font-size:14px;">
                <li>High fecal coliform levels directly correlate with diarrhoeal disease burden.</li>
                <li>Low DO in river water threatens fisheries — a critical protein source for riparian communities.</li>
                <li>High BOD depletes oxygen, creating "dead zones" in rivers and lakes.</li>
                <li>Nitrate contamination in groundwater poses methemoglobinaemia risk for infants in agricultural areas.</li>
                <li>Long-term exposure to heavy metals (often co-occurring with high EC) can cause kidney damage and cancer.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Technology role ----
    st.subheader("💻 Role of Technology & Data Science in Water Governance")

    tech_items = [
        ("🛰️", "Remote Sensing & GIS",
         "Satellite imagery detects algal blooms, turbidity changes and illegal discharge plumes "
         "across large river basins without physical access."),
        ("📡", "IoT & Real-time Sensors",
         "Online Continuous Monitoring Systems (OCMS) installed on rivers and STPs transmit "
         "live pH, DO, BOD and turbidity data to cloud dashboards."),
        ("🤖", "Machine Learning & AI",
         "Predictive models trained on historical water-quality data can forecast pollution events, "
         "enabling proactive intervention by pollution control authorities."),
        ("📊", "Open Data & Dashboards",
         "Platforms like India-WRIS, CPCB data portal and apps like this one enable citizens, "
         "researchers and policymakers to visualise and act on water-quality information."),
        ("⚗️", "Advanced Treatment Tech",
         "Membrane bioreactors (MBR), UV disinfection, and nano-filtration are being deployed "
         "in Namami Gange STPs to achieve near-zero pollutant discharge."),
    ]

    cols_tech = st.columns(len(tech_items))
    for col, (icon, title, desc) in zip(cols_tech, tech_items):
        col.markdown(
            f"""
            <div style="background:white;border-radius:12px;padding:16px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.09);text-align:center;height:100%;">
                <div style="font-size:36px;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700;color:#1565c0;font-size:13px;
                            margin-bottom:8px;">{title}</div>
                <div style="font-size:12px;color:#546e7a;line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Recommendations ----
    st.subheader("📋 Recommendations")
    with st.expander("View Detailed Recommendations", expanded=True):
        recs = [
            ("Short-term (0–2 years)",
             "#c62828",
             ["Deploy portable water testing kits in all gram panchayats.",
              "Strict enforcement of ZLD (Zero Liquid Discharge) for textile, leather, paper mills.",
              "Commission bio-remediation pilots on highly polluted river stretches.",
              "Digitise all SPCB complaint data for public access."]),
            ("Medium-term (2–5 years)",
             "#e65100",
             ["Expand OCMS coverage to all Class I & II towns on major rivers.",
              "Integrate ML-based early-warning systems for pollution events.",
              "Scale up faecal-sludge management and decentralised STPs for small towns.",
              "Strengthen SPCBs with trained data scientists and GIS professionals."]),
            ("Long-term (5+ years)",
             "#2e7d32",
             ["Achieve 100% sewage treatment before river outfall in all urban areas.",
              "Transition to a circular water economy — treated wastewater for irrigation.",
              "Establish a national open-access water quality API for researchers.",
              "Mainstream water literacy in school curriculum."]),
        ]
        for period, color, points in recs:
            st.markdown(
                f'<h5 style="color:{color};">▸ {period}</h5>', unsafe_allow_html=True
            )
            for p in points:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp; • {p}")
            st.markdown("<br>", unsafe_allow_html=True)

    # ---- Final message ----
    st.markdown(
        """
        <div style="background: linear-gradient(135deg,#e3f2fd,#e8f5e9);
                    border-radius:14px;padding:24px 28px;margin-top:10px;
                    border:1px solid #b3e5fc;">
            <h3 style="color:#1565c0;margin-top:0;">
                💧 Water is Life — Monitor it. Protect it. Sustain it.
            </h3>
            <p style="color:#37474f;font-size:15px;line-height:1.8;margin:0;">
                Access to safe drinking water is a <b>Fundamental Right</b> (enshrined under
                Article 21 of the Indian Constitution by the Supreme Court). Data-driven
                governance, multi-stakeholder collaboration, and community participation are
                not optional — they are essential prerequisites for achieving
                <b>SDG Goal 6: Clean Water & Sanitation</b> by 2030. This dashboard is a
                small step toward making water-quality data accessible and actionable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER  (shown on all pages)
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
        💧 <b>Water Pollution, Human Health &amp; Government Initiatives</b>
        &nbsp;|&nbsp; EVS Dashboard Project
        &nbsp;|&nbsp; Built with Python &amp; Streamlit
        &nbsp;|&nbsp; Data: CPCB National Water Quality Monitoring Programme
        <br>
        <span style="opacity:0.8;font-size:12px;">
            Developed for academic presentation &nbsp;•&nbsp; 2025–26
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
