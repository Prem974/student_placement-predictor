"""Run:  python -m streamlit run app.py   (after python train.py)
Background image venum na: oru image-a 'bg.jpg' nu indha folder la vai (1 MB kulla irundha nalladhu)."""
import os
import base64
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")


# ---------- background + styling ----------
@st.cache_data
def bg_image():
    for name, mime in [("bg.jpg", "image/jpeg"), ("bg.jpeg", "image/jpeg"),
                       ("bg.png", "image/png"), ("bg.webp", "image/webp")]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    return None


MAINS = ['[data-testid="stMainBlockContainer"]', "section.main .block-container"]


def m(suffix=""):
    return ", ".join(f"{s} {suffix}".strip() for s in MAINS)


img = bg_image()
if img:
    bg = f"linear-gradient(rgba(15,23,42,.45), rgba(15,23,42,.45)), url('{img}') center/cover fixed"
    icons_layer = ""
else:
    bg = "linear-gradient(135deg,#e0e7ff 0%,#fce7f3 50%,#cffafe 100%) fixed"
    pattern = "🎓 💼 🏢 📈 🚀 💻 🏆 📚 🎯 ⭐ 🤝 🧠 " * 40
    icons_layer = (".stApp::before{content:'" + pattern + "';position:fixed;inset:0;font-size:2.4rem;"
                   "line-height:4.6rem;letter-spacing:1.6rem;opacity:.16;overflow:hidden;"
                   "pointer-events:none;z-index:0;word-break:break-all;}")

CSS = f"""
<style>
.stApp{{background:{bg};}}
{icons_layer}
{m()}{{position:relative;z-index:1;background:rgba(255,255,255,.92);border-radius:22px;
  padding:2rem 2.2rem !important;margin-top:1rem;box-shadow:0 10px 40px rgba(0,0,0,.20);}}
{m('*')}{{color:#111827;}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#1e1b4b,#4c1d95);}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{{color:#fff !important;}}
{m('.hero')}{{background:linear-gradient(135deg,#4f46e5,#7c3aed,#db2777);padding:26px 32px;border-radius:18px;margin-bottom:18px;}}
{m('.hero h1')}{{margin:0;font-size:2.1rem;color:#fff !important;}}
{m('.hero p')}{{margin:6px 0 0;color:#e0e7ff !important;}}
{m('.stat')}{{border-radius:16px;padding:14px 16px;display:flex;align-items:center;gap:12px;box-shadow:0 4px 14px rgba(0,0,0,.15);}}
{m('.stat *')}{{color:#fff !important;}}
{m('.stat .ico')}{{font-size:2rem;}}
{m('.stat .lbl')}{{font-size:.78rem;opacity:.9;}}
{m('.stat .val')}{{font-size:1.3rem;font-weight:700;}}
{m('.card')}{{background:#fff;padding:22px;border-radius:16px;border:1px solid #e5e7eb;box-shadow:0 2px 10px rgba(0,0,0,.08);text-align:center;}}
{m('.donut')}{{width:210px;height:210px;border-radius:50%;margin:8px auto 14px;display:flex;align-items:center;justify-content:center;}}
{m('.donut-in')}{{width:156px;height:156px;border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
{m('.donut-in b')}{{font-size:2.3rem;line-height:1.1;}}
{m('.donut-in span')}{{font-size:.8rem;color:#6b7280;}}
{m('.badge')}{{display:inline-block;padding:7px 18px;border-radius:999px;font-weight:600;color:#fff !important;}}
{m('.tip')}{{background:#f5f3ff;border-left:5px solid #7c3aed;padding:10px 14px;border-radius:10px;margin-bottom:10px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- model ----------
@st.cache_resource
def load():
    return joblib.load("model.joblib")


art = load()
model, cols, schema, best_name = art["model"], art["columns"], art["schema"], art["name"]

ICONS = [("cgpa", "📚"), ("age", "🎂"), ("gender", "🧑"), ("branch", "🏫"), ("college", "🏛️"),
         ("intern", "💼"), ("project", "🛠️"), ("certif", "📜"), ("coding", "💻"), ("aptitude", "🧮"),
         ("communication", "🗣️"), ("logical", "🧠"), ("hackathon", "🏆"), ("github", "🐙"),
         ("linkedin", "🔗"), ("mock", "🎤"), ("attendance", "📅"), ("backlog", "⚠️"),
         ("extracurricular", "🎭"), ("leadership", "👑"), ("volunteer", "🤝"), ("sleep", "😴"),
         ("study", "📖")]


def icon(c):
    for k, e in ICONS:
        if k in c.lower():
            return e
    return "🔹"


def pretty(c):
    return c.replace("_", " ").title()


BRANCH_COL, EXTRA_BRANCH, PROXY_BRANCH = "branch", "AIDS", "CSE"


def fix(r):
    # AIDS dataset la illa, so model-ku CSE profile-a anuppurom
    if r.get(BRANCH_COL) == EXTRA_BRANCH:
        proxy = PROXY_BRANCH if PROXY_BRANCH in schema[BRANCH_COL]["options"] else schema[BRANCH_COL]["options"][0]
        return {**r, BRANCH_COL: proxy}
    return r


def proba(rows):
    df = pd.get_dummies(pd.DataFrame([fix(r) for r in rows])).reindex(columns=cols, fill_value=0).astype(float)
    return model.predict_proba(df)[:, 1]


# ---------- header ----------
acc = None
if os.path.exists("results.csv"):
    r = pd.read_csv("results.csv")
    r = r[r["Model"] == best_name]
    if len(r):
        acc = r.iloc[0]["Accuracy"] * 100

st.markdown(
    "<div class='hero'><h1>🎓 Student Placement Predictor 🚀</h1>"
    "<p>💼 Machine-learning based placement chance estimator for campus students</p></div>",
    unsafe_allow_html=True,
)

# ---------- sidebar inputs ----------
st.sidebar.title("🧑‍🎓 Student Profile")
st.sidebar.caption("Details fill panni, kela 🔮 Predict click pannu")
vals = {}
with st.sidebar:
    with st.form("profile"):
        for c, s in schema.items():
            label = f"{icon(c)} {pretty(c)}"
            if s["type"] == "cat":
                opts = list(s["options"])
                if c == BRANCH_COL and EXTRA_BRANCH not in opts:
                    opts.append(EXTRA_BRANCH)
                vals[c] = st.selectbox(label, opts)
                continue
            lo, hi, d = s["min"], s["max"], s["default"]
            if hi <= lo:
                vals[c] = lo
            elif lo.is_integer() and hi.is_integer() and d.is_integer():
                vals[c] = st.slider(label, int(lo), int(hi), int(d))
            else:
                vals[c] = st.slider(label, float(lo), float(hi), float(d), step=0.1 if hi - lo <= 20 else 1.0)
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    st.session_state["predicted"] = True
if not st.session_state.get("predicted"):
    st.info("👈 Left sidebar la student details fill panni **🔮 Predict** button click pannu")
    st.stop()

base = float(proba([vals])[0])
state = "high" if base >= 0.6 else "mid" if base >= 0.4 else "low"
color = {"high": "#16a34a", "mid": "#f59e0b", "low": "#dc2626"}[state]
verdict = {"high": "🎉 High chance", "mid": "👍 Moderate chance", "low": "📉 Needs improvement"}[state]

if st.session_state.get("state") != state:
    if state == "high":
        st.balloons()
    st.session_state["state"] = state


def stat(ico, label, val, c1, c2):
    return (f"<div class='stat' style='background:linear-gradient(135deg,{c1},{c2})'>"
            f"<div class='ico'>{ico}</div><div><div class='lbl'>{label}</div><div class='val'>{val}</div></div></div>")


s1, s2, s3, s4 = st.columns(4)
s1.markdown(stat("🎯", "Placement chance", f"{base * 100:.1f}%", color, "#334155"), unsafe_allow_html=True)
s2.markdown(stat("🤖", "Best model", best_name, "#4f46e5", "#7c3aed"), unsafe_allow_html=True)
s3.markdown(stat("✅", "Test accuracy", f"{acc:.1f}%" if acc else "-", "#0891b2", "#0e7490"), unsafe_allow_html=True)
s4.markdown(stat("🧩", "Features used", str(len(schema)), "#db2777", "#be185d"), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "🔀 What-if", "📊 Model Results", "ℹ️ About"])

# ---------- tab 1: prediction ----------
with tab1:
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f"<div class='card'><div class='donut' style='background:conic-gradient({color} {base * 100:.1f}%, #e5e7eb 0)'>"
            f"<div class='donut-in'><b style='color:{color}'>{base * 100:.1f}%</b><span>placement chance</span></div></div>"
            f"<span class='badge' style='background:{color}'>{verdict}</span></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("💡 Improvement tips")
        rows, meta = [], []
        for c, s in schema.items():
            if s["type"] != "num" or s["max"] <= s["min"]:
                continue
            step = (s["max"] - s["min"]) * 0.10
            up, dn = min(vals[c] + step, s["max"]), max(vals[c] - step, s["min"])
            rows += [{**vals, c: up}, {**vals, c: dn}]
            meta += [(c, "Increase", up), (c, "Decrease", dn)]
        probs = proba(rows) if rows else []
        tips = sorted(((p - base, c, a, v) for p, (c, a, v) in zip(probs, meta)), reverse=True)
        shown = [t for t in tips if t[0] > 0.002][:3]
        if shown:
            for gain, c, act, v in shown:
                st.markdown(
                    f"<div class='tip'>{icon(c)} <b>{act} {pretty(c)}</b> to {v:.1f} → chance "
                    f"<b>+{gain * 100:.1f}%</b> 📈</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Oru single change-um adhigam help pannala. Model-ku indha features romba strong illa.")
        st.caption("Idhu model-oda estimate mattum dhaan, guarantee illa.")
        if vals.get(BRANCH_COL) == EXTRA_BRANCH:
            st.info(f"ℹ️ Training dataset la AIDS branch illa, so prediction {PROXY_BRANCH} profile base pannirukku.")

# ---------- tab 2: what-if ----------
with tab2:
    num_cols = [c for c, s in schema.items() if s["type"] == "num" and s["max"] > s["min"]]
    f = st.selectbox("Edha feature maathanum?", num_cols, format_func=lambda c: f"{icon(c)} {pretty(c)}")
    s = schema[f]
    new = st.slider("Pudhu value", float(s["min"]), float(s["max"]), float(vals[f]))
    alt = float(proba([{**vals, f: new}])[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Current chance", f"{base * 100:.1f}%")
    m2.metric(f"With {pretty(f)} = {new:.1f}", f"{alt * 100:.1f}%", f"{(alt - base) * 100:+.1f}%")
    m3.metric("Current value", f"{float(vals[f]):.1f}")
    grid = [s["min"] + (s["max"] - s["min"]) * i / 24 for i in range(25)]
    curve = proba([{**vals, f: g} for g in grid]) * 100
    st.line_chart(pd.DataFrame({"Placement chance %": curve}, index=[round(g, 2) for g in grid]))
    st.caption(f"{pretty(f)} maara maara placement chance eppadi maarudhu nu graph 📈")

# ---------- tab 3: model results ----------
with tab3:
    if os.path.exists("results.csv"):
        st.subheader("🤖 Algorithm comparison")
        st.dataframe(pd.read_csv("results.csv"), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    for i, (fn, cap) in enumerate([("model_comparison.png", "📊 Model comparison"),
                                   ("confusion_matrix.png", "🧮 Confusion matrix"),
                                   ("feature_importance.png", "⭐ Feature importance"),
                                   ("eda.png", "🔍 EDA")]):
        if os.path.exists(fn):
            (c1 if i % 2 == 0 else c2).image(fn, caption=cap, use_container_width=True)

# ---------- tab 4: about ----------
with tab4:
    st.markdown(
        """
### 🎓 About this project
**Student Placement Prediction** uses machine learning to estimate a student's placement chance
from academic, skill and experience features.

**🔧 Pipeline:** data cleaning → encoding → scaling → train/test split → 5-fold CV → model comparison → best model → Streamlit app

**🤖 Algorithms compared:** Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost

**⚠️ Limitations:** Prediction accuracy is only slightly above the majority-class baseline,
so this tool is a demo and should not be used for real decisions.
"""
    )
