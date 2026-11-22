import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LaunderLens", layout="wide")
st.title("LaunderLens: alert review")
st.warning("Demo on simulated data (IBM AML, HI-Small). Model-based alerts, not proof of laundering. "
           "Early baseline results: one model, one seed, three test days.")

FILE = "explanations_top500.csv"   # put this file in the same folder as app.py


@st.cache_data
def load(path):
    df = pd.read_csv(path)
    # Pull sender, receiver, amount, currency and time out of the explanation text
    p = r"Transaction (\S+) -> (\S+), ([\d,\.]+) (.+?) on (\d{4}-\d\d-\d\d) (\d\d:\d\d)\."
    parts = df["explanation"].str.extract(p)
    df["from_account"], df["to_account"] = parts[0], parts[1]
    df["amount"], df["currency"], df["time"] = parts[2], parts[3], parts[5]
    # The three main reasons
    reasons = df["explanation"].str.extract(r"Main reasons: (.*?)\. (?:No other|It is)")[0]
    df["reasons"] = reasons.fillna("")
    # Reliability sentence (only for named shapes)
    df["reliability"] = df["explanation"].str.extract(r"(In our test, [^.]*\.(?: This is a weak signal\.)?)")[0].fillna("")
    return df.sort_values(["day", "rank_in_day"]).reset_index(drop=True)


df = load(FILE)
tab1, tab2, tab3 = st.tabs(["Alerts", "Results (evaluation)", "About"])

# ---------------- Tab 1: alert queue + detail ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    day = c1.selectbox("Day", sorted(df["day"].unique()))
    k = c2.slider("Alert budget (top k per day)", 50, 500, 100, step=50)
    shapes = ["All"] + sorted(df["group_shape"].unique())
    shape = c3.selectbox("Pattern shape", shapes)

    top = df[(df["day"] == day) & (df["rank_in_day"] <= k)]
    if shape != "All":
        top = top[top["group_shape"] == shape]

    if top.empty:
        st.info("No alerts for this selection.")
    else:
        st.caption(f"{len(top)} alerts shown")
        st.dataframe(
            top[["rank_in_day", "score", "from_account", "to_account", "amount", "currency",
                 "time", "group_shape", "group_size"]],
            use_container_width=True, hide_index=True)

        st.subheader("Alert detail")
        pick = st.selectbox("Choose an alert by rank", top["rank_in_day"])
        row = top[top["rank_in_day"] == pick].iloc[0]
        a, b = st.columns([1, 2])
        a.metric("Risk score", f"{row['score']:.3f}")
        a.metric("Rank of the day", int(row["rank_in_day"]))
        a.write(f"**Pattern shape:** {row['group_shape']}")
        a.write(f"**Linked alerts:** {int(row['group_size'])}")
        b.write("**Main reasons**")
        for r in [x.strip() for x in row["reasons"].split(";") if x.strip()]:
            b.write(f"- {r}")
        if row["reliability"]:
            b.write(f"**Reliability:** {row['reliability']}")
        st.info(row["explanation"])

# ---------------- Tab 2: metrics computed from the file ----------------
with tab2:
    st.subheader("Alert-budget results on the 3 test days")
    st.caption("Computed from the `label` column of the file (label = 1 means real laundering). "
               "Investigators would not see this column; it is used only for evaluation.")
    rows = []
    for d, g in df.groupby("day"):
        row = {"day": d}
        for kk in (100, 500):
            sub = g[g["rank_in_day"] <= kk]
            row[f"precision_at_{kk}"] = round(sub["label"].mean(), 3)
        rows.append(row)
    met = pd.DataFrame(rows)
    st.dataframe(met, hide_index=True)
    st.bar_chart(met.set_index("day"))
    st.caption("Precision@k = of the top k alerts of a day, the share that were real laundering. "
               "Recall@k is not computed here because the file holds only the top 500 alerts, "
               "not the total laundering per day. Ask your friends for that number.")

    st.subheader("Share of real laundering by pattern shape (top 500 per day)")
    shape_tbl = (df.groupby("group_shape")
                   .agg(alerts=("label", "size"), real_laundering=("label", "sum"),
                        share=("label", "mean"))
                   .round(2).reset_index().sort_values("alerts", ascending=False))
    st.dataframe(shape_tbl, hide_index=True)
    st.caption("SINGLE = alert not connected to any other alert. Fan-in is a weak signal.")

# ---------------- Tab 3: about ----------------
with tab3:
    st.markdown("""
**What this is:** a decision-support demo for investigators. It shows ranked alerts with reasons.

**Limitations**
- Data are simulated (IBM AML), not real bank or Indian data.
- Part of the performance comes from a simulator-specific feature (payment format).
- Early baseline: one seed, three test days.
- Wrongly flagging a customer causes real harm, so this supports humans and decides nothing.
""")