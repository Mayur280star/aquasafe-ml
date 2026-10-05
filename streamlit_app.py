"""AquaSafe-ML as a Streamlit app, for free hosting on Streamlit Community Cloud.

    streamlit run streamlit_app.py

Same models and prediction code as the FastAPI service (aquasafe.predictor for
Dataset 1, aquasafe.safety for Dataset 2); only the interface differs.
"""

import io
import json
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import streamlit as st

from aquasafe import config, schema
from aquasafe.predictor import Predictor
from aquasafe.safety import SafetyPredictor

st.set_page_config(page_title="AquaSafe-ML", page_icon="💧", layout="wide")

TEAM = ("Riddhesh V. Budha (A 712) · Mayur D. Chaudhari (A 714) · "
        "Yashodeep Y. Chaudhary (A 715) · Hrishita Chavan (A 717)")
GROUP_TEXT = {"default": "default", "tuned": "tuned (Optuna)", "ensemble": "ensemble"}


@st.cache_resource(show_spinner="Loading the Dataset 1 model…")
def potability():
    return Predictor.load()


@st.cache_resource(show_spinner="Loading the 13 Dataset 2 models…")
def safety():
    return SafetyPredictor.load()


def pct(v):
    """Percentage rounded half up (0.9725 -> 97.3%), matching the report."""
    return f"{Decimal(str(round(v * 100, 9))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def example_picker(label, examples, key):
    """Selectbox of examples; returns (index or None, example or None)."""
    names = ["—"] + [e["name"] for e in examples]
    choice = st.selectbox(label, range(len(names)), format_func=lambda i: names[i], key=key)
    return (choice, examples[choice - 1]) if choice else (0, None)


def number_inputs(fields, defaults, columns, prefix, version):
    """One number input per field. The key includes the example index, so picking an
    example resets every input to that example's values."""
    values, cols = {}, st.columns(columns)
    for i, (name, label, help_text, placeholder) in enumerate(fields):
        with cols[i % columns]:
            values[name] = st.number_input(label, value=defaults.get(name), placeholder=placeholder,
                                           help=help_text, format="%.4f", key=f"{prefix}-{name}-{version}")
    return values


def show_issues(issues):
    for i in issues:
        {"error": st.error, "warning": st.warning}.get(i["level"], st.info)(i["message"])


# ------------------------------------------------------------------ header

st.title("💧 AquaSafe-ML")
st.caption("Drinking-water potability and safety screening with machine learning · B.E. SEM-VII Machine Learning "
           f"mini-project, Rajiv Gandhi Institute of Technology, Mumbai · {TEAM}")

tab1, tab2 = st.tabs(["Dataset 1 · Potability (9 readings)", "Dataset 2 · Water safety (20 readings, 13 models)"])

# --------------------------------------------------------------- dataset 1

with tab1:
    P = potability()
    info = P.info()
    m = info["metrics"]
    test, ci, three = m["test_at_decision"], m["test_95ci"], m["test_three_way"]
    c = st.columns(4)
    c[0].metric("Test accuracy", pct(test["accuracy"]), help=f"95% CI {pct(ci['accuracy'][0])}–{pct(ci['accuracy'][1])}",
                delta=f"baseline {pct(m['majority_baseline'])}", delta_color="off")
    c[1].metric("Test ROC AUC", f"{test['roc_auc']:.3f}")
    c[2].metric("Precision of 'potable' calls", pct(three["precision_of_potable_calls"]),
                delta=f"{three['n_potable_calls']} of 656 test samples", delta_color="off")
    c[3].metric("Accuracy when it decides", pct(three["accuracy_on_decided"]),
                delta=f"decides on {pct(three['coverage'])}", delta_color="off")
    st.caption(f"Model: {info['model']}")

    examples = json.loads((config.ARTIFACT_DIR / "examples.json").read_text())
    ranges = info["training_ranges"]
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Sample readings")
        idx, ex = example_picker("Load an example", examples, "d1-example")
        st.caption("Leave a reading empty if it is not available; it will be estimated from the others.")
        with st.form("d1-form"):
            fields = [(f.name, f"{f.label} ({f.unit})" if f.unit else f.label, f.description,
                       f"typical {ranges[f.name]['p0_5']:.3g} to {ranges[f.name]['p99_5']:.3g}") for f in schema.FEATURES]
            values = number_inputs(fields, ex["values"] if ex else {}, 2, "d1", idx)
            submitted = st.form_submit_button("Check this sample", type="primary", use_container_width=True)

    with right:
        if submitted or ex:
            r = P.predict(values)
            if not r["ok"]:
                st.error("**Cannot assess this sample.** Fix the readings below.")
            else:
                text = f"**{r['verdict_text']}** · probability of potable {pct(r['probability_potable'])}"
                {"potable": st.success, "not_potable": st.error}.get(r["verdict"], st.warning)(text)
                st.progress(r["probability_potable"], text=f"Probability of potable: {r['probability_potable']:.2f} "
                            f"(cuts: not potable < {r['thresholds']['not_potable']:.2f}, "
                            f"potable ≥ {r['thresholds']['potable']:.2f})")
            show_issues([i for i in r["issues"] if i["level"] != "info"])
            if r["ok"] and r.get("contributions"):
                st.markdown("**What moved this prediction** (change in probability if each reading were typical)")
                contrib = pd.DataFrame(r["contributions"]).set_index("label")["effect"]
                st.bar_chart(contrib, horizontal=True, height=260, color="#2a78d6")
            st.markdown("**Regulatory guideline check** (WHO / IS 10500, for reference)")
            g = pd.DataFrame(r["guidelines"])
            g["limit"] = [f"{a:g} – {b:g}" if a is not None else f"≤ {b:g}" for a, b in zip(g["min"], g["max"])]
            g["status"] = g["status"].map({"ok": "✅ within", "above": "🔴 above", "below": "🔴 below",
                                           "missing": "⚪ not measured"})
            st.dataframe(g[["label", "value", "limit", "unit", "status"]], hide_index=True, use_container_width=True,
                         column_config={"label": "Parameter", "value": st.column_config.NumberColumn("Reading", format="%.2f"),
                                        "limit": "Limit", "unit": "Unit", "status": "Status"})
        else:
            st.info("Pick an example or enter the readings, then press **Check this sample**.")

    with st.expander("Batch screening: upload a CSV"):
        st.caption("One sample per row, columns: " + ", ".join(schema.FEATURE_NAMES))
        upload = st.file_uploader("CSV file", type="csv", key="d1-batch")
        if upload is not None:
            frame = pd.read_csv(upload)
            lower = {c.lower(): c for c in frame.columns}
            frame = frame.rename(columns={lower[n.lower()]: n for n in schema.FEATURE_NAMES if n.lower() in lower})
            results = P.predict_batch(frame)
            out = frame.copy()
            out["probability_potable"] = [x.get("probability_potable") for x in results]
            out["verdict"] = [x.get("verdict", "rejected") for x in results]
            out["notes"] = ["; ".join(i["message"] for i in x["issues"] if i["level"] != "info") for x in results]
            counts = out["verdict"].value_counts()
            c = st.columns(4)
            for col, (key, label) in zip(c, [("potable", "Likely potable"), ("uncertain", "Uncertain → lab test"),
                                             ("not_potable", "Likely not potable"), ("rejected", "Rejected input")]):
                col.metric(label, int(counts.get(key, 0)))
            st.dataframe(out, use_container_width=True, hide_index=True)
            buffer = io.StringIO()
            out.to_csv(buffer, index=False)
            st.download_button("Download results as CSV", buffer.getvalue(), "aquasafe_predictions.csv", "text/csv")

# --------------------------------------------------------------- dataset 2

with tab2:
    S = safety()
    sinfo = S.info()
    board = pd.DataFrame(sinfo["leaderboard"])
    final = board.set_index("model").loc[sinfo["final_model"]]
    c = st.columns(4)
    c[0].metric("Final model accuracy", pct(final["accuracy"]), delta=f"baseline {pct(sinfo['baseline'])}",
                delta_color="off")
    c[1].metric("Precision of 'safe' calls", pct(final["precision"]),
                delta=f"{int(final['tp'])} of {int(final['tp'] + final['fp'])} correct", delta_color="off")
    c[2].metric("Recall of safe water", pct(final["recall"]),
                delta=f"{int(final['tp'])} of {int(final['tp'] + final['fn'])} found", delta_color="off")
    c[3].metric("F1 score", f"{final['f1']:.3f}", delta=f"{sinfo['final_model']}, cut {final['cut']:.2f}",
                delta_color="off")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Contaminant readings")
        idx, ex = example_picker("Load a held-out test sample", sinfo["examples"], "d2-example")
        st.caption('All 20 readings are needed. The help icon on each field shows its "dangerous if greater than" limit.')
        with st.form("d2-form"):
            fields = [(f["name"], f["label"], f"Dangerous if greater than {f['limit']:g}",
                       f"typical {f['p0_5']:.3g} to {f['p99_5']:.3g}") for f in sinfo["features"]]
            values = number_inputs(fields, ex["values"] if ex else {}, 4, "d2", idx)
            submitted = st.form_submit_button("Test with every model", type="primary", use_container_width=True)

    with right:
        if submitted or ex:
            r = S.predict(values)
            if not r["ok"]:
                st.error("**Cannot assess this sample.** Fix the readings below.")
                show_issues(r["issues"])
            else:
                text = (f"**{'Likely safe' if r['verdict'] == 'safe' else 'Likely unsafe'}** · {r['final_model']} "
                        f"(final model) · probability of safe {pct(r['probability_safe'])} · cut {r['cut']:.2f}")
                (st.success if r["verdict"] == "safe" else st.error)(text)
                if ex:
                    right_call = ex["actual"] == r["verdict"]
                    (st.info if right_call else st.warning)(
                        f"Actual label of this test sample: **{ex['actual']}** · final model is "
                        f"**{'correct' if right_call else 'wrong'}**")
                show_issues(r["issues"])
                st.markdown(f"**What every model says** · {r['votes_safe']} of {r['n_models']} say safe")
                rows = pd.DataFrame(r["models"])
                rows["model"] = [("★ " if f else "") + n for n, f in zip(rows["model"], rows["final"])]
                rows["group"] = rows["group"].map(GROUP_TEXT)
                rows["verdict"] = rows["verdict"].map({"safe": "🔵 safe", "unsafe": "🟠 unsafe"})
                cols = ["model", "group", "probability_safe", "cut", "verdict", "test_f1"]
                if ex:
                    rows["correct"] = ["✅" if v == ex["actual"] else "❌" for v in pd.DataFrame(r["models"])["verdict"]]
                    cols.insert(5, "correct")
                st.dataframe(rows[cols], hide_index=True, use_container_width=True, height=493, column_config={
                    "model": "Model", "group": "Type",
                    "probability_safe": st.column_config.ProgressColumn("P(safe)", min_value=0, max_value=1, format="%.3f"),
                    "cut": st.column_config.NumberColumn("Cut", format="%.2f"), "verdict": "Verdict",
                    "correct": "Correct?", "test_f1": st.column_config.NumberColumn("Test F1", format="%.3f")})
                with st.expander("Contaminant limit check"):
                    lim = pd.DataFrame(r["limits"]).sort_values("status")
                    lim["status"] = lim["status"].map({"above": "🔴 above limit", "ok": "✅ within limit"})
                    st.dataframe(lim[["label", "value", "limit", "status"]], hide_index=True, use_container_width=True,
                                 column_config={"label": "Contaminant", "value": "Reading", "limit": "Limit",
                                                "status": "Status"})
                    st.caption("For reference only; the labels in this dataset do not follow the limits exactly.")
        else:
            st.info("Load a held-out test sample or enter all 20 readings, then press **Test with every model**.")

    st.subheader("Model results on the test set")
    st.caption(f"Each of the {len(board)} models was fitted on the training split and scored once on the same "
               f"{sinfo['n_test']:,} held-out test samples. The final model was chosen by cross-validation on the "
               f"training split, not from this table. Always answering 'unsafe' scores {pct(sinfo['baseline'])} accuracy.")
    view = board.sort_values("f1", ascending=False).copy()
    view["model"] = [("★ " if n == sinfo["final_model"] else "") + n for n in view["model"]]
    view["group"] = view["group"].map(GROUP_TEXT)
    for k in ("accuracy", "precision", "recall"):
        view[k] = view[k] * 100
    st.dataframe(view[["model", "group", "cut", "accuracy", "precision", "recall", "f1", "roc_auc"]], hide_index=True,
                 use_container_width=True, height=36 * (len(view) + 1) + 3, column_config={
                     "model": "Model", "group": "Type", "cut": st.column_config.NumberColumn("Cut", format="%.2f"),
                     "accuracy": st.column_config.NumberColumn("Accuracy", format="%.1f%%"),
                     "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
                     "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
                     "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                     "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.3f")})
    st.caption("Dataset 2 is the Kaggle Water Quality dataset (mssmartypants): 7,996 samples, 11.4% safe. Its author "
               "describes it as imaginary data for educational use, so judge models by precision, recall and F1.")

st.divider()
st.caption("A screening aid, not a certified water test. Confirm any sample intended for drinking with an accredited "
           "laboratory.")
