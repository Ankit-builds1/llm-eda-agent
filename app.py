# ============================================================
# LLM-Powered Automated EDA Agent
# Author: Ankit Dash | Centurion University | 2025
# Deploy: HuggingFace Spaces (Gradio)
# ============================================================

import os
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import gradio as gr
from openai import OpenAI
from fpdf import FPDF
from datetime import datetime

# ── NVIDIA API Setup ─────────────────────────────────────────
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")

client = OpenAI(
    base_url = NVIDIA_BASE_URL,
    api_key  = NVIDIA_API_KEY
)

# ============================================================
# STEP 1 — DATA PROFILER
# ============================================================

def profile_dataframe(df: pd.DataFrame) -> str:
    """
    Extracts a full statistical profile from any DataFrame.
    Returns a clean text summary to send to the LLM.
    """
    profile = []

    # Basic shape
    profile.append("=== DATASET OVERVIEW ===")
    profile.append(f"Total Rows    : {df.shape[0]}")
    profile.append(f"Total Columns : {df.shape[1]}")
    profile.append(f"Column Names  : {list(df.columns)}")

    # Data types
    profile.append("\n=== COLUMN DATA TYPES ===")
    for col, dtype in df.dtypes.items():
        profile.append(f"  {col:30s} → {str(dtype)}")

    # Missing values
    profile.append("\n=== MISSING VALUES ===")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    has_nulls   = False
    for col in df.columns:
        if null_counts[col] > 0:
            profile.append(
                f"  {col:30s} → {null_counts[col]} missing ({null_pct[col]}%)"
            )
            has_nulls = True
    if not has_nulls:
        profile.append("  No missing values found!")

    # Numeric stats
    profile.append("\n=== NUMERIC COLUMN STATISTICS ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().round(2)
        profile.append(stats.to_string())
    else:
        profile.append("  No numeric columns found.")

    # Categorical insights
    profile.append("\n=== CATEGORICAL COLUMN INSIGHTS ===")
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        unique_vals = df[col].nunique()
        top_vals    = df[col].value_counts().head(5).to_dict()
        profile.append(f"\n  Column   : {col}")
        profile.append(f"  Unique   : {unique_vals}")
        profile.append(f"  Top 5    : {top_vals}")

    # Duplicates
    profile.append("\n=== DUPLICATE ROWS ===")
    profile.append(f"  Duplicate Rows: {df.duplicated().sum()}")

    return "\n".join(profile)


# ============================================================
# STEP 2 — LLM INSIGHT GENERATOR
# ============================================================

def generate_llm_insights(data_profile: str,
                          dataset_name: str = "Dataset") -> str:
    """
    Sends data profile to NVIDIA LLM.
    Returns detailed EDA insights in plain English.
    """
    prompt = f"""
You are an expert Data Scientist performing Exploratory Data Analysis.

Below is a complete statistical profile of the '{dataset_name}' dataset.
Analyze it carefully and provide insights in this structure:

{data_profile}

1. DATASET SUMMARY
   - What kind of data is this?
   - What is the likely objective/use case?

2. DATA QUALITY ISSUES
   - Which columns have missing values and how serious?
   - Any columns that should be dropped?
   - Any data type issues?

3. KEY PATTERNS & INSIGHTS
   - Most interesting statistical observations?
   - Which numeric columns show high variance or skewness?
   - What do categorical distributions tell us?

4. POTENTIAL CORRELATIONS TO INVESTIGATE
   - Which column pairs are likely correlated?
   - What relationships would you prioritize?

5. ANOMALIES & OUTLIERS
   - Which columns likely contain outliers?
   - How might these affect analysis?

Be specific. Use actual column names and numbers from the profile.
Write in clear professional English for a business audience.
"""

    response = client.chat.completions.create(
        model    = "meta/llama-3.1-8b-instruct",
        messages = [
            {
                "role": "system",
                "content": "You are an expert data scientist who provides clear, accurate, and actionable EDA insights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens  = 1500,
        temperature = 0.3
    )
    return response.choices[0].message.content


# ============================================================
# STEP 3 — HYPOTHESIS GENERATOR
# ============================================================

def generate_hypotheses(data_profile: str,
                        llm_insights: str,
                        dataset_name: str = "Dataset") -> str:
    """
    Generates 5 testable business hypotheses with
    causal reasoning, confounders, and statistical tests.
    """
    prompt = f"""
You are a senior Data Scientist analyzing the '{dataset_name}' dataset.

DATASET PROFILE:
{data_profile}

INITIAL INSIGHTS:
{llm_insights}

Generate exactly 5 testable business hypotheses.
For each hypothesis provide:

HYPOTHESIS [N]:
- CLAIM          : Specific, measurable, testable statement
- TYPE           : Direct Effect / Proxy Variable / Confounded Relationship
- REASONING      : Data-driven reasoning using actual stats
- CONFOUNDERS    : Which variables might explain this relationship
- HOW TO TEST    : Exact statistical test + Python library
- EXPECTED RESULT: Quantified expected outcome
- BUSINESS IMPACT: Real-world decision this enables

Focus on CAUSAL reasoning, not just correlation.
Use actual column names and statistics from the profile.
"""

    response = client.chat.completions.create(
        model    = "meta/llama-3.1-8b-instruct",
        messages = [
            {
                "role": "system",
                "content": "You are a senior data scientist who generates precise, testable, and business-relevant hypotheses."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens  = 2000,
        temperature = 0.3
    )
    return response.choices[0].message.content


# ============================================================
# STEP 4 — AUTO VISUALIZATION ENGINE
# ============================================================

def generate_visualizations(df: pd.DataFrame,
                            output_dir: str) -> list:
    """
    Auto-generates charts based on column types.
    Returns list of saved PNG file paths.
    """
    chart_paths  = []
    numeric_cols = df.select_dtypes(
        include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(
        include=["object"]).columns.tolist()

    # Detect target column
    target = None
    for t in ["Survived", "survived",
              "SalePrice", "saleprice", "target"]:
        if t in df.columns:
            target = t
            break

    # 1. Histograms for numeric columns (max 6)
    for col in numeric_cols[:6]:
        use_color = (
            target and
            col != target and
            df[target].nunique() <= 5
        )
        fig = px.histogram(
            df, x=col,
            color  = target if use_color else None,
            title  = f"Distribution of {col}",
            opacity= 0.75,
            barmode= "overlay"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(output_dir, f"hist_{col}.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 2. Bar charts for categorical columns (max 3)
    for col in cat_cols[:3]:
        if df[col].nunique() <= 15:
            if target and target in df.columns:
                data = df.groupby(col)[target]\
                         .mean().reset_index()
                data.columns = [col, f"Avg {target}"]
                fig = px.bar(
                    data, x=col, y=f"Avg {target}",
                    title  = f"Avg {target} by {col}",
                    color  = f"Avg {target}",
                    color_continuous_scale = "RdYlGn"
                )
            else:
                counts = df[col]\
                           .value_counts().reset_index()
                counts.columns = [col, "Count"]
                fig = px.bar(
                    counts, x=col, y="Count",
                    title = f"Distribution of {col}",
                    color = "Count",
                    color_continuous_scale = "Blues"
                )
            fig.update_layout(template="plotly_white",
                              title_font_size=14)
            path = os.path.join(
                output_dir, f"bar_{col}.png")
            fig.write_image(path)
            chart_paths.append(path)

    # 3. Correlation heatmap
    if len(numeric_cols) >= 2:
        top_numeric = numeric_cols[:10]
        corr = df[top_numeric].corr().round(2)
        fig  = px.imshow(
            corr,
            text_auto          = True,
            color_continuous_scale = "RdBu_r",
            title              = "Correlation Heatmap"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(
            output_dir, "correlation_heatmap.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 4. Scatter plot (top 2 numeric vs target)
    if target and len(numeric_cols) >= 2:
        x_col = [c for c in numeric_cols
                 if c != target][0]
        fig = px.scatter(
            df, x=x_col, y=target,
            title  = f"{x_col} vs {target}",
            opacity= 0.6,
            trendline="ols"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(
            output_dir, f"scatter_{x_col}_{target}.png")
        fig.write_image(path)
        chart_paths.append(path)

    return chart_paths


# ============================================================
# STEP 5 — PDF REPORT GENERATOR
# ============================================================

class EDAReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(20, 20, 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10,
                  "  LLM-Powered Automated EDA Report",
                  fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10,
            f"Page {self.page_no()} | Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            align="C"
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(20, 20, 20)
        self.cell(0, 9, f"  {title}",
                  fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        clean = text.encode(
            "latin-1", "replace").decode("latin-1")
        self.multi_cell(0, 5, clean)
        self.ln(2)


def generate_pdf_report(dataset_name: str,
                        data_profile: str,
                        llm_insights: str,
                        hypotheses: str,
                        chart_paths: list,
                        output_dir: str) -> str:
    """
    Generates complete EDA PDF report.
    Returns path to the saved PDF file.
    """
    pdf = EDAReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover
    pdf.set_font("Helvetica", "B", 18)
    pdf.ln(4)
    pdf.cell(0, 12, f"{dataset_name} Dataset",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7,
             "Automated Exploratory Data Analysis Report",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 7,
             f"Generated: {datetime.now().strftime('%B %d, %Y')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Section 1
    pdf.section_title("1. Dataset Profile")
    pdf.body_text(data_profile)

    # Section 2
    pdf.add_page()
    pdf.section_title("2. AI-Generated EDA Insights")
    pdf.body_text(llm_insights)

    # Section 3
    pdf.add_page()
    pdf.section_title("3. Auto-Generated Visualizations")
    pdf.ln(2)
    for i, path in enumerate(chart_paths):
        if os.path.exists(path):
            name = os.path.basename(path)\
                     .replace(".png", "")\
                     .replace("_", " ").title()
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, f"Chart {i+1}: {name}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.image(path, x=15, w=175)
            pdf.ln(3)
            if (i + 1) % 2 == 0 and \
               i + 1 < len(chart_paths):
                pdf.add_page()

    # Section 4
    pdf.add_page()
    pdf.section_title("4. AI-Generated Business Hypotheses")
    pdf.body_text(hypotheses)

    out = os.path.join(
        output_dir, f"{dataset_name}_EDA_Report.pdf")
    pdf.output(out)
    return out


# ============================================================
# MASTER PIPELINE FUNCTION
# ============================================================

def run_eda_agent(csv_file, progress=gr.Progress()):
    """
    Master function called by Gradio UI.
    Runs full EDA pipeline on uploaded CSV.
    """
    if csv_file is None:
        return (
            "⚠️ Please upload a CSV file first.",
            None, None
        )

    try:
        progress(0.1, desc="Loading dataset...")
        df = pd.read_csv(csv_file)
        dataset_name = os.path.basename(csv_file)\
                         .replace(".csv", "")

        # Drop useless ID columns
        drop_cols = [
            c for c in df.columns
            if c.lower() in [
                "id", "passengerid", "name",
                "ticket", "cabin"
            ]
        ]
        df_clean = df.drop(
            columns=drop_cols, errors="ignore")

        progress(0.2, desc="Profiling data...")
        data_profile = profile_dataframe(df_clean)

        progress(0.4, desc="Generating AI insights...")
        llm_insights = generate_llm_insights(
            data_profile, dataset_name)

        progress(0.6, desc="Generating hypotheses...")
        hypotheses = generate_hypotheses(
            data_profile, llm_insights, dataset_name)

        progress(0.75, desc="Creating visualizations...")
        charts_dir = tempfile.mkdtemp()
        chart_paths = generate_visualizations(
            df_clean, charts_dir)

        progress(0.9, desc="Generating PDF report...")
        pdf_path = generate_pdf_report(
            dataset_name = dataset_name,
            data_profile = data_profile,
            llm_insights = llm_insights,
            hypotheses   = hypotheses,
            chart_paths  = chart_paths,
            output_dir   = charts_dir
        )

        progress(1.0, desc="Done!")

        # Build output text
        output_text = f"""✅ Dataset: {dataset_name}
📊 Shape  : {df.shape[0]} rows × {df.shape[1]} columns
⏱️ Status : Analysis Complete!

{'='*60}
📋 DATASET PROFILE
{'='*60}
{data_profile}

{'='*60}
🤖 AI-GENERATED INSIGHTS
{'='*60}
{llm_insights}

{'='*60}
🧠 BUSINESS HYPOTHESES
{'='*60}
{hypotheses}
"""
        return output_text, chart_paths, pdf_path

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title = "LLM-Powered EDA Agent",
    theme = gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🤖 LLM-Powered Automated EDA Agent
### Upload any CSV → Instant AI insights, charts & PDF report
---
**Powered by:** NVIDIA API (Llama 3.1) + Plotly + Gradio  
**Built by:** Ankit Dash | Centurion University | 2025
    """)

    with gr.Row():
        with gr.Column(scale=1):
            csv_input   = gr.File(
                label      = "📁 Upload Your CSV File",
                file_types = [".csv"]
            )
            analyze_btn = gr.Button(
                "🔍 Analyze My Dataset",
                variant = "primary",
                size    = "lg"
            )
            gr.Markdown("""
**How it works:**
1. 📁 Upload any CSV
2. 🔍 Click Analyze
3. 📊 Get AI insights + charts
4. 📄 Download PDF report

**Works with any domain:**
- 🏥 Healthcare data
- 🏦 Banking & Finance
- 🛒 E-Commerce
- 📱 SaaS & Tech
- 🎓 Education
- 🏭 Manufacturing
            """)

    with gr.Tabs():
        with gr.TabItem("📊 Insights & Profile"):
            insights_out = gr.Textbox(
                label    = "AI Analysis Results",
                lines    = 35,
                max_lines= 60
            )

        with gr.TabItem("📈 Visualizations"):
            gallery_out = gr.Gallery(
                label   = "Auto-Generated Charts",
                columns = 2,
                height  = 700
            )

        with gr.TabItem("📄 Download PDF Report"):
            pdf_out = gr.File(
                label = "⬇️ Download Full EDA Report"
            )
            gr.Markdown("""
### Your PDF Report Includes:
- ✅ Complete Dataset Profile & Statistics
- ✅ AI-Generated EDA Insights (5 sections)
- ✅ All Auto-Generated Visualizations
- ✅ 5 Testable Business Hypotheses
- ✅ Professional formatting with page numbers
            """)

    analyze_btn.click(
        fn      = run_eda_agent,
        inputs  = [csv_input],
        outputs = [insights_out, gallery_out, pdf_out]
    )

    gr.Markdown("""
---
### 💡 Try these datasets:
| Dataset | Link |
|---------|------|
| Titanic | kaggle.com/c/titanic |
| House Prices | kaggle.com/c/house-prices-advanced-regression-techniques |
| Netflix Shows | kaggle.com/datasets/shivamb/netflix-shows |
    """)

if __name__ == "__main__":
    demo.launch()
