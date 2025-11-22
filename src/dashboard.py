import streamlit as st
import pandas as pd
import altair as alt
import boto3
import json
import os

st.set_page_config(layout="wide", page_title="AI Eval Queue")

@st.cache_data(ttl=3600)
def load_data():
    s3 = boto3.client('s3')
    bucket = os.getenv("S3_MODEL_BUCKET")
    prefix = "benchmarks/artificial_analysis/"
    
    data = []
    # (Simplified paginator logic)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if obj['Key'].endswith('.json'):
                try:
                    obj_body = s3.get_object(Bucket=bucket, Key=obj['Key'])
                    content = json.loads(obj_body['Body'].read())
                    
                    # LightEval JSON parsing logic
                    res = {"model": content.get("config", {}).get("model_name", "Unknown")}
                    
                    if "results" in content:
                        for k, v in content["results"].items():
                            # Clean metric name (e.g. "lighteval|mmlu|5|0" -> "MMLU")
                            short_name = k.split('|')[1].upper()
                            metric_val = next(iter(v)) # Get first metric (acc)
                            res[short_name] = metric_val * 100
                            
                    data.append(res)
                except: continue
    return pd.DataFrame(data)

st.title("🤖 Overnight Evaluation Report")
df = load_data()

if not df.empty:
    st.dataframe(df, use_container_width=True)
    
    # Scatter Plot: Artificial Analysis Style
    # Assuming we have MMLU (Quality) vs Reasoning (GSM8K)
    if 'MMLU:MAJOR' in df.columns and 'GSM8K' in df.columns:
        st.subheader("Quality vs Reasoning Trade-off")
        chart = alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X('MMLU:MAJOR', title='MMLU (Knowledge)'),
            y=alt.Y('GSM8K', title='GSM8K (Math)'),
            tooltip=['model']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No results found in S3 yet.")