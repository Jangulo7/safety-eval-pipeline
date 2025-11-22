import streamlit as st
import pandas as pd
import altair as alt
import boto3
import json
import os
from botocore.exceptions import ClientError

st.set_page_config(layout="wide", page_title="AI Eval Queue")

@st.cache_data(ttl=3600)
def load_data():
    s3 = boto3.client('s3')
    bucket = os.getenv("S3_MODEL_BUCKET")
    prefix = "benchmarks/artificial_analysis/"
    
    data = []
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page: continue
            
            for obj in page['Contents']:
                if obj['Key'].endswith('.json'):
                    try:
                        obj_body = s3.get_object(Bucket=bucket, Key=obj['Key'])
                        content = json.loads(obj_body['Body'].read())
                        
                        # Extract basic info
                        res = {
                            "model": content.get("config", {}).get("model_name", "Unknown"),
                            "timestamp": obj['LastModified']
                        }
                        
                        if "results" in content and isinstance(content["results"], dict):
                            for k, v in content["results"].items():
                                # Clean metric name: "lighteval|mmlu|5|0" -> "MMLU"
                                # Fix Error #5: Get VALUE not KEY
                                # Fix Error #2: Handle empty dicts
                                short_name = k.split('|')[1].upper() if '|' in k else k
                                
                                if isinstance(v, dict) and v:
                                    # Safely get the first value
                                    metric_val = next(iter(v.values()), 0)
                                    res[short_name] = metric_val * 100
                                else:
                                    res[short_name] = 0.0
                                
                        data.append(res)
                    except Exception as e:
                        print(f"Error parsing {obj['Key']}: {e}")
                        continue
    except ClientError as e:
        st.error(f"S3 Connection Error: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(data)

st.title("🤖 Overnight Evaluation Report")
df = load_data()

if not df.empty:
    st.dataframe(df, use_container_width=True)
    
    # Visualization logic (simplified)
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("X Axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y Axis", numeric_cols, index=1)
        
        st.subheader(f"{y_axis} vs {x_axis}")
        chart = alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X(x_axis),
            y=alt.Y(y_axis),
            tooltip=['model', x_axis, y_axis]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No results found in S3 yet.")