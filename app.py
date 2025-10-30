
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
from scipy.sparse import hstack
import base64
import io
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(layout="wide", page_title="Task Management")


@st.cache_resource
def load_models():
    # update paths if needed
    vec = joblib.load("tfidf_vectorizer.pkl")           
    clf = joblib.load("task_classifier_model.pkl")     
    prio_model = joblib.load("priority_prediction_model.pkl")  
   
    try:
        cat_encoder = joblib.load("category_encoder.pkl")
    except:
        cat_encoder = None
    try:
        prio_encoder = joblib.load("priority_label_encoder.pkl")
    except:
        prio_encoder = None
    return vec, clf, prio_model, cat_encoder, prio_encoder

# If you don't have models yet, comment out the model loading and use mock predictions
with st.spinner("Loading models..."):
    try:
        vectorizer, classifier, priority_model, cat_enc, prio_enc = load_models()
        models_loaded = True
    except Exception as e:
        st.warning("Model files not found or failed to load. Running in demo/mock mode.")
        vectorizer = classifier = priority_model = cat_enc = prio_enc = None
        models_loaded = False

# -------------------------
# Data loading (DB or CSV)
# -------------------------
@st.cache_resource
def get_db_connection():
    username = "root"             
    password = "HS@2223"     
    host = "localhost"            
    port = 3306                
    database = "task_management" 

    
    engine = create_engine("mysql+pymysql://root:HS%402223@localhost:3306/task_management")

    return engine


@st.cache_data
def load_data():
    engine = get_db_connection()
    query = """
        SELECT task_id, task_title, task_description, category, priority,
               estimated_hours, assigned_to, status, created_at, deadline
        FROM tasks;
    """
    df = pd.read_sql(query, engine, parse_dates=["created_at", "deadline"])
    return df

df = load_data()



def get_all_tasks():
    engine = get_db_connection()
    query = "SELECT * FROM tasks ORDER BY created_at DESC"
    df = pd.read_sql(query, engine)
    return df

def get_task_by_id(task_id):
    engine = get_db_connection()
    query = f"SELECT * FROM tasks WHERE task_id = {task_id}"
    df = pd.read_sql(query, engine)
    return df

def insert_task(task_title, description, category, priority, estimated_hours, assigned_to, status, deadline):
    engine = get_db_connection()
    query = """
        INSERT INTO tasks (task_title, task_description, category, priority, estimated_hours, assigned_to, status, deadline, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """
    with engine.connect() as conn:
        conn.execute(query, (task_title, description, category, priority, estimated_hours, assigned_to, status, deadline))
        conn.commit()

def update_task_status(task_id, new_status):
    engine = get_db_connection()  # <-- add this line
    query = "UPDATE tasks SET status = :status WHERE task_id = :task_id"
    with engine.begin() as conn:
        conn.execute(
            text(query),
            {"status": new_status, "task_id": task_id}
        )



def reassign_task(task_id, new_user):
    engine = get_db_connection()  
    query = "UPDATE tasks SET assigned_to = :new_user WHERE task_id = :task_id"
    with engine.begin() as conn:
        conn.execute(
            text(query),
            {"new_user": new_user, "task_id": task_id}
        )




df['days_left'] = (pd.to_datetime(df['deadline']).dt.date - pd.Timestamp.now().date()).apply(lambda x: x.days)
df['is_open'] = df['status'] != 'Closed'
df['text'] = df['task_title'].fillna('') + ' ' + df['task_description'].fillna('')


st.sidebar.header("Filters & Controls")
date_max = df['created_at'].max()
date_min = df['created_at'].min()

date_range = st.sidebar.date_input("Date range", [date_min.date(), date_max.date()])
selected_categories = st.sidebar.multiselect("Categories", options=df['category'].unique().tolist(), default=df['category'].unique().tolist())
selected_priority = st.sidebar.multiselect("Priority", options=df['priority'].unique().tolist(), default=df['priority'].unique().tolist())
selected_users = st.sidebar.multiselect("Assigned to", options=df['assigned_to'].unique().tolist(), default=None)
prio_prob_thresh = st.sidebar.slider("Priority (High) probability threshold", 0.0, 1.0, 0.5, 0.01)


start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
mask = (df['created_at'] >= start_date) & (df['created_at'] < end_date) & (df['category'].isin(selected_categories)) & (df['priority'].isin(selected_priority))
if selected_users:
    mask &= df['assigned_to'].isin(selected_users)
df_filtered = df[mask].copy()


st.header("AI-Powered Task Management")
k1, k2, k3, k4, k5, k6 = st.columns(6)

total_tasks = len(df_filtered)
open_tasks = df_filtered['is_open'].sum()
high_tasks = (df_filtered['priority'] == 'High').sum()
avg_ttc = df_filtered[df_filtered['status']=='Closed'].apply(lambda r: np.nan).shape[0]  # placeholder
avg_load = df_filtered.groupby('assigned_to').size().mean()


model_acc = 0.82 if models_loaded else None
model_pr_recall_high = 0.78 if models_loaded else None

k1.metric("Total tasks", total_tasks)
k2.metric("Open tasks", open_tasks)
k3.metric("High priority", high_tasks)
k4.metric("Avg load (tasks/user)", round(avg_load,2))
k5.metric("Model acc", f"{model_acc*100:.1f}%" if model_acc else "N/A")
k6.metric("Priority recall (High)", f"{model_pr_recall_high*100:.1f}%" if model_pr_recall_high else "N/A")


def predict_batch(df_in):
    df_local = df_in.copy()
    if models_loaded and vectorizer is not None:
        X_text = vectorizer.transform(df_local['text'].tolist())
        # predict category if needed
        try:
            df_local['pred_category'] = classifier.predict(X_text)
        except Exception:
            df_local['pred_category'] = df_local['category']  # fallback

        try:
            if cat_enc is not None:
                cats_enc = cat_enc.transform(df_local['pred_category'])
            else:
                # naive labelmap
                cats_enc = pd.factorize(df_local['pred_category'])[0]
            num_feats = df_local[['estimated_hours']].fillna(0).values

            X_full = hstack([X_text, num_feats])
            pr_probs = priority_model.predict_proba(X_full) if hasattr(priority_model, "predict_proba") else None
            if pr_probs is not None:
                # guess index for 'High' if prio encoder available
                if prio_enc is not None:
                    high_idx = list(prio_enc.classes_).index('High') if 'High' in prio_enc.classes_ else -1
                else:
                    # fallback: assume classes sorted as Low/Medium/High -> index -1
                    high_idx = -1
                df_local['prio_high_prob'] = pr_probs[:, high_idx] if high_idx >= 0 else pr_probs.max(axis=1)
                df_local['pred_priority'] = priority_model.predict(X_full)
                # decode if encoder exists
                if prio_enc is not None:
                    df_local['pred_priority'] = prio_enc.inverse_transform(df_local['pred_priority'])
            else:
                df_local['prio_high_prob'] = np.nan
                df_local['pred_priority'] = np.nan
        except Exception as e:
            df_local['pred_priority'] = df_local['priority']
            df_local['prio_high_prob'] = np.nan
    else:
        # demo mode: set predicted priority = actual priority
        df_local['pred_category'] = df_local['category']
        df_local['pred_priority'] = df_local['priority']
        df_local['prio_high_prob'] = df_local['priority'].apply(lambda p: 0.9 if p=='High' else (0.5 if p=='Medium' else 0.1))
    return df_local

df_pred = predict_batch(df_filtered)




st.subheader("Assign & Update")

col1, col2 = st.columns(2)

with col1:
    st.write(" **Reassign Task**")
    with st.form("reassign_form"):
        task_id = st.text_input("Task ID")
        new_user = st.text_input("New Assigned User")
        submitted = st.form_submit_button("Reassign")
        if submitted:
            reassign_task(task_id, new_user)
            st.success(f"Task {task_id} reassigned to {new_user}")
            st.rerun()


with col2:
    st.write(" **Update Status**")
    with st.form("status_form"):
        task_id = st.text_input("Task ID (for status update)")
        new_status = st.selectbox("New Status", ["Open", "In Progress", "Closed"])
        submitted = st.form_submit_button("Update Status")
        if submitted:
            update_task_status(task_id, new_status)
            st.success(f"Task {task_id} marked as {new_status}")
            st.rerun()


st.subheader("Urgent Tasks (next 3 days or High priority)")
urgent_mask = (df_pred['days_left'] <= 3) | (df_pred['pred_priority']=='High')
urgent = df_pred[urgent_mask].sort_values(by=['pred_priority','days_left'])
if urgent.empty:
    st.info("No urgent tasks under current filters.")
else:
   
    cols = ["task_id","task_title","pred_priority","prio_high_prob","assigned_to","days_left","estimated_hours"]
    st.dataframe(urgent[cols].assign(days_left=urgent['days_left'].astype(int)).head(100))

   
    def get_table_download_link(df_to_download):
        csv = df_to_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="urgent_tasks.csv">Download CSV</a>'
    st.markdown(get_table_download_link(urgent[cols]), unsafe_allow_html=True)



st.subheader("Model Performance")
# If you store true/predicted values in DB, load and compute metrics; here show mock
if models_loaded:
    st.write("Model accuracy (task classifier):", f"{model_acc*100:.1f}%")
    st.write("Priority recall (High):", f"{model_pr_recall_high*100:.1f}%")
else:
    st.info("No saved evaluation metrics found. Train models and save accuracies to show them here.")




st.subheader("Priority Distribution")
prio_counts = df_pred['pred_priority'].value_counts().reset_index()
prio_counts.columns = ['priority', 'count']
fig_prio = px.pie(prio_counts, values='count', names='priority', hole=0.4, title="Predicted Priority")
st.plotly_chart(fig_prio, use_container_width=True)


st.subheader("Tasks Over Time")
time_df = df_pred.groupby(pd.Grouper(key='created_at', freq='7D')).size().reset_index(name='count')
fig_time = px.line(time_df, x='created_at', y='count', markers=True, title="Tasks created per week")
st.plotly_chart(fig_time, use_container_width=True)


st.subheader("Workload per User")
wl = df_pred[df_pred['is_open']].groupby('assigned_to').size().reset_index(name='open_tasks').sort_values('open_tasks', ascending=False)
fig_wl = px.bar(wl, x='assigned_to', y='open_tasks', title="Open tasks per user")
st.plotly_chart(fig_wl, use_container_width=True)





st.markdown("---")
st.caption(f"Data range: {start_date.date()} — {end_date.date()} | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

