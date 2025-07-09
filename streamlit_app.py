import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import os
import tempfile

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

# File path for CSV
data_path = "data/hr_dataset_switzerland.csv"

# Load or create dataset
def load_data():
    if os.path.exists(data_path):
        return pd.read_csv(data_path, parse_dates=["Hire Date"])
    else:
        return pd.DataFrame(columns=[
            "First Name", "Last Name", "Residence", "Age", "Department",
            "Seniority Level", "Workload", "Vacation Days Total",
            "Vacation Days Taken", "Hire Date"
        ])

def save_data(df):
    df.to_csv(data_path, index=False)

# UI setup
st.set_page_config(page_title="HR Dashboard", layout="wide")
st.title("HR Dashboard \U0001F4BC")

df = load_data()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Summary", "Add New Hire", "Chat with PDF"])

# --- TAB 1: Summary & Visualizations ---
with tab1:
    st.header("Overview Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        st.metric("Average Age", f"{df['Age'].mean():.1f}" if not df.empty else "-")
    with col3:
        workloads = df['Workload'].str.rstrip('%').astype(float) if not df.empty else []
        st.metric("Avg Workload", f"{workloads.mean():.1f}%" if len(workloads) else "-")
    with col4:
        if not df.empty:
            pct_used = (df['Vacation Days Taken'] / df['Vacation Days Total']).mean() * 100
            st.metric("Avg Vacation Used", f"{pct_used:.0f}%")
        else:
            st.metric("Avg Vacation Used", "-")

    st.divider()
    st.subheader("Employees per Department")
    if not df.empty:
        dept_counts = df['Department'].value_counts().reset_index()
        dept_counts.columns = ['Department', 'Count']
        fig1 = px.bar(dept_counts, x='Department', y='Count', color='Department',
                     title="Employees per Department", height=300)
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Age Distribution")
    if not df.empty:
        fig2 = px.histogram(df, x='Age', nbins=10, title="Age Distribution",
                            color_discrete_sequence=['indianred'], height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Hiring Trend Over Time")
    if not df.empty:
        df['Hire Date'] = pd.to_datetime(df['Hire Date'], errors='coerce')
        hires_over_time = df['Hire Date'].dt.to_period("M").value_counts().sort_index()
        hires_df = hires_over_time.reset_index()
        hires_df.columns = ["Month", "Hires"]
        hires_df["Month"] = hires_df["Month"].astype(str)
        fig3 = px.line(hires_df, x="Month", y="Hires", title="Hiring Trend", markers=True, height=300)
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 2: Form ---
with tab2:
    st.header("Add New Hire")

    with st.form("new_hire_form"):
        first = st.text_input("First Name")
        last = st.text_input("Last Name")
        residence = st.selectbox("Residence", [
            'Zürich', 'Bern', 'Luzern', 'Uri', 'Schwyz', 'Obwalden', 'Nidwalden', 'Glarus',
            'Zug', 'Fribourg', 'Solothurn', 'Basel-Stadt', 'Basel-Landschaft', 'Schaffhausen',
            'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'St. Gallen', 'Graubünden',
            'Aargau', 'Thurgau', 'Ticino', 'Vaud', 'Valais', 'Neuchâtel', 'Geneva', 'Jura'])
        age = st.number_input("Age", min_value=18, max_value=70, step=1)
        dept = st.selectbox("Department", ['HR', 'Production', 'IT', 'Finance', 'Sales'])
        level = st.selectbox("Seniority Level", ['Intern', 'Junior', 'Mid', 'Senior', 'Lead'])
        workload = st.selectbox("Workload", ["60%", "70%", "80%", "90%", "100%"])
        hire_date = st.date_input("Hire Date", value=date.today())

        submitted = st.form_submit_button("Add New Hire")

        if submitted:
            workload_value = int(workload.rstrip('%')) / 100
            total_vac = int(25 * workload_value)
            taken_vac = 0

            new_row = pd.DataFrame([{
                "First Name": first,
                "Last Name": last,
                "Residence": residence,
                "Age": age,
                "Department": dept,
                "Seniority Level": level,
                "Workload": workload,
                "Vacation Days Total": total_vac,
                "Vacation Days Taken": taken_vac,
                "Hire Date": hire_date
            }])

            df = pd.concat([df, new_row], ignore_index=True)
            save_data(df)
            st.success(f"Added {first} {last} to the dataset!")

# --- TAB 3: PDF Chat ---
with tab3:
    st.header("Ask Questions About a PDF")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(pages, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            retriever=vectorstore.as_retriever()
        )

        question = st.text_input("Ask a question about the PDF")
        if question:
            response = qa_chain.run(question)
            st.write("**Answer:**", response)