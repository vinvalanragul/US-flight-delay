"""
Page 8: Hadoop Architecture, Ubuntu/WSL2 Setup & Viva Preparation
"""
import streamlit as st
import os

st.title("🐘 Hadoop Architecture & Viva Preparation")
st.markdown("Distributed architecture documentation, Ubuntu/WSL2 setup steps, HDFS commands, and Viva Voce question-and-answer repository.")

tab_arch, tab_setup, tab_viva = st.tabs(["1. Hadoop Architecture", "2. Ubuntu / WSL2 Setup Guide", "3. Viva Q&A Repository"])

with tab_arch:
    st.subheader("HDFS & MapReduce Architectural Workflow")
    st.markdown("""
    In a distributed Hadoop cluster:
    1. **Client Submission:** The job runner script stages the dataset in HDFS (`/user/hadoop/flight_delay/input/flights.csv`).
    2. **Input Splits & YARN:** YARN ResourceManager coordinates NodeManagers to allocate ApplicationMaster and container workers.
    3. **Map Phase:** Worker nodes execute `mapper.py` in parallel over HDFS block splits (128MB each).
    4. **Shuffle & Sort:** Map outputs are partitioned by hash key and sorted lexicographically across the cluster network.
    5. **Reduce Phase:** Reducer containers execute `reducer.py` to aggregate counts, delays, and percentages into `/user/hadoop/flight_delay/output/`.
    """)
    st.code("""
# Unix Pipeline Verification Command
cat data/flights_sample.csv | head -n 500 | python3 hadoop_streaming/mapper.py | sort | python3 hadoop_streaming/reducer.py
    """, language="bash")

with tab_setup:
    st.subheader("Ubuntu / Ubuntu on WSL2 Configuration Guide")
    st.markdown("""
    Execute the following commands in your Ubuntu terminal:
    
    ```bash
    # 1. Install Java Development Kit (OpenJDK 11)
    sudo apt update
    sudo apt install -y openjdk-11-jdk
    java -version
    
    # 2. Download and Extract Apache Hadoop
    wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
    tar -xzf hadoop-3.3.6.tar.gz
    sudo mv hadoop-3.3.6 /usr/local/hadoop
    
    # 3. Configure Environment Variables in ~/.bashrc
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    export HADOOP_HOME=/usr/local/hadoop
    export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
    source ~/.bashrc
    
    # 4. Format NameNode (First time ONLY - NEVER format a populated cluster!)
    hdfs namenode -format
    
    # 5. Start Hadoop Daemons
    start-dfs.sh
    start-yarn.sh
    jps
    
    # 6. Execute Genuine Hadoop Streaming Job
    chmod +x hadoop_streaming/run_hadoop_job.sh
    ./hadoop_streaming/run_hadoop_job.sh AIRLINE data/flights_sample.csv
    ```
    """)

with tab_viva:
    st.subheader("Comprehensive Viva Voce Questions & Answers")
    viva_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "viva_questions.md")
    if os.path.exists(viva_file):
        with open(viva_file, "r", encoding="utf-8") as f:
            viva_content = f.read()
        st.markdown(viva_content)
    else:
        st.info("Viva document available in reports/viva_questions.md")
