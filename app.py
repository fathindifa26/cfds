import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx

st.title("Simulasi Centrality pada Jaringan Sosial (SNA)")

st.markdown("""
Simulasi ini menghitung **indegree, outdegree, betweenness, closeness, eigenvector centrality, dan PageRank** pada jaringan sosial berbasis adjacency matrix.  
- **Baris** = dari siapa (source), **kolom** = ke siapa (target).
- Isikan 1 jika ada hubungan (directed edge), 0 jika tidak.
- Edit angka atau tambah baris/kolom untuk simulasi jumlah node/aktor berbeda.
""")

# Input nama node
node_names = st.text_input("Nama node (pisahkan koma, contoh: A,B,C,D):", value="A,B,C,D")
node_names = [n.strip() for n in node_names.split(",") if n.strip()]
N = len(node_names) if len(node_names) > 1 else 4

# Default matrix
default_matrix = np.array([
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])
if N != 4:
    default_matrix = np.zeros((N,N), dtype=int)

adj_df = pd.DataFrame(
    default_matrix,
    index=[f"Dari: {n}" for n in node_names],
    columns=[f"Ke: {n}" for n in node_names]
)

# Editable matrix
st.markdown("### Edit Adjacency Matrix:")
df_edited = st.data_editor(
    adj_df,
    num_rows="dynamic",
    use_container_width=True,
    key="adj_editor"
)

actual_names = [n.replace("Dari: ","") for n in df_edited.index]
try:
    matrix = df_edited.to_numpy(dtype=int)
    
    # Buat directed graph dari adjacency matrix
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    mapping = {i: n for i, n in enumerate(actual_names)}
    G = nx.relabel_nodes(G, mapping)
    
    # INDEGREE & OUTDEGREE
    indegree = dict(G.in_degree())
    outdegree = dict(G.out_degree())

    # BETWEENNESS
    betweenness = nx.betweenness_centrality(G, normalized=True)
    # CLOSENESS
    closeness = nx.closeness_centrality(G)
    
    # EIGENVECTOR CENTRALITY (by matrix transpose)
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    idx = np.argmax(np.real(eigvals))
    v = np.abs(np.real(eigvecs[:, idx]))
    v_norm = v / v.max() if v.max() != 0 else v

    # PAGE RANK
    pagerank = nx.pagerank(G, alpha=0.85)
    # Normalisasi agar PageRank max = 1 (opsional)
    pr_values = np.array([pagerank[n] for n in actual_names])
    pr_norm = pr_values / pr_values.max() if pr_values.max() != 0 else pr_values

    st.markdown("### Centrality Metrics per Node")
    st.markdown("""
    - **Indegree**: jumlah edge masuk ke node (seberapa sering dihubungi)
    - **Outdegree**: jumlah edge keluar dari node (seberapa aktif menghubungi)
    - **Betweenness**: seberapa sering node menjadi "jembatan" pada jalur terpendek antar node lain
    - **Closeness**: seberapa dekat node ke semua node lain dalam jaringan (aksesibilitas)
    - **Eigenvector (IN-centrality)**: pengaruh global, tinggi jika dihubungi oleh node penting
    - **PageRank**: kemungkinan node dikunjungi "random surfer" dalam jaringan (skala max=1)
    """)

    df_result = pd.DataFrame({
        'Node': actual_names,
        'Indegree': [indegree[n] for n in actual_names],
        'Outdegree': [outdegree[n] for n in actual_names],
        'Betweenness': [np.round(betweenness[n], 4) for n in actual_names],
        'Closeness': [np.round(closeness[n], 4) for n in actual_names],
        'Eigenvector (IN)': np.round(v_norm, 4),
        'PageRank': np.round(pr_norm, 4)
    })
    st.dataframe(df_result, use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")
