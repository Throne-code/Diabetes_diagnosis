import streamlit as st
import pandas as pd

# ==========================================
# SISTEM PAKAR DIAGNOSIS FLU DAN DEMAM
# METODE CASE-BASED REASONING (CBR) - STREAMLIT
# Jalankan: streamlit run app_cbr.py
# ==========================================

st.set_page_config(page_title="CBR Flu & Demam", page_icon="🩺", layout="wide")

DAFTAR_GEJALA = ["Demam", "Batuk", "Pilek", "Sakit kepala", "Nyeri otot"]

BASIS_AWAL = [
    {"kode": "K001", "gejala": [1, 1, 1, 0, 1], "diagnosis": "Flu"},
    {"kode": "K002", "gejala": [1, 0, 0, 1, 0], "diagnosis": "Demam"},
    {"kode": "K003", "gejala": [1, 1, 0, 1, 1], "diagnosis": "Flu"},
    {"kode": "K004", "gejala": [0, 1, 1, 0, 0], "diagnosis": "Flu"},
    {"kode": "K005", "gejala": [1, 0, 0, 0, 0], "diagnosis": "Demam"},
]

# ---------- State ----------
if "basis_kasus" not in st.session_state:
    st.session_state.basis_kasus = [dict(k) for k in BASIS_AWAL]
if "kasus_baru" not in st.session_state:
    st.session_state.kasus_baru = None
if "hasil_retrieve" not in st.session_state:
    st.session_state.hasil_retrieve = None
if "pesan" not in st.session_state:
    st.session_state.pesan = None


# ---------- Fungsi CBR ----------
def hitung_similarity(kasus_baru, kasus_lama):
    sama = sum(1 for a, b in zip(kasus_baru, kasus_lama) if a == b)
    return sama / len(kasus_baru)


def retrieve(kasus_baru):
    hasil = [
        {
            "kode": k["kode"],
            "diagnosis": k["diagnosis"],
            "similarity": hitung_similarity(kasus_baru, k["gejala"]),
            "gejala": k["gejala"],
        }
        for k in st.session_state.basis_kasus
    ]
    hasil.sort(key=lambda x: x["similarity"], reverse=True)
    return hasil


def reuse(hasil):
    return hasil[0] if hasil else None


def retain(kasus_baru, diagnosis):
    kode = "K" + str(len(st.session_state.basis_kasus) + 1).zfill(3)
    st.session_state.basis_kasus.append(
        {"kode": kode, "gejala": kasus_baru, "diagnosis": diagnosis}
    )
    return kode


def tabel_gejala(daftar):
    rows = []
    for k in daftar:
        row = {"Kode": k["kode"]}
        for nama, v in zip(DAFTAR_GEJALA, k["gejala"]):
            row[nama] = "✔" if v else "–"
        row["Diagnosis"] = k["diagnosis"]
        if "similarity" in k:
            row["Similarity (%)"] = round(k["similarity"] * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Callback ----------
def terima():
    hasil = reuse(st.session_state.hasil_retrieve)
    kode = retain(st.session_state.kasus_baru, hasil["diagnosis"])
    st.session_state.pesan = ("success", f"Kasus baru disimpan sebagai **{kode}**.")
    st.session_state.kasus_baru = None
    st.session_state.hasil_retrieve = None


def tolak():
    st.session_state.pesan = (
        "warning",
        "Kasus tidak disimpan. Hasil perlu ditinjau kembali.",
    )
    st.session_state.kasus_baru = None
    st.session_state.hasil_retrieve = None


def reset():
    st.session_state.basis_kasus = [dict(k) for k in BASIS_AWAL]
    st.session_state.kasus_baru = None
    st.session_state.hasil_retrieve = None
    st.session_state.pesan = ("info", "Basis kasus dikembalikan ke kondisi awal.")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📚 Basis Kasus")
    st.caption(f"Total kasus: {len(st.session_state.basis_kasus)}")
    st.dataframe(tabel_gejala(st.session_state.basis_kasus), hide_index=True)
    st.button("Reset basis kasus", on_click=reset, use_container_width=True)

# ---------- Header ----------
st.title("🩺 Sistem Pakar Diagnosis Flu dan Demam")
st.caption("Metode Case-Based Reasoning (Retrieve → Reuse → Revise → Retain)")

if st.session_state.pesan:
    tipe, teks = st.session_state.pesan
    getattr(st, tipe)(teks)
    st.session_state.pesan = None

# ---------- Input gejala ----------
st.subheader("1. Input Gejala")
with st.form("form_gejala"):
    cols = st.columns(len(DAFTAR_GEJALA))
    pilihan = [
        int(col.checkbox(g, key=f"gejala_{i}"))
        for i, (col, g) in enumerate(zip(cols, DAFTAR_GEJALA))
    ]
    kirim = st.form_submit_button("🔍 Diagnosis", type="primary")

if kirim:
    st.session_state.kasus_baru = pilihan
    st.session_state.hasil_retrieve = retrieve(pilihan)

# ---------- Hasil ----------
if st.session_state.kasus_baru is not None:
    hasil = st.session_state.hasil_retrieve

    st.subheader("2. Retrieve — Kasus Paling Mirip")
    st.dataframe(
        tabel_gejala(hasil),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Similarity (%)": st.column_config.ProgressColumn(
                "Similarity (%)", min_value=0, max_value=100, format="%.2f"
            )
        },
    )

    st.subheader("3. Reuse — Rekomendasi")
    terpilih = reuse(hasil)
    c1, c2, c3 = st.columns(3)
    c1.metric("Kasus terpilih", terpilih["kode"])
    c2.metric("Rekomendasi diagnosis", terpilih["diagnosis"])
    c3.metric("Similarity", f"{terpilih['similarity'] * 100:.2f}%")
    st.progress(terpilih["similarity"])

    st.subheader("4. Revise — Konfirmasi Pakar")
    st.info(
        "Hasil ini perlu ditinjau oleh pakar. Terima hasil sebagai solusi "
        "untuk simulasi pembelajaran?"
    )
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("✅ Terima", on_click=terima, type="primary", use_container_width=True)
    b2.button("❌ Tolak", on_click=tolak, use_container_width=True)

st.divider()
st.caption("⚠️ Seluruh data adalah simulasi. Hasil hanya untuk pembelajaran, bukan diagnosis medis.")