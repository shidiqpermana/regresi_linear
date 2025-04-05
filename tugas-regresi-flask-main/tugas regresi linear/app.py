import os
import numpy as np
import joblib
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan encoder lokasi
if os.path.exists("model.pkl"):
    model, le = joblib.load("model.pkl")
else:
    raise FileNotFoundError("File model.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")

# Fungsi untuk membuat grafik prediksi
def generate_graph(luas_tanah, harga_prediksi):
    luas_sample = np.linspace(50, 500, 100)  # Contoh luas tanah
    harga_sample = model.predict(np.column_stack([luas_sample, np.zeros_like(luas_sample)]))  # Asumsi lokasi = 0

    plt.figure(figsize=(6, 4))
    plt.plot(luas_sample, harga_sample, label="Regresi Linear", color="blue")
    plt.scatter(luas_tanah, harga_prediksi, color="red", marker="o", label="Prediksi Anda")
    plt.xlabel("Luas Tanah (m²)")
    plt.ylabel("Harga Rumah")
    plt.legend()
    plt.grid()

    # Simpan ke buffer
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{graph_url}"

# Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None
    prediction_text = None

    if request.method == "POST":
        try:
            luas_tanah = float(request.form["luas_tanah"])
            lokasi = request.form["lokasi"]

            # Ubah lokasi ke bentuk numerik
            lokasi_encoded = le.transform([lokasi])[0]

            # Prediksi harga
            input_data = np.array([[luas_tanah, lokasi_encoded]])
            harga_prediksi = model.predict(input_data)[0]

            # Buat grafik
            graph_url = generate_graph(luas_tanah, harga_prediksi)
            prediction_text = f"Prediksi harga rumah: Rp {harga_prediksi:,.2f}"

        except Exception as e:
            prediction_text = f"Terjadi kesalahan: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text, graph_url=graph_url)

# Jalankan Aplikasi
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
