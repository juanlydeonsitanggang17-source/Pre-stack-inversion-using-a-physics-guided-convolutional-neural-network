# Pre-stack-inversion-using-a-physics-guided-convolutional-neural-network
Implementasi Unsupervised Physics-Guided 1D CNN menggunakan TensorFlow untuk Pre-Stack Seismic Inversion. Model ini tidak memerlukan data berlabel, melainkan belajar secara mandiri melalui integrasi persamaan Aki-Richards dan operasi konvolusi di dalam custom training loop.
# Physics-Guided CNN for Pre-Stack Seismic Inversion 🌍💻

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python)
![Geophysics](https://img.shields.io/badge/Domain-Geophysics-4CAF50)
![Machine Learning](https://img.shields.io/badge/AI-Physics_Informed_Neural_Networks-blue)

## 📌 Deskripsi Proyek
Repositori ini berisi implementasi dari **Physics-Informed Neural Network (PINN)** untuk melakukan inversi seismik *pre-stack* (estimasi Vp, Vs, dan Densitas). Berbeda dengan arsitektur *Deep Learning* konvensional yang bergantung pada *supervised learning* (data berpasangan dengan label), proyek ini menggunakan pendekatan **Unsupervised Learning**. 

Jaringan saraf konvolusi (1D CNN) ini dilatih menggunakan hukum fisika perambatan gelombang seismik. Persamaan **Aki-Richards** dan operasi **konvolusi *angle-dependent wavelet*** diintegrasikan langsung ke dalam *computational graph* TensorFlow sebagai *custom loss function*. Model ini belajar secara mandiri untuk meminimalkan *misfit* (Mean Squared Error) antara data seismik observasi dengan data sintetik yang dikalkulasi secara *real-time*.

Proyek ini terinspirasi oleh pendekatan metodologi dalam jurnal riset SEG (Biswas et al., 2019).

## 🚀 Fitur Utama
1. **Unsupervised Physics-Guided Training:** Tidak menggunakan label *ground truth* saat pelatihan. Model belajar dari hukum fisika dan *differentiable forward modeling* di dalam `tf.GradientTape`.
2. **Advanced Synthetic Geology Generator:** Skrip Python bawaan untuk membangun ekosistem bumi 2D yang sangat kompleks, mencakup:
   - *Dipping Graben Faults* (Sesar normal miring) dengan efek *plastic fault drag*.
   - *Angular Unconformities* dan *Stratigraphic Pinch-outs*.
   - Tekstur *Seismic Facies* spesifik (Laminated, Channel, Chaotic) berbasis *1/f Fractal Noise*.
3. **Rock Physics Integration:** Hubungan empiris (seperti persamaan Castagna modifikasi dan Gardner) untuk menjaga validitas geologis.
4. **Custom Training Loop di TensorFlow:** Optimasi komputasi tensor presisi tinggi untuk inversi properti ruang (*spatial properties*).

## 📂 Struktur Repositori
- `data_generator.py` : Skrip untuk menghasilkan model geologi bumi 2D kompleks dan data *angle gathers* seismik (Forward Modeling).
- `pinn_inversion.py` : Skrip utama berisi arsitektur 1D CNN, *differentiable physics loss function*, dan *custom training loop* menggunakan TensorFlow.
- `advanced_geology_dataset.npz` : (Contoh Output) Dataset matriks properti elastis dan tensor seismik hasil *generate*.

## 🛠️ Persyaratan Pustaka (Dependencies)
Pastikan Anda memiliki *environment* Python dengan pustaka berikut:
```bash
pip install numpy scipy matplotlib tensorflow
