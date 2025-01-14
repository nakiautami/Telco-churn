# Telco-chur
# **Business Problem Understanding**
**Context**

Perusahaan telekomunikasi yang menyediakan layanan internent berkecepatan tinggi menghadapi tantangan untuk mempertahankan pelanggan di tengah persaingan yang ketat di industri ini. Kehilangan pelanggan (*customer churn*) dapat berdampak signifikan pada pendapatan, mengingat  biaya akusisi pelanggan baru jauh lebih tinggi dibandingkan mempertahankan pelanggan yang ada. Dengan memanfaatkan data pelanggan dan teknik *machine learning*, perusahaan dapat mengidentifikasi pelanggan yang berpotensi churn dan mengambil langkah proaktif untuk meningkatkan pengalaman pelanggan serta menjaga loyalitas mereka.

**Churn**:

**Definisi Churn** merujuk pada situasi dimana pelanggan memutuskan untuk berhenti menggunakan layanan suatu perusahaan. Hal ini dapat terjadi karena berbagai alasan, seperti:
- Membatalkan langganan secara langsung.
- Beralih ke penyedia layanan lain.

**Klasifikasi Churn**:

No(0) : Pelanggan tetap melanjutkan layanan dengan perusahaan internet saat ini.

**Problem Statement:**
Industri telekomunikasi menghadapi persaingan ketat di pasar yang hampir jenuh. Pelanggan sering beralih ke penyedia layanan lain yang menawarkan harga lebih kompetitif, fitur menarik, atau kualitas layanan yang lebih baik. Sebagian besar layanan telekomunikasi bergantung pada kontrak jangka panjang. Ketika pelanggan tidak memperpanjang kontrak atau mengakhiri kontrak lebih awal, perusahaan kehilangan pendapatan sekaligus menghadapi dampak operasional yang signifikan. Tingkat churn yang tinggi dapat mengindikasikan masalah seperti harga yang kurang kompetitif, kualitas layanan yang buruk, atau pengalaman pelanggan yang tidak memuaskan. Oleh karena itu, penting bagi perusahaan untuk memahami alasan di balik churn pelanggan untuk menyusun strategi yang efektif dalam meningkatkan kepuasan pelanggan, mempertahankan pendapatan, dan menjaga keberlanjutan bisnis.

Perusahaan telekomunikasi juga menginvestasikan dana besar dalam infrastruktur untuk memastikan layanan berjalan dengan baik. Namun, churn pelanggan dapat menyebabkan penurunan pendapatan yang signifikan, yang berdampak negatif pada keuntungan dan tingkat return on investment (ROI). Kehilangan pelanggan membutuhkan biaya yang lebih tinggi untuk akuisisi pelanggan baru dibandingkan mempertahankan pelanggan yang ada. Dalam konteks ini, perusahaan harus menerapkan strategi retensi yang efektif guna mengurangi churn dan menjaga daya saing di pasar yang kompetitif.

Tingkat churn pelanggan yang tinggi adalah tantangan utama yang dapat merusak stabilitas pendapatan perusahaan telekomunikasi. Dengan memahami faktor-faktor yang menyebabkan churn, perusahaan dapat menurunkan tingkat churn secara signifikan, meningkatkan kepuasan pelanggan, dan mempertahankan keberlanjutan bisnis. Hal ini juga memungkinkan perusahaan untuk mengalokasikan sumber daya secara lebih efisien, meningkatkan pengalaman pelanggan, dan menjaga loyalitas mereka.

**Goals:**
Berdasarkan pemasalahan tersebut, perusahaan ingin dapat memprediksi kemungkinan seorang pelanggan berhenti menggunakan layanan mereka (churn) atau tetap mejadi pelanggan. Dengan kemampuan ini, perusahaan dapat fokus pada upaya mempertahankan pealnggan yang memiliki risiko tinggi untuk churn.
Selain itu, perusahaan juga ingin memahami faktor-faktor yang menyebabkan pelanggan berhenti menggunakan layanan mereka. Informasi ini akan membantu perusahaan merancang strategi yang lebi efektif untuk mempertahankan pelanggan, meningkatkan kepuasan, dan memperkuat loyalitas pelanggan terhadap perusahaan.

**Analytic Approach:**
Pendekatan analitik dalam kasus ini adalah dengan membuat, menguji dan menggunakan model machine learning. Model ini bertujuan untuk memprediksi apakah seorang pelanggan akan berhenti mengguankan layanan (churn) atau tetap menggunakan layanan, berdasarkan data riwayat pelanggan.

**Metric Evaluation**
**False Positive** terjadi ketika model secara kelitu memprediksi bahwa seorang pelanggan akan churn, padahal sebenarnya pelangggan tersebut tidak akan churn. Akibatnya, perusahaan mungkin mengalokasikan sumber daya secara tidak efisien, seperti memberikan diskon atau promosi yang sebenarnya tidak diperlukan. 

**False Negative** adalah situasi di mana model memprediksi bahwa seorang pelanggan tidak akan churn, tetapi sebenarnya mereka churn. ini memiliki dampak yang lebih serius, seperti kehilangan pendapatan langsung karena pelanggan tersebut meninggalkan layanana tanpa dilakukan upaya retensi. Selain itu, churn yang tidak terdeteksi dapat menurunkan loyalitas pelanggan dan menciptakan efek domino melaklui ulasan yang mempengaruhi pelanggan lain.

Dalam konteks prediksi churn, **False Negative** biasanya dianggap lebih merugikan dalam konteks prediksi churn karena memiliki peluan yang terlewatkan untuk melakukan intervensi. Jika seorang pelanggan yang sebenarnya berisiko churn salah diklasifikasikan sebagai tidak churn oleh model, perusahaan akan kehilangan kesempatan untuk secara proaktif mempertahankan pelanggan tersebut, yang pada akhirnya mengakibatkan kerugian pendapatan secara langsung.
**Attribute Information**
### Attribute Information

| Attribute | Data Type, Length | Description |
| --- | --- | --- |
| Dependents | object | Menunjukkan apakah pelanggan memiliki tanggungan (seperti anggota keluarga yang bergantung pada mereka) atau tidak. |
| tenure | int64  | Jumlah bulan pelanggan telah mengguanakan layanan perusahaan |
|OnlineSecurity | object | Menunjukkan apakah pelanggan memiliki layanan keamanan online atau tidak |
| OnlineBackup | object | Menunjukkan apakah pelanggan memiliki layanan pecandangan data online atau tidak |
| InternetService | object | Menunjukkan apakah pelanggan berlangganan layanan atau tidak |
| DeviceProtection | object | Menunjukkan apakah pelanggan memiliki perlindungan perangkat atau tidak |
| TechSupport | object | Menunjukkan apakah pelanggan memiliki dukungan teknis (*technical support*) atau tidak |
| Contract | object | jenis kontrak pelanggan berdasarkan durasi, seperti bulanan, tahunan atau lainnya |
| PaperlessBilling | object | Menunjukkan apakah tagihan diberikan dalam bentuk tanpa kerta(digital) atau tidak |
| MonthlyCharges | float64 | Jumlah biaya layanan yang dibebankan kepada pelanggan setiap bulan |
| Churn | object | Menunjukkan apakah pelanggan berhenti menggunakan layanan (churn) atau tetap menjadi pelanggan |
# **Summary**
1. Pemilihan Fitur dan Pembersihan Data:

    Kolom PaperlessBilling diputuskan untuk dihapus karena berdasarkan pengetahuan domain, pelanggan tidak memutuskan untuk pindah ke kompetitor berdasarkan cara mereka menerima tagihan, karena hal ini dapat dengan mudah diubah sesuai permintaan pelanggan.
    Nilai No Service pada beberapa kolom juga dihapus karena dalam analisis ini, pelanggan yang diprediksi adalah pelanggan yang memiliki koneksi internet saja.
    Data yang memiliki nilai duplikat juga dihapus. Setelah pembersihan, data tidak memiliki nilai yang hilang atau nilai NaN.
2. Pendefinisian Fitur X dan y:

    Fitur X terdiri dari semua kolom kecuali kolom Churn, yang digunakan sebagai fitur y. Kolom Churn merupakan indikator apakah pelanggan pindah ke kompetitor atau tidak.
3. Proses Preprocessing:

    Semua kolom kecuali Tenure dan MonthlyCharges diproses menggunakan OneHotEncoder, karena kolom-kolom ini adalah kategori. kolom contract mengguanakn ordinal encoding.
    Kolom Tenure dan MonthlyCharges adalah kolom numerik, sehingga menggunakan Robust Scaler untuk mengurangi dampak nilai ekstrem
4. Evaluasi Model:

    Setelah preprocessing, data divalidasi silang untuk mendapatkan nilai rata-rata recall dan menentukan model terbaik untuk tuning. LOgistic regression,  dengan recall 0.50 hingga 0.59, dengan rata-rata 0.553 dan standar deviasi yang relatife kecil (0.033

    Karena data tidak seimbang, menggunakan random undersampler(RUS), randomoversample(ROS), nearmiss.
Pemilihan Model untuk Tuning:

    Berdasarkan hasil, Logistic Regression dipilih untuk proses tuning karena hasil mereka lebih baik 

setelah tunning:
    Peningkatan Nilai Recall:
        Recall sebelum: 0.6008
        Recall sesudah: 0.8025
        Terdapat peningkatan yang signifikan pada nilai Recall dari 0.6008 menjadi 0.8025, yang menunjukkan bahwa model sekarang mampu mendeteksi lebih banyak kasus positif yang sebenarnya.

# **Kesimpulan bisnis**
1. Fokus pada Fitur Penting:

Jenis Layanan Internet: Pelanggan yang memilih layanan internet tertentu mungkin lebih rentan terhadap churn jika layanan tersebut tidak memenuhi kebutuhan mereka. Misalnya, pelanggan dengan koneksi lambat atau sering terputus dapat merasa tidak puas dan mencari penyedia lain.
Dukungan Teknis: Ketersediaan dan kualitas dukungan teknis berperan penting dalam kepuasan pelanggan. Pelanggan yang tidak mendapatkan bantuan cepat dan efektif saat mengalami masalah lebih mungkin untuk berhenti menggunakan layanan.
Kontrak: Pelanggan dengan kontrak jangka panjang mungkin merasa terikat dan lebih cenderung bertahan, sedangkan pelanggan dengan opsi pembayaran bulanan mungkin lebih mudah churn. Memahami preferensi kontrak dapat membantu dalam merancang penawaran yang lebih menarik.

Strategi Bisnis:

Peningkatan layanan internet dengan penawaran kecepatan yang lebih tinggi atau paket yang disesuaikan.
Pelatihan staf dukungan teknis untuk meningkatkan kecepatan dan kualitas respons.
Menawarkan opsi kontrak fleksibel yang memberikan nilai tambah kepada pelanggan.

2. Strategi Retensi:

Analisis Fitur yang Menyebabkan Churn: Dengan mengidentifikasi fitur yang memiliki dampak besar terhadap churn, perusahaan dapat menyesuaikan strategi retensinya. Misalnya, jika pelanggan dengan paket tertentu lebih cenderung churn, perusahaan dapat menyesuaikan paket tersebut atau menawarkan insentif tambahan.
Penawaran Personal: Pelanggan yang dianggap berisiko tinggi dapat diberi perhatian lebih melalui penawaran yang dipersonalisasi, seperti diskon, peningkatan layanan gratis, atau hadiah loyalitas.

Strategi Bisnis:

Menjalankan program retensi yang proaktif berdasarkan data, seperti menawarkan diskon kepada pelanggan yang menunjukkan tanda-tanda ketidakpuasan.
Mengembangkan layanan pelanggan yang berbasis nilai untuk mendorong loyalitas.

3. Pengelolaan Risiko:

Identifikasi Pelanggan Berisiko Tinggi: Dengan model prediktif, perusahaan dapat secara proaktif mengidentifikasi pelanggan yang berisiko churn sebelum mereka memutuskan untuk pergi. Hal ini memungkinkan perusahaan untuk mengambil langkah-langkah seperti follow-up secara personal atau menawarkan insentif untuk mempertahankan mereka.
Pengurangan Biaya Akuisisi: Mempertahankan pelanggan yang ada lebih murah daripada memperoleh pelanggan baru. Dengan fokus pada pelanggan berisiko tinggi, perusahaan dapat mengurangi churn, yang pada akhirnya mengurangi kebutuhan akan akuisisi pelanggan baru yang mahal.

Strategi Bisnis:

Menggunakan dashboard prediktif yang memantau pelanggan berisiko tinggi secara real-time.
Mengimplementasikan kebijakan retensi yang langsung diaktifkan saat tanda-tanda churn terdeteksi.

4. Optimalisasi Sumber Daya:

Efisiensi Sumber Daya: Dengan mengetahui pelanggan mana yang lebih mungkin untuk churn, perusahaan dapat mengarahkan upaya pemasaran dan sumber daya lainnya ke segmen pelanggan yang paling membutuhkan perhatian. Ini memastikan bahwa usaha pemasaran dan dukungan dilakukan secara lebih efektif.
Targeted Marketing: Alih-alih menjalankan kampanye pemasaran luas, perusahaan dapat memfokuskan kampanye pada segmen pelanggan yang paling berisiko churn, sehingga meningkatkan efisiensi dan efektivitas dari pengeluaran pemasaran.

Strategi Bisnis:

Mengalokasikan tim dukungan pelanggan untuk secara khusus menangani pelanggan berisiko tinggi.
Menggunakan analitik prediktif untuk mengoptimalkan waktu dan tempat kampanye pemasaran.

# Rekomendasi untuk bisnis
strategi bisnis untuk mengurangi churn berdasarkan wawasan yang telah diperoleh:
1. Durasi Kontrak

Analisis: Pelanggan dengan kontrak bulanan lebih cenderung churn dibandingkan dengan pelanggan yang terikat pada kontrak jangka panjang. Hal ini bisa disebabkan oleh fleksibilitas yang lebih besar yang dimiliki pelanggan dengan kontrak bulanan, yang memungkinkan mereka untuk meninggalkan layanan kapan saja tanpa konsekuensi finansial yang besar.

Strategi:
Penawaran Insentif: Tawarkan insentif kepada pelanggan yang bersedia beralih dari kontrak bulanan ke kontrak jangka panjang. Insentif ini bisa berupa diskon bulanan, peningkatan layanan tanpa biaya tambahan, atau fitur eksklusif seperti akses prioritas ke dukungan pelanggan.
Peningkatan Nilai: Komunikasikan dengan jelas manfaat dari kontrak jangka panjang, seperti stabilitas biaya dan perlindungan dari kenaikan harga di masa depan.
Fleksibilitas Kontrak: Berikan opsi kontrak yang lebih fleksibel namun tetap menguntungkan untuk jangka panjang, seperti pengurangan biaya penalti untuk pengakhiran kontrak lebih awal.

2. Retensi Berdasarkan Tenure

Analisis:Pelanggan baru menunjukkan tingkat churn yang lebih tinggi dibandingkan dengan pelanggan yang sudah lama. Ini bisa jadi karena pelanggan baru belum sepenuhnya memahami nilai dari layanan atau masih dalam proses membandingkan dengan penyedia layanan lainnya.

Strategi:
Program Onboarding: Buat program onboarding yang kuat untuk membantu pelanggan baru merasa nyaman dan memahami manfaat dari layanan sejak awal. Ini bisa berupa pelatihan pengguna, panduan penggunaan, atau dukungan personal yang lebih intensif selama bulan pertama.
Program Loyalitas: Implementasikan program loyalitas yang memberikan penghargaan kepada pelanggan atas kesetiaan mereka, seperti poin yang bisa ditukar dengan diskon atau hadiah lainnya.
Feedback Rutin: Kumpulkan umpan balik secara berkala dari pelanggan baru untuk memahami kebutuhan mereka dan menyesuaikan layanan agar lebih sesuai.

3. Evaluasi Layanan Internet

Analisis:Pelanggan yang menggunakan layanan fiber optic cenderung churn lebih tinggi dibandingkan dengan pelanggan DSL. Ini mungkin disebabkan oleh perbedaan dalam harga, kualitas layanan, atau harapan yang tidak terpenuhi terhadap teknologi fiber optic yang biasanya lebih mahal.

Strategi:Penyesuaian Harga: Tinjau struktur harga untuk layanan fiber optic untuk memastikan bahwa pelanggan merasa mereka mendapatkan nilai yang setara dengan harga yang mereka bayar.
Peningkatan Kualitas: Pastikan kualitas layanan fiber optic sesuai dengan harapan pelanggan. Ini termasuk kecepatan internet, stabilitas koneksi, dan responsivitas layanan pelanggan.
Kampanye Edukasi: Luncurkan kampanye edukasi untuk menunjukkan manfaat unik dari layanan fiber optic, seperti kecepatan tinggi dan kapasitas yang lebih besar, yang mungkin tidak dimiliki oleh DSL.

4. Keamanan Online & Dukungan Teknis

Analisis: Keamanan online dan dukungan teknis adalah aspek yang sangat penting bagi pelanggan, dan kekurangan dalam area ini dapat meningkatkan risiko churn. Pelanggan mengharapkan perlindungan terhadap ancaman online serta akses ke dukungan teknis yang responsif dan efektif.

Strategi:
Peningkatan Layanan Keamanan: Investasikan dalam teknologi keamanan terbaru untuk melindungi pelanggan dari ancaman siber. Ini bisa mencakup firewall canggih, antivirus, dan fitur seperti VPN yang disediakan sebagai bagian dari paket layanan.
Dukungan Teknis Proaktif: Tingkatkan dukungan teknis dengan memberikan layanan proaktif seperti pemantauan jaringan yang bisa mendeteksi dan menyelesaikan masalah sebelum pelanggan menyadari adanya masalah. Juga, pastikan tim dukungan mudah dihubungi dan mampu memberikan solusi cepat.
Edukasi Pelanggan: Edukasi pelanggan tentang langkah-langkah yang bisa mereka ambil untuk melindungi diri mereka secara online, serta cara terbaik untuk memanfaatkan dukungan teknis yang tersedia.

# **Rekomendasi Machine learning**
1. Eksplorasi Algoritma Lain

Analisis:
Meskipun Logistic Regression memberikan hasil yang baik, masih ada potensi peningkatan akurasi dengan mencoba algoritma machine learning lainnya yang mungkin lebih cocok untuk dataset yang lebih kompleks atau yang memiliki pola non-linear.

Strategi:
Gradient Boosting: Algoritma ini meningkatkan akurasi model dengan membangun model prediksi secara berurutan, di mana setiap model baru mencoba memperbaiki kesalahan yang dibuat oleh model sebelumnya. Contoh populer termasuk XGBoost, LightGBM, dan CatBoost.
Neural Networks: Jika dataset memiliki pola kompleks, Neural Networks dapat menangkap hubungan non-linear dengan baik. Meskipun lebih memerlukan sumber daya komputasi, model ini sangat berguna untuk dataset besar dengan banyak fitur.

2. Penggunaan Teknik Oversampling Lain

Analisis:
Dalam kasus data yang tidak seimbang, di mana jumlah sampel dalam satu kelas jauh lebih sedikit daripada kelas lainnya, teknik oversampling dapat membantu untuk menyeimbangkan kelas dan meningkatkan performa model prediktif.

Strategi:
ADASYN (Adaptive Synthetic Sampling): ADASYN adalah varian dari SMOTE (Synthetic Minority Over-sampling Technique) yang berfokus pada sampel minoritas yang lebih sulit diklasifikasikan. Teknik ini menciptakan sampel sintetis berdasarkan distribusi yang adaptif terhadap kebutuhan setiap sampel.
Kombinasi Oversampling dan Undersampling: Menggabungkan oversampling pada kelas minoritas dan undersampling pada kelas mayoritas dapat membantu menciptakan dataset yang lebih seimbang, mengurangi risiko overfitting dan meningkatkan generalisasi model.

3. Penambahan Fitur Baru

Analisis:
Menambahkan fitur-fitur baru yang relevan ke dataset dapat memberikan lebih banyak informasi kepada model untuk membuat prediksi yang lebih akurat. Fitur tambahan bisa berasal dari analisis domain atau sumber data eksternal.

Strategi:
Data Demografi: Menambahkan informasi seperti usia, lokasi, pekerjaan, dan tingkat pendidikan pelanggan dapat membantu model memahami perilaku pelanggan dengan lebih baik.
Riwayat Interaksi Pelanggan: Data tentang interaksi pelanggan sebelumnya dengan perusahaan, seperti panggilan ke layanan pelanggan, keluhan, atau penggunaan layanan tambahan, dapat memberikan wawasan lebih dalam tentang kemungkinan churn.
Fitur Temporal: Menambahkan informasi temporal seperti tren penggunaan layanan dalam waktu tertentu, atau musiman, dapat membantu model menangkap pola perilaku pelanggan yang berubah seiring waktu.

4. Peningkatan Jumlah Data

Analisis:
Data yang lebih banyak dan beragam dapat membantu model machine learning untuk generalisasi lebih baik dan menghasilkan prediksi yang lebih akurat. Dataset yang lebih besar juga dapat membantu mengurangi variabilitas dan meningkatkan stabilitas model.

Strategi:
Kumpulkan Data Historis Lebih Banyak: Mengumpulkan data historis yang lebih panjang dapat memberikan model konteks yang lebih kaya, memungkinkan untuk menangkap pola yang mungkin tidak terlihat dalam data yang lebih terbatas.
Sumber Data Beragam: Selain data internal perusahaan, eksplorasi data dari sumber eksternal seperti laporan industri, data sosial media, atau data pasar dapat memberikan tambahan wawasan yang berharga.
Data Augmentation: Jika pengumpulan data tambahan tidak memungkinkan, pertimbangkan teknik data augmentation untuk meningkatkan jumlah data, seperti menambahkan noise ke data yang ada atau membuat variasi baru dari sampel yang ada.

#Yes(1) : Pelanggan tidak melanjutkan layanan dengan perusahaan internet saat ini.
