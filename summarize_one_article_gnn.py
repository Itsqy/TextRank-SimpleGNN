from gnn import summarize
from rouge_score import rouge_scorer

# Replace these with your article and reference summary
# article = """
# Liputan6 . com , Sukabumi : Menjelang Lebaran , harga daging sapi dan ayam diperkirakan melonjak tinggi . Ini akibat sistem distribusi dan mahalnya harga pakan . Demikian Menteri Pertanian Anton Apriantono saat mengunjungi kelompok peternak sapi dan ayam ras di Sukabumi , Jawa Barat , Kamis ( 20/9 ) . Menurut Anton , supaya distribusi daging sapi dan ayam lebih teratur Departemen Pertanian akan mengambil sejumlah langkah . Meski harga naik , Anton menjamin stok daging sapi dan ayam untuk Lebaran masih mencukupi . Disinggung mengenai impor jeroan sapi , Mentan membantah bahwa impor jeroan sapi hanya dari dua negara . Deptan , kata Anton , sudah mendatangkan jeroan sapi dari beberapa negara untuk memenuhi kebutuhan nasional . Sementara itu , harga bahan kebutuhan pokok di beberapa pasar di Jakarta masih stabil . Tapi beberapa bahan untuk membuat kue , seperti terigu , kacang-kacangan , mentega , dan telur naik cukup tinggi . Untuk mengamankan kebutuhan Lebaran , Presiden Susilo Bambang Yudhoyono telah menggelar rapat persiapan Idul Fitri . Sebelumnya Menteri Perdagangan dan Pertanian melakukan inspeksi mendadak ke sejumlah pasar . Tak hanya itu , Departemen Perdagangan juga telah menyiapkan posko untuk memantau harga dan stok kebutuhan pokok [ baca : Mari : Stok Aman dan Harga Stabil ] . Sementara itu , Umi warga Duri Kepa , Jakarta Barat yang ditemui reporter SCTV Nova Rini , mengaku , melonjaknya harga kebutuhan pokok sangat mempengaruhi keperluan sehari-hari . Umi , misalnya . Semula dia cukup menyediakan uang sebesar Rp 20 ribu untuk kebutuhan makan suami dan tiga anaknya . Tapi , kini kebutuhan melonjak hingga mencapai Rp 30 ribu . " Akhirnya , kami hanya bisa makan seadanya , " kata Umi . Erik , suami Umi , mengaku penghasilannya sebulan hanya Rp 800 ribu . Uang sebanyak itu digunakan untuk bayar kontrakan rumah , biaya sekolah , makan sehari-hari , serta ongkos kerja . " Gaji saya hanya cukup untuk satu minggu , " kata Erik . ( IAN/Tim Liputan 6 SCTV ) .
# """

# reference_summary = """
# Meski harga naik , Anton menjamin stok daging sapi dan ayam untuk Lebaran masih mencukupi . Semula dia cukup menyediakan uang sebesar Rp 20 ribu untuk kebutuhan makan suami dan tiga anaknya . Sementara itu , harga bahan kebutuhan pokok di beberapa pasar di Jakarta masih stabil . com , Sukabumi : Menjelang Lebaran , harga daging sapi dan ayam diperkirakan melonjak tinggi . Sementara itu , Umi warga Duri Kepa , Jakarta Barat yang ditemui reporter SCTV Nova Rini , mengaku , melonjaknya harga kebutuhan pokok sangat mempengaruhi keperluan sehari-hari .
# """

# Replace these with your article and reference summary
# article = """
# Liputan6 . com , Bantul : Bayi berusia 1 , 5 bulan ditemukan tergeletak di rumah bidan Hery Ningsih di Bantul , Yogyakarta , baru-baru ini . Kondisi bayi mungil itu mengenaskan karena hanya terbungkus kardus . Penemuan itu terjadi sekitar pukul 02 . 30 WIB , tatkala terdengar tangisan bayi . Semula , Hery dan suaminya menyangka ada pasien yang hendak memeriksakan bayinya . Namun , Hery terkejut karena melihat bayi terbungkus kardus dari balik pintu rumahnya . Diduga , ibu si bayi memang sengaja meninggalkan anaknya . Buktinya si ibu menyimpan popok , alat mandi , dan susu bubuk di dalam kardus . Seorang tetangga mengaku melihat dua orang remaja mengendarai sepeda motor masuk ke rumah Hery . Lantas , kedua orang tadi menyimpan sebuah kardus berukuran besar . Kasus ini sekarang ditangani jajaran Kepolisian Resor Bantul . ( KEN/Wiwik Susilo dan Mardianto ) .
# """

# reference_summary = """
# Bidan Hery terkejut melihat bayi mungil terbungkus kardus di depan pintu rumahnya di Bantul , Yogyakarta . Bayi itu sengaja dibuang oleh ibunya yang datang menyelinap di malam hari .
# """
# article_data = {
#     "id": 13020,
#   "url": "https://www.liputan6.com/news/read/13020/bi-dinilai-masih-akan-dirundung-masalah",
#   "clean_article": [
#     ["Liputan6", ".", "com", ",", "Jakarta", ":", "Bank", "Indonesia", "dinilai", "masih", "akan", "menghadapi", "situasi", "sulit", "kendati", "Bank", "Sentral", "Amerika", "Serikat", "(", "The", "FED", ")", "terus", "menurunkan", "tingkat", "suku", "bunga", "yang", "dimiliki", "."],
#     ["Penilaian", "itu", "dikemukakan", "pengamat", "ekonomi", "Didiek", "J", "."],
#     ["Rachbini", ",", "di", "Jakarta", ",", "baru-baru", "ini", "."],
#     ["Menurut", "perhitungan", "Didiek", ",", "dalam", "tahun", "ini", ",", "The", "FED", "telah", "lima", "kali", "menurunkan", "nilai", "suku", "bunga", "yang", "mereka", "miliki", "."],
#     ["Bahkan", ",", "Didiek", "memperkirakan", ",", "tingkat", "suku", "bunga", "The", "FED", "akan", "diturunkan", "hingga", "menjadi", "empat", "persen", "."],
#     ["Dengan", "keadaan", "itu", ",", "tambah", "Didiek", ",", "di", "atas", "kertas", "dapat", "dimanfaatkan", "BI", "untuk", "meningkatkan", "suku", "bunga", "BI", "sebagai", "upaya", "mempertahankan", "nilai", "tukar", "rupiah", "."],
#     ["Namun", "demikian", ",", "Didiek", "pesimistis", ",", "hal", "itu", "akan", "tercapai", "mengingat", "kondisi", "bangsa", "masih", "carut", "marut", "."],
#     ["\"", "Jika", "keadaan", "terus", "seperti", "ini", ",", "tak", "tertutup", "kemungkinan", ",", "BI", "akan", "tetap", "memberlakukan", "nilai", "suku", "bunga", "tinggi", ",", "\"", "ujar", "Didiek", "."],
#     ["Sementara", "itu", ",", "The", "FED", "terpaksa", "menurunkan", "tingkat", "suku", "bunga", "karena", "pertembuhan", "ekonomi", "di", "negeri", "Paman", "Sam", "terus", "melemah", "."],
#     ["Padahal", ",", "selama", "ini", ",", "AS", "menjadi", "pasar", "ekspor", "penting", "untuk", "Indonesia", ".", "(", "ICH/Fahmi", "Ihsan", "dan", "Donny", "Indradi", ")", "."]
#   ],
#   "clean_summary": [
#     ["Kendati", "Bank", "Sentral", "AS", "menurunkan", "suku", "bunganya", ",", "namun", "BI", "dinilai", "masih", "akan", "menemui", "masa", "sulit", "."],
#     ["Suku", "bunga", "Bank", "Sentral", "AS", "akan", "diturunkan", "menjadi", "empat", "persen", "."]
#   ],
#   "extractive_summary": [0, 4]
# }
# #  konglomerat : 
article_data = {
    "id": 1,
    "url": "https://www.liputan6.com/news/read/1/batas-penyerahan-aset-dua-pekan-lagi",
    "clean_article": [
        ["Liputan6", ".", "com", ",", "Jakarta", ":", "Pemerintah", "masih", "memberikan", "waktu", "dua", "minggu", "lagi", "kepada", "seluruh", "konglomerat", "yang", "telah", "menandatangani", "perjanjian", "pengembalian", "bantuan", "likuiditas", "Bank", "Indonesia", "dengan", "jaminan", "aset", "(", "MSAA", ")", ",", "untuk", "secepatnya", "menyerahkan", "jaminan", "pribadi", "serta", "aset", "."],
        ["Jika", "lewat", "dari", "tenggat", "tersebut", ",", "pemerintah", "akan", "menerapkan", "tindakan", "hukum", "."],
        ["Hal", "tersebut", "dikemukakan", "Menteri", "Koordinator", "Bidang", "Perekonomian", "Rizal", "Ramli", "di", "Jakarta", ",", "baru-baru", "ini", "."],
        ["Rizal", "mengakui", "bahwa", "permintaan", "untuk", "meminta", "jaminan", "pribadi", "atau", "personal", "guarantee", "pada", "awalnya", "ditentang", "sejumlah", "konglomerat", "."],
        ["Sebab", "para", "debitor", "menganggap", "tindakan", "tersebut", "memungkinkan", "pemerintah", "untuk", "menyita", "seluruh", "aset", "mereka", "baik", "yang", "berada", "di", "dalam", "maupun", "luar", "negeri", "."],
        ["Sejauh", "ini", ",", "penilaian", "jaminan", "MSAA", "baru", "dilakukan", "atas", "aset", "milik", "Grup", "Salim", "."],
        ["Tetapi", ",", "nilai", "aset", "yang", "dijaminkan", "Kelompok", "Salim", "atas", "utang", "BLBI", "Bank", "Central", "Asia", "diperkirakan", "tak", "lebih", "dari", "Rp", "20", "triliun", "."],
        ["Padahal", ",", "kewajiban", "mereka", "mencapai", "Rp", "52", "triliun", "."],
        ["Sementara", "itu", ",", "pemerintah", "dengan", "DPR", "sepakat", ",", "hingga", "akhir", "Oktober", "mendatang", ",", "para", "konglomerat", "penandatangan", "MSAA", "harus", "sudah", "menutupi", "kekurangan", "mereka", "dengan", "menyerahkan", "aset", "baru", "."],
        ["Selain", "itu", ",", "para", "pengutang", "tersebut", "diwajibkan", "memberikan", "jaminan", "pribadinya", ".", "(", "TNA/Merdi", "Sofansyah", "dan", "Anto", "Susanto", ")", "."]
    ],
    "clean_summary": [
        ["Pemerintah", "memberikan", "tenggat", "14", "hari", "kepada", "para", "konglomerat", "penandatangan", "MSAA", "untuk", "menyerahkan", "aset", "."],
        ["Jika", "mangkir", ",", "mereka", "bakal", "dihukum", "."]
    ],
    "extractive_summary": [1, 8]
}
# Convert clean_article and clean_summary to text
article = ' '.join([' '.join(sent) for sent in article_data['clean_article']])
reference_summary = ' '.join([' '.join(sent) for sent in article_data['clean_summary']])

# 

summary, _ = summarize(article, num_sentences=5)
print("Summary textrank + gnn:")
print(summary)

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)

print("\nROUGE Scores textrank + gnn:")
for key in ['rouge1', 'rouge2', 'rougeL']:
    score = scores[key]
    print(f"{key.upper()} - F1: {score.fmeasure:.4f}, Precision: {score.precision:.4f}, Recall: {score.recall:.4f}") 