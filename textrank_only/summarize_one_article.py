from textrank import summarize_textrank
from rouge_score import rouge_scorer

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

summary = summarize_textrank(article, num_sentences=5)
print("Summary:")
print(summary)

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)

print("\nROUGE Scores textrank only:")
for key in ['rouge1', 'rouge2', 'rougeL']:
    score = scores[key]
    print(f"{key.upper()} - F1: {score.fmeasure:.4f}, Precision: {score.precision:.4f}, Recall: {score.recall:.4f}") 