from gnn import summarize as summarize_gnn
from textrank_only.textrank import summarize_textrank
from rouge_score import rouge_scorer

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


print("=== Comparison: TextRank + GNN vs TextRank Only ===")
print(f"Article ID: {article_data['id']}")
print(f"Article URL: {article_data['url']}")

print("\n--- Original Article ---")
print(article)

print("\n--- Reference Summary ---")
print(reference_summary)

print("\n=================================================================================")
print("GENERATING SUMMARIES...")
print("=================================================================================")

# Summarize with TextRank + GNN
print("\n--- TextRank + GNN Summary ---")
summary_gnn, _ = summarize_gnn(article, num_sentences=5)
print(summary_gnn)

# Summarize with TextRank only
print("\n--- TextRank Only Summary ---")
summary_textrank = summarize_textrank(article, num_sentences=5)
print(summary_textrank)

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores_gnn = scorer.score(reference_summary, summary_gnn)
scores_textrank = scorer.score(reference_summary, summary_textrank)

print("\n=================================================================================")
print("ROUGE SCORE COMPARISON")
print("=================================================================================")
print(f"{'Metric':<10} | {'F1 (GNN)':>9} | {'F1 (TR)':>9} | {'Prec (GNN)':>10} | {'Prec (TR)':>10} | {'Rec (GNN)':>10} | {'Rec (TR)':>10}")
print('-'*80)
for key in ['rouge1', 'rouge2', 'rougeL']:
    g = scores_gnn[key]
    t = scores_textrank[key]
    print(f"{key.upper():<10} | {g.fmeasure:.4f}   | {t.fmeasure:.4f}   | {g.precision:.4f}   | {t.precision:.4f}   | {g.recall:.4f}   | {t.recall:.4f}")

print("\n=================================================================================")
print("ANALYSIS")
print("=================================================================================")
# Calculate improvements
rouge1_improvement = ((scores_gnn['rouge1'].fmeasure - scores_textrank['rouge1'].fmeasure) / scores_textrank['rouge1'].fmeasure) * 100
rouge2_improvement = ((scores_gnn['rouge2'].fmeasure - scores_textrank['rouge2'].fmeasure) / scores_textrank['rouge2'].fmeasure) * 100
rougeL_improvement = ((scores_gnn['rougeL'].fmeasure - scores_textrank['rougeL'].fmeasure) / scores_textrank['rougeL'].fmeasure) * 100

print("Improvement of TextRank + GNN over TextRank Only:")
print(f"ROUGE1: {rouge1_improvement:+.2f}% ({'BETTER' if rouge1_improvement > 0 else 'WORSE' if rouge1_improvement < 0 else 'SAME'})")
print(f"ROUGE2: {rouge2_improvement:+.2f}% ({'BETTER' if rouge2_improvement > 0 else 'WORSE' if rouge2_improvement < 0 else 'SAME'})")
print(f"ROUGEL: {rougeL_improvement:+.2f}% ({'BETTER' if rougeL_improvement > 0 else 'WORSE' if rougeL_improvement < 0 else 'SAME'})")

print("\n=================================================================================")
print("CONCLUSION")
print("=================================================================================")
best_improvement = max(rouge1_improvement, rouge2_improvement, rougeL_improvement)
worst_improvement = min(rouge1_improvement, rouge2_improvement, rougeL_improvement)

if best_improvement > 0:
    print(f"✅ TextRank + GNN shows improvement over TextRank Only!")
    print(f"   Best improvement: {best_improvement:+.2f}%")
else:
    print(f"❌ TextRank + GNN does not show improvement over TextRank Only.")
    print(f"   Worst performance: {worst_improvement:+.2f}%")

print("\nNote: This is a single article test. For more reliable comparison, run evaluation on multiple articles.") 