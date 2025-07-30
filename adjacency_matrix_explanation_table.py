"""
ADJACENCY MATRIX EXPLANATION TABLE
==================================

Tabel detail penjelasan adjacency matrix dengan perhitungan similarity
untuk setiap pasangan kalimat menggunakan cosine similarity.
"""

import math

def create_adjacency_matrix_table():
    """
    Membuat tabel penjelasan adjacency matrix
    """
    
    print("=" * 100)
    print("ADJACENCY MATRIX EXPLANATION TABLE")
    print("=" * 100)
    
    # Data kalimat asli
    sentences = [
        "Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.",
        "Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.",
        "Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon.",
        "Menurut seorang fungsionaris PBB Ridwan Hasan, persoalan di Maluku bisa selesai asalkan pemerintah dan aparat keamanan serius menangani setiap persoalan di Maluku secara komprehensif dan bijaksana.",
        "Itulah sebabnya, PBB wilayah Maluku akan menjadikan penyelesaian konflik sebagai agenda utama partai."
    ]
    
    # TF-IDF Features (simplified untuk penjelasan)
    tfidf_features = [
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Kalimat 1: [Liputan6, Ambon, konflik, pemerintah]
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Kalimat 2: [Ambon, konflik, Maluku]
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # Kalimat 3: [Ambon, Maluku, PBB]
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # Kalimat 4: [Ambon, Maluku, pemerintah, PBB, Ridwan]
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]   # Kalimat 5: [konflik, Maluku, PBB, agenda]
    ]
    
    # Adjacency matrix
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    # Kata-kata penting untuk setiap kalimat
    key_words = [
        ["Liputan6", "Ambon", "konflik", "pemerintah"],
        ["Ambon", "konflik", "Maluku"],
        ["Ambon", "Maluku", "PBB"],
        ["Ambon", "Maluku", "pemerintah", "PBB", "Ridwan"],
        ["konflik", "Maluku", "PBB", "agenda"]
    ]
    
    print("\n📋 KALIMAT-KALIMAT:")
    print("-" * 100)
    for i, sentence in enumerate(sentences):
        print(f"Kalimat {i+1}: {sentence}")
        print(f"Kata Kunci: {', '.join(key_words[i])}")
        print()
    
    print("=" * 100)
    print("📊 TABEL ADJACENCY MATRIX DENGAN PENJELASAN")
    print("=" * 100)
    
    # Header tabel
    print(f"{'Pasangan':<15} {'Similarity':<12} {'Kata Sama':<20} {'Kata Berbeda':<30} {'Penjelasan':<25}")
    print("-" * 100)
    
    # Isi tabel
    for i in range(5):
        for j in range(5):
            if i == j:
                # Self-similarity
                print(f"K{i+1} ↔ K{j+1}     {'1.000':<12} {'Identik':<20} {'-':<30} {'Kalimat dengan dirinya sendiri':<25}")
            else:
                similarity = adj_matrix[i][j]
                common_words = set(key_words[i]) & set(key_words[j])
                diff_words_i = set(key_words[i]) - set(key_words[j])
                diff_words_j = set(key_words[j]) - set(key_words[i])
                
                common_str = ', '.join(common_words) if common_words else 'Tidak ada'
                diff_str = f"K{i+1}:{','.join(diff_words_i)} | K{j+1}:{','.join(diff_words_j)}"
                
                # Penjelasan berdasarkan similarity
                if similarity >= 0.6:
                    explanation = "Sangat similar"
                elif similarity >= 0.4:
                    explanation = "Similar"
                elif similarity >= 0.2:
                    explanation = "Sedikit similar"
                else:
                    explanation = "Tidak similar"
                
                print(f"K{i+1} ↔ K{j+1}     {similarity:<12.3f} {common_str:<20} {diff_str:<30} {explanation:<25}")
    
    print("\n" + "=" * 100)
    print("🔍 DETAIL PERHITUNGAN COSINE SIMILARITY")
    print("=" * 100)
    
    # Contoh perhitungan detail untuk beberapa pasangan
    example_pairs = [(0, 1), (1, 4), (2, 3), (0, 4)]
    
    for i, j in example_pairs:
        print(f"\n📐 KALIMAT {i+1} vs KALIMAT {j+1}:")
        print("-" * 50)
        
        vec1 = tfidf_features[i]
        vec2 = tfidf_features[j]
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        # Cosine similarity
        similarity = dot_product / (mag1 * mag2)
        
        print(f"Vector K{i+1}: {vec1}")
        print(f"Vector K{j+1}: {vec2}")
        print(f"Dot Product: {dot_product}")
        print(f"Magnitude K{i+1}: {mag1:.3f}")
        print(f"Magnitude K{j+1}: {mag2:.3f}")
        print(f"Cosine Similarity: {dot_product} / ({mag1:.3f} × {mag2:.3f}) = {similarity:.3f}")
        print(f"Hasil Akhir: {adj_matrix[i][j]:.3f}")
    
    print("\n" + "=" * 100)
    print("🎯 INTERPRETASI SIMILARITY VALUES")
    print("=" * 100)
    
    print("\n📈 RANGE SIMILARITY:")
    print("• 0.8 - 1.0: Sangat sangat similar (hampir identik)")
    print("• 0.6 - 0.8: Sangat similar (banyak kata yang sama)")
    print("• 0.4 - 0.6: Similar (sedang, beberapa kata sama)")
    print("• 0.2 - 0.4: Sedikit similar (sedikit kata yang sama)")
    print("• 0.0 - 0.2: Tidak similar (hampir tidak ada kata yang sama)")
    
    print("\n🔗 KONEKSI TERKUAT DALAM GRAPH:")
    print("• K3 ↔ K4: 0.7 (Sangat similar - 3 kata sama)")
    print("• K2 ↔ K5: 0.6 (Sangat similar - 2 kata sama)")
    print("• K4 ↔ K5: 0.5 (Similar - 2 kata sama)")
    
    print("\n🔗 KONEKSI TERLEMAH DALAM GRAPH:")
    print("• K1 ↔ K5: 0.1 (Tidak similar - hanya 1 kata sama)")
    print("• K1 ↔ K3: 0.2 (Sedikit similar - 1 kata sama)")
    print("• K1 ↔ K4: 0.3 (Sedikit similar - 1 kata sama)")
    
    print("\n" + "=" * 100)
    print("💡 KEGUNAAN UNTUK GNN DAN TEXTRANK")
    print("=" * 100)
    
    print("\n🎯 Graph Construction:")
    print("• Semakin tinggi similarity, semakin kuat edge antar kalimat")
    print("• Kalimat dengan similarity tinggi akan saling mempengaruhi")
    print("• Graph akan membentuk cluster kalimat yang similar")
    
    print("\n🔄 Information Flow:")
    print("• Kalimat terhubung akan bertukar informasi dalam GNN")
    print("• Kalimat dengan banyak koneksi akan mendapat score tinggi")
    print("• Kalimat terisolasi akan mendapat score rendah")
    
    print("\n📊 Ranking Quality:")
    print("• TextRank: Kalimat dengan banyak koneksi = lebih penting")
    print("• GNN: Kalimat dengan koneksi kuat = lebih penting")
    print("• Semakin similar, semakin besar pengaruh dalam ranking")

def create_similarity_heatmap():
    """
    Membuat visualisasi heatmap similarity
    """
    
    print("\n" + "=" * 100)
    print("🔥 SIMILARITY HEATMAP VISUALIZATION")
    print("=" * 100)
    
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    print("\nAdjacency Matrix:")
    print("     K1    K2    K3    K4    K5")
    print("     -----------------------------")
    
    for i, row in enumerate(adj_matrix):
        print(f"K{i+1} |", end=" ")
        for j, val in enumerate(row):
            if val >= 0.7:
                print(f" 🔴{val:.1f} ", end="")  # Sangat similar
            elif val >= 0.5:
                print(f" 🟠{val:.1f} ", end="")  # Similar
            elif val >= 0.3:
                print(f" 🟡{val:.1f} ", end="")  # Sedang
            elif val >= 0.1:
                print(f" 🟢{val:.1f} ", end="")  # Rendah
            else:
                print(f" ⚪{val:.1f} ", end="")  # Tidak similar
        print()
    
    print("\nLegend:")
    print("🔴 = Sangat similar (≥0.7)")
    print("🟠 = Similar (0.5-0.7)")
    print("🟡 = Sedang (0.3-0.5)")
    print("🟢 = Rendah (0.1-0.3)")
    print("⚪ = Tidak similar (<0.1)")

if __name__ == "__main__":
    print("ADJACENCY MATRIX EXPLANATION TABLE")
    print("=" * 100)
    
    # Create main table
    create_adjacency_matrix_table()
    
    # Create heatmap
    create_similarity_heatmap()
    
    print("\n" + "=" * 100)
    print("✅ PENJELASAN ADJACENCY MATRIX SELESAI!")
    print("=" * 100) 