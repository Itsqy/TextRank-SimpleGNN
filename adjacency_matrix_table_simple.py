"""
ADJACENCY MATRIX TABLE - SIMPLE VERSION
=======================================

Tabel sederhana untuk menjelaskan adjacency matrix dengan perhitungan similarity
"""

def create_simple_adjacency_table():
    """
    Membuat tabel adjacency matrix yang sederhana dan mudah dipahami
    """
    
    print("=" * 120)
    print("ADJACENCY MATRIX TABLE - SIMPLE VERSION")
    print("=" * 120)
    
    # Data kalimat
    sentences = [
        "Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.",
        "Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.",
        "Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon.",
        "Menurut seorang fungsionaris PBB Ridwan Hasan, persoalan di Maluku bisa selesai asalkan pemerintah dan aparat keamanan serius menangani setiap persoalan di Maluku secara komprehensif dan bijaksana.",
        "Itulah sebabnya, PBB wilayah Maluku akan menjadikan penyelesaian konflik sebagai agenda utama partai."
    ]
    
    # Adjacency matrix
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    # Kata kunci setiap kalimat
    keywords = [
        ["Liputan6", "Ambon", "konflik", "pemerintah"],
        ["Ambon", "konflik", "Maluku"],
        ["Ambon", "Maluku", "PBB"],
        ["Ambon", "Maluku", "pemerintah", "PBB", "Ridwan"],
        ["konflik", "Maluku", "PBB", "agenda"]
    ]
    
    print("\n📋 KALIMAT-KALIMAT:")
    print("-" * 120)
    for i, sentence in enumerate(sentences):
        print(f"K{i+1}: {sentence}")
        print(f"    Kata Kunci: {', '.join(keywords[i])}")
        print()
    
    print("=" * 120)
    print("📊 ADJACENCY MATRIX TABLE")
    print("=" * 120)
    
    # Header tabel
    print(f"{'Pasangan':<12} {'Similarity':<12} {'Kata Sama':<25} {'Kata Berbeda':<40} {'Keterangan':<20}")
    print("-" * 120)
    
    # Isi tabel
    for i in range(5):
        for j in range(5):
            if i == j:
                # Self-similarity
                print(f"K{i+1} ↔ K{j+1}    {'1.000':<12} {'Identik':<25} {'-':<40} {'Kalimat sama':<20}")
            else:
                similarity = adj_matrix[i][j]
                common_words = set(keywords[i]) & set(keywords[j])
                diff_words_i = set(keywords[i]) - set(keywords[j])
                diff_words_j = set(keywords[j]) - set(keywords[i])
                
                common_str = ', '.join(common_words) if common_words else 'Tidak ada'
                diff_str = f"K{i+1}:{','.join(diff_words_i)} | K{j+1}:{','.join(diff_words_j)}"
                
                # Keterangan berdasarkan similarity
                if similarity >= 0.6:
                    keterangan = "Sangat similar"
                elif similarity >= 0.4:
                    keterangan = "Similar"
                elif similarity >= 0.2:
                    keterangan = "Sedikit similar"
                else:
                    keterangan = "Tidak similar"
                
                print(f"K{i+1} ↔ K{j+1}    {similarity:<12.3f} {common_str:<25} {diff_str:<40} {keterangan:<20}")
    
    print("\n" + "=" * 120)
    print("🎯 INTERPRETASI SIMILARITY")
    print("=" * 120)
    
    print("\n📈 RANGE SIMILARITY:")
    print("• 0.8 - 1.0: Sangat sangat similar (hampir identik)")
    print("• 0.6 - 0.8: Sangat similar (banyak kata yang sama)")
    print("• 0.4 - 0.6: Similar (sedang, beberapa kata sama)")
    print("• 0.2 - 0.4: Sedikit similar (sedikit kata yang sama)")
    print("• 0.0 - 0.2: Tidak similar (hampir tidak ada kata yang sama)")
    
    print("\n🔗 KONEKSI TERKUAT:")
    print("• K3 ↔ K4: 0.7 (Sangat similar - 3 kata sama: Ambon, Maluku, PBB)")
    print("• K2 ↔ K5: 0.6 (Sangat similar - 2 kata sama: konflik, Maluku)")
    print("• K4 ↔ K5: 0.5 (Similar - 2 kata sama: Maluku, PBB)")
    
    print("\n🔗 KONEKSI TERLEMAH:")
    print("• K1 ↔ K5: 0.1 (Tidak similar - hanya 1 kata sama: konflik)")
    print("• K1 ↔ K3: 0.2 (Sedikit similar - 1 kata sama: Ambon)")
    print("• K1 ↔ K4: 0.3 (Sedikit similar - 1 kata sama: Ambon)")
    
    print("\n" + "=" * 120)
    print("💡 KEGUNAAN UNTUK GRAPH")
    print("=" * 120)
    
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

def create_visual_adjacency_matrix():
    """
    Membuat visualisasi adjacency matrix dengan emoji
    """
    
    print("\n" + "=" * 120)
    print("🔥 VISUAL ADJACENCY MATRIX")
    print("=" * 120)
    
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    print("\nAdjacency Matrix dengan Visual:")
    print("     K1    K2    K3    K4    K5")
    print("     -----------------------------")
    
    for i, row in enumerate(adj_matrix):
        print(f"K{i+1} |", end=" ")
        for j, val in enumerate(row):
            if i == j:
                print(f" 🔵{val:.1f} ", end="")  # Self-connection
            elif val >= 0.7:
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
    print("🔵 = Self-connection (1.0)")
    print("🔴 = Sangat similar (≥0.7)")
    print("🟠 = Similar (0.5-0.7)")
    print("🟡 = Sedang (0.3-0.5)")
    print("🟢 = Rendah (0.1-0.3)")
    print("⚪ = Tidak similar (<0.1)")

if __name__ == "__main__":
    print("ADJACENCY MATRIX TABLE - SIMPLE VERSION")
    print("=" * 120)
    
    # Create simple table
    create_simple_adjacency_table()
    
    # Create visual matrix
    create_visual_adjacency_matrix()
    
    print("\n" + "=" * 120)
    print("✅ ADJACENCY MATRIX TABLE SELESAI!")
    print("=" * 120) 