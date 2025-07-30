"""
SIMPLE GNN CALCULATION EXAMPLE
==============================

Contoh kalkulasi sederhana untuk SimpleGNN dengan perhitungan manual
step-by-step untuk memahami bagaimana GNN bekerja.
"""

import numpy as np
import math

def simple_gnn_calculation_example():
    """
    Contoh kalkulasi SimpleGNN dengan perhitungan manual
    """
    
    print("=" * 100)
    print("SIMPLE GNN CALCULATION EXAMPLE")
    print("=" * 100)
    
    # Data kalimat
    sentences = [
        "Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.",
        "Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.",
        "Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon.",
        "Menurut seorang fungsionaris PBB Ridwan Hasan, persoalan di Maluku bisa selesai asalkan pemerintah dan aparat keamanan serius menangani setiap persoalan di Maluku secara komprehensif dan bijaksana.",
        "Itulah sebabnya, PBB wilayah Maluku akan menjadikan penyelesaian konflik sebagai agenda utama partai."
    ]
    
    # Adjacency Matrix (Graph Structure)
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    # Node Features (TF-IDF vectors - simplified)
    # Setiap kalimat direpresentasikan sebagai vector 4 dimensi
    node_features = [
        [1.0, 1.0, 1.0, 1.0],  # Kalimat 1: [Liputan6, Ambon, konflik, pemerintah]
        [0.0, 1.0, 1.0, 0.0],  # Kalimat 2: [Ambon, konflik]
        [0.0, 1.0, 0.0, 0.0],  # Kalimat 3: [Ambon]
        [0.0, 1.0, 0.0, 1.0],  # Kalimat 4: [Ambon, pemerintah]
        [0.0, 0.0, 1.0, 0.0]   # Kalimat 5: [konflik]
    ]
    
    print("\n📋 DATA INPUT:")
    print("-" * 100)
    
    print("\n1️⃣ KALIMAT-KALIMAT:")
    for i, sentence in enumerate(sentences):
        print(f"Kalimat {i+1}: {sentence}")
    
    print("\n2️⃣ ADJACENCY MATRIX (Graph Structure):")
    print("     K1    K2    K3    K4    K5")
    print("     -----------------------------")
    for i, row in enumerate(adj_matrix):
        print(f"K{i+1} |", end=" ")
        for val in row:
            print(f" {val:.1f} ", end="")
        print()
    
    print("\n3️⃣ NODE FEATURES (TF-IDF Vectors):")
    print("     F1    F2    F3    F4")
    print("     ---------------------")
    for i, features in enumerate(node_features):
        print(f"K{i+1} |", end=" ")
        for val in features:
            print(f" {val:.1f} ", end="")
        print()
    
    print("\n" + "=" * 100)
    print("🔧 STEP-BY-STEP GNN CALCULATION")
    print("=" * 100)
    
    # Step 1: Message Passing
    print("\n📤 STEP 1: MESSAGE PASSING")
    print("-" * 50)
    print("Setiap node mengirim pesan ke tetangganya berdasarkan adjacency matrix")
    
    # Menghitung pesan yang diterima setiap node
    messages_received = []
    
    for i in range(5):  # Untuk setiap node
        print(f"\n📍 Node {i+1} menerima pesan dari:")
        node_messages = []
        
        for j in range(5):  # Dari setiap node lain
            if i != j and adj_matrix[i][j] > 0:
                # Pesan = adjacency_weight × neighbor_features
                message = [adj_matrix[i][j] * val for val in node_features[j]]
                node_messages.append(message)
                print(f"   Node {j+1}: {adj_matrix[i][j]:.1f} × {node_features[j]} = {message}")
        
        messages_received.append(node_messages)
    
    # Step 2: Aggregation
    print("\n📥 STEP 2: AGGREGATION")
    print("-" * 50)
    print("Menggabungkan semua pesan yang diterima (SUM)")
    
    aggregated_messages = []
    
    for i in range(5):
        if messages_received[i]:
            # Sum semua pesan
            aggregated = [0.0] * 4
            for message in messages_received[i]:
                for k in range(4):
                    aggregated[k] += message[k]
            
            aggregated_messages.append(aggregated)
            print(f"Node {i+1} aggregated: {aggregated}")
        else:
            aggregated_messages.append([0.0] * 4)
            print(f"Node {i+1} aggregated: [0.0, 0.0, 0.0, 0.0] (no messages)")
    
    # Step 3: Update Node Features
    print("\n🔄 STEP 3: UPDATE NODE FEATURES")
    print("-" * 50)
    print("New_Features = Original_Features + Aggregated_Messages")
    
    updated_features = []
    
    for i in range(5):
        new_features = []
        for j in range(4):
            new_val = node_features[i][j] + aggregated_messages[i][j]
            new_features.append(new_val)
        
        updated_features.append(new_features)
        print(f"Node {i+1}: {node_features[i]} + {aggregated_messages[i]} = {new_features}")
    
    # Step 4: Final Scoring
    print("\n🎯 STEP 4: FINAL SCORING")
    print("-" * 50)
    print("Score = Sum dari semua features yang diupdate")
    
    final_scores = []
    
    for i in range(5):
        score = sum(updated_features[i])
        final_scores.append(score)
        print(f"Node {i+1} score: {updated_features[i]} → Sum = {score:.3f}")
    
    # Step 5: Ranking
    print("\n🏆 STEP 5: RANKING")
    print("-" * 50)
    
    # Sort berdasarkan score
    ranked_indices = sorted(range(5), key=lambda i: final_scores[i], reverse=True)
    
    print("Ranking berdasarkan score (descending):")
    for rank, idx in enumerate(ranked_indices):
        print(f"Rank {rank+1}: Kalimat {idx+1} (Score: {final_scores[idx]:.3f})")
    
    print("\n" + "=" * 100)
    print("📊 COMPARISON: MANUAL vs AUTOMATED")
    print("=" * 100)
    
    # Simulasi GNN layer sederhana
    print("\n🤖 SIMULASI GNN LAYER:")
    
    # Weight matrix sederhana (4x4)
    W = [
        [0.5, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.3, 0.1],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.1, 0.2, 0.6]
    ]
    
    print("\nWeight Matrix W:")
    for row in W:
        print(f"    {row}")
    
    # GNN Update: H' = σ(A × H × W)
    print("\nGNN Update Formula: H' = σ(A × H × W)")
    print("Dimana:")
    print("• A = Adjacency Matrix")
    print("• H = Node Features")
    print("• W = Weight Matrix")
    print("• σ = Activation Function (ReLU)")
    
    # Matrix multiplication: A × H
    AH = np.dot(adj_matrix, node_features)
    print(f"\nA × H = {AH.tolist()}")
    
    # Matrix multiplication: (A × H) × W
    AHW = np.dot(AH, W)
    print(f"\n(A × H) × W = {AHW.tolist()}")
    
    # Apply ReLU activation
    def relu(x):
        return max(0, x)
    
    AHW_relu = [[relu(val) for val in row] for row in AHW]
    print(f"\nσ(A × H × W) = {AHW_relu}")
    
    # Final scores
    gnn_scores = [sum(row) for row in AHW_relu]
    print(f"\nGNN Scores: {gnn_scores}")
    
    # Ranking
    gnn_ranked = sorted(range(5), key=lambda i: gnn_scores[i], reverse=True)
    print("\nGNN Ranking:")
    for rank, idx in enumerate(gnn_ranked):
        print(f"Rank {rank+1}: Kalimat {idx+1} (Score: {gnn_scores[idx]:.3f})")
    
    print("\n" + "=" * 100)
    print("💡 INTERPRETASI HASIL")
    print("=" * 100)
    
    print("\n🎯 Mengapa Kalimat Tertentu Mendapat Score Tinggi:")
    
    # Analisis untuk kalimat dengan score tertinggi
    top_ranked = gnn_ranked[0]
    print(f"\nKalimat {top_ranked+1} mendapat score tertinggi karena:")
    
    # Hitung koneksi
    connections = sum(1 for j in range(5) if j != top_ranked and adj_matrix[top_ranked][j] > 0.3)
    avg_similarity = sum(adj_matrix[top_ranked][j] for j in range(5) if j != top_ranked) / 4
    
    print(f"• Memiliki {connections} koneksi kuat (>0.3)")
    print(f"• Rata-rata similarity: {avg_similarity:.3f}")
    print(f"• Features: {node_features[top_ranked]}")
    print(f"• Terhubung dengan kalimat yang memiliki informasi penting")
    
    print("\n🔗 Graph Structure Impact:")
    print("• Kalimat dengan banyak koneksi = lebih banyak informasi")
    print("• Kalimat dengan similarity tinggi = pengaruh lebih besar")
    print("• Kalimat terisolasi = score rendah")
    
    print("\n📈 TextRank vs GNN:")
    print("• TextRank: Hanya menghitung jumlah koneksi")
    print("• GNN: Menghitung kualitas koneksi + content features")
    print("• GNN: Lebih sophisticated dalam information aggregation")

def create_simple_example():
    """
    Contoh yang lebih sederhana untuk pemahaman dasar
    """
    
    print("\n" + "=" * 100)
    print("🎯 SIMPLE EXAMPLE: 3 KALIMAT")
    print("=" * 100)
    
    # Contoh dengan 3 kalimat saja
    simple_sentences = [
        "Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.",
        "Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.",
        "Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon."
    ]
    
    # Adjacency matrix 3x3
    simple_adj = [
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.5],
        [0.2, 0.5, 1.0]
    ]
    
    # Node features 3x3
    simple_features = [
        [1.0, 1.0, 1.0],  # [Liputan6, Ambon, konflik]
        [0.0, 1.0, 1.0],  # [Ambon, konflik]
        [0.0, 1.0, 0.0]   # [Ambon]
    ]
    
    print("\n📋 DATA:")
    print("Kalimat 1: Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.")
    print("Kalimat 2: Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.")
    print("Kalimat 3: Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon.")
    
    print("\n🔗 ADJACENCY MATRIX:")
    print("     K1    K2    K3")
    print("     ---------------")
    for i, row in enumerate(simple_adj):
        print(f"K{i+1} |", end=" ")
        for val in row:
            print(f" {val:.1f} ", end="")
        print()
    
    print("\n📊 NODE FEATURES:")
    print("     F1    F2    F3")
    print("     ---------------")
    for i, features in enumerate(simple_features):
        print(f"K{i+1} |", end=" ")
        for val in features:
            print(f" {val:.1f} ", end="")
        print()
    
    print("\n🧮 MANUAL CALCULATION:")
    
    # Message passing
    print("\n1. Message Passing:")
    for i in range(3):
        print(f"Node {i+1} receives:")
        for j in range(3):
            if i != j:
                message = [simple_adj[i][j] * val for val in simple_features[j]]
                print(f"  From Node {j+1}: {simple_adj[i][j]:.1f} × {simple_features[j]} = {message}")
    
    # Aggregation
    print("\n2. Aggregation:")
    for i in range(3):
        aggregated = [0.0] * 3
        for j in range(3):
            if i != j:
                for k in range(3):
                    aggregated[k] += simple_adj[i][j] * simple_features[j][k]
        print(f"Node {i+1}: {aggregated}")
    
    # Update
    print("\n3. Update:")
    for i in range(3):
        updated = []
        for j in range(3):
            new_val = simple_features[i][j] + sum(simple_adj[i][k] * simple_features[k][j] for k in range(3) if k != i)
            updated.append(new_val)
        print(f"Node {i+1}: {simple_features[i]} + aggregated = {updated}")
    
    # Final scores
    print("\n4. Final Scores:")
    for i in range(3):
        score = sum(simple_features[i][j] + sum(simple_adj[i][k] * simple_features[k][j] for k in range(3) if k != i) for j in range(3))
        print(f"Node {i+1}: {score:.3f}")

if __name__ == "__main__":
    print("SIMPLE GNN CALCULATION EXAMPLE")
    print("=" * 100)
    
    # Main calculation
    simple_gnn_calculation_example()
    
    # Simple example
    create_simple_example()
    
    print("\n" + "=" * 100)
    print("✅ SIMPLE GNN CALCULATION EXAMPLE SELESAI!")
    print("=" * 100) 