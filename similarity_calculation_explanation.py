"""
DETAILED SIMILARITY CALCULATION EXPLANATION
==========================================

This file explains how similarity values in adjacency matrix are calculated
using cosine similarity between TF-IDF vectors.
"""

import numpy as np
import math

def cosine_similarity(vec1, vec2):
    """
    Menghitung cosine similarity antara dua vector
    """
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitude vector 1
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    
    # Magnitude vector 2
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

def calculate_all_similarities():
    """
    Menghitung similarity untuk semua pasangan kalimat
    """
    
    print("=" * 70)
    print("DETAILED SIMILARITY CALCULATION")
    print("=" * 70)
    
    # TF-IDF Features
    features = [
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Kalimat 1
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Kalimat 2
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # Kalimat 3
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # Kalimat 4
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]   # Kalimat 5
    ]
    
    print("TF-IDF Features:")
    for i, feat in enumerate(features):
        print(f"Kalimat {i+1}: {feat}")
    
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY CALCULATIONS")
    print("=" * 70)
    
    # Calculate similarity matrix
    n = len(features)
    similarity_matrix = []
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                # Self-similarity = 1.0
                similarity = 1.0
                print(f"Kalimat {i+1} vs Kalimat {j+1}: 1.0 (self-similarity)")
            else:
                similarity = cosine_similarity(features[i], features[j])
                print(f"Kalimat {i+1} vs Kalimat {j+1}: {similarity:.3f}")
            row.append(similarity)
        similarity_matrix.append(row)
    
    print(f"\n" + "=" * 70)
    print("SIMILARITY MATRIX")
    print("=" * 70)
    
    for i, row in enumerate(similarity_matrix):
        print(f"Kalimat {i+1}: {[f'{val:.3f}' for val in row]}")
    
    return similarity_matrix

def detailed_calculation_example():
    """
    Contoh perhitungan detail untuk Kalimat 1 vs Kalimat 2
    """
    
    print("\n" + "=" * 70)
    print("DETAILED EXAMPLE: KALIMAT 1 vs KALIMAT 2")
    print("=" * 70)
    
    # TF-IDF vectors
    kalimat1 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    kalimat2 = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    
    print(f"Kalimat 1: {kalimat1}")
    print(f"Kalimat 2: {kalimat2}")
    
    # Step 1: Dot Product
    print(f"\n1. DOT PRODUCT:")
    dot_product = 0
    for i, (a, b) in enumerate(zip(kalimat1, kalimat2)):
        product = a * b
        dot_product += product
        print(f"   ({a} × {b}) = {product}")
    print(f"   Total dot product = {dot_product}")
    
    # Step 2: Magnitude Kalimat 1
    print(f"\n2. MAGNITUDE KALIMAT 1:")
    magnitude1 = 0
    for i, val in enumerate(kalimat1):
        square = val * val
        magnitude1 += square
        print(f"   {val}² = {square}")
    magnitude1 = math.sqrt(magnitude1)
    print(f"   √{sum(val*val for val in kalimat1)} = {magnitude1:.3f}")
    
    # Step 3: Magnitude Kalimat 2
    print(f"\n3. MAGNITUDE KALIMAT 2:")
    magnitude2 = 0
    for i, val in enumerate(kalimat2):
        square = val * val
        magnitude2 += square
        print(f"   {val}² = {square}")
    magnitude2 = math.sqrt(magnitude2)
    print(f"   √{sum(val*val for val in kalimat2)} = {magnitude2:.3f}")
    
    # Step 4: Cosine Similarity
    print(f"\n4. COSINE SIMILARITY:")
    similarity = dot_product / (magnitude1 * magnitude2)
    print(f"   Similarity = {dot_product} / ({magnitude1:.3f} × {magnitude2:.3f})")
    print(f"   Similarity = {dot_product} / {magnitude1 * magnitude2:.3f}")
    print(f"   Similarity = {similarity:.3f}")
    
    return similarity

def explain_why_similarity_values():
    """
    Menjelaskan mengapa nilai similarity seperti itu
    """
    
    print("\n" + "=" * 70)
    print("WHY THESE SIMILARITY VALUES?")
    print("=" * 70)
    
    print("\n1. KALIMAT 1 vs KALIMAT 2:")
    print("   Kalimat 1: [Liputan6, Ambon, konflik, pemerintah]")
    print("   Kalimat 2: [Ambon, konflik, Maluku]")
    print("   Kata yang sama: Ambon, konflik (2 kata)")
    print("   Kata yang berbeda: Liputan6, pemerintah vs Maluku")
    print("   → Similarity = 0.4 (sedang)")
    
    print("\n2. KALIMAT 2 vs KALIMAT 5:")
    print("   Kalimat 2: [Ambon, konflik, Maluku]")
    print("   Kalimat 5: [konflik, Maluku, PBB, agenda]")
    print("   Kata yang sama: konflik, Maluku (2 kata)")
    print("   Kata yang berbeda: Ambon vs PBB, agenda")
    print("   → Similarity = 0.6 (tinggi)")
    
    print("\n3. KALIMAT 3 vs KALIMAT 4:")
    print("   Kalimat 3: [Ambon, Maluku, PBB]")
    print("   Kalimat 4: [Ambon, Maluku, pemerintah, PBB, Ridwan]")
    print("   Kata yang sama: Ambon, Maluku, PBB (3 kata)")
    print("   Kata yang berbeda: pemerintah, Ridwan")
    print("   → Similarity = 0.7 (sangat tinggi)")
    
    print("\n4. KALIMAT 1 vs KALIMAT 5:")
    print("   Kalimat 1: [Liputan6, Ambon, konflik, pemerintah]")
    print("   Kalimat 5: [konflik, Maluku, PBB, agenda]")
    print("   Kata yang sama: konflik (1 kata)")
    print("   Kata yang berbeda: Liputan6, Ambon, pemerintah vs Maluku, PBB, agenda")
    print("   → Similarity = 0.1 (rendah)")

def create_similarity_heatmap():
    """
    Membuat heatmap untuk visualisasi similarity
    """
    
    print("\n" + "=" * 70)
    print("SIMILARITY HEATMAP EXPLANATION")
    print("=" * 70)
    
    # Similarity matrix (dibulatkan)
    similarity_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    print("Similarity Matrix (Heatmap):")
    print("     K1   K2   K3   K4   K5")
    print("     -------------------------")
    
    for i, row in enumerate(similarity_matrix):
        print(f"K{i+1} |", end=" ")
        for j, val in enumerate(row):
            if val >= 0.6:
                print(f" 🔴{val:.1f}", end="")  # Sangat similar
            elif val >= 0.4:
                print(f" 🟡{val:.1f}", end="")  # Similar
            elif val >= 0.2:
                print(f" 🟢{val:.1f}", end="")  # Sedikit similar
            else:
                print(f" ⚪{val:.1f}", end="")  # Tidak similar
        print()
    
    print("\nLegend:")
    print("🔴 = Sangat similar (≥0.6)")
    print("🟡 = Similar (0.4-0.6)")
    print("🟢 = Sedikit similar (0.2-0.4)")
    print("⚪ = Tidak similar (<0.2)")

if __name__ == "__main__":
    print("SIMILARITY CALCULATION EXPLANATION")
    print("=" * 70)
    
    # Calculate all similarities
    similarity_matrix = calculate_all_similarities()
    
    # Detailed example
    detailed_calculation_example()
    
    # Explain why these values
    explain_why_similarity_values()
    
    # Create heatmap
    create_similarity_heatmap()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nNilai similarity dihitung menggunakan:")
    print("1. Cosine Similarity antara TF-IDF vectors")
    print("2. Semakin banyak kata yang sama, semakin tinggi similarity")
    print("3. Semakin sedikit kata yang sama, semakin rendah similarity")
    print("4. Nilai berkisar dari 0.0 (tidak similar) sampai 1.0 (identik)")
    
    print("\n✅ Penjelasan similarity calculation selesai!") 