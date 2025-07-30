"""
CREATE PDF REPORT
================

Script untuk membuat laporan PDF dari adjacency matrix table dan SimpleGNN calculations
"""

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

def create_pdf_report():
    """
    Membuat laporan PDF dengan adjacency matrix dan SimpleGNN calculations
    """
    
    # Data untuk laporan
    sentences = [
        "Liputan6, Ambon: Partai Bulan Bintang wilayah Maluku bertekad membantu pemerintah menyelesaikan konflik di provinsi tersebut.",
        "Syaratnya, penanganan penyelesaian konflik Maluku harus dimulai dari awal kerusuhan, yakni 19 Januari 1999.",
        "Demikian hasil Musyawarah Wilayah I PBB Maluku yang dimulai Sabtu pekan silam dan berakhir Senin (31/12) di Ambon.",
        "Menurut seorang fungsionaris PBB Ridwan Hasan, persoalan di Maluku bisa selesai asalkan pemerintah dan aparat keamanan serius menangani setiap persoalan di Maluku secara komprehensif dan bijaksana.",
        "Itulah sebabnya, PBB wilayah Maluku akan menjadikan penyelesaian konflik sebagai agenda utama partai."
    ]
    
    keywords = [
        ["Liputan6", "Ambon", "konflik", "pemerintah"],
        ["Ambon", "konflik", "Maluku"],
        ["Ambon", "Maluku", "PBB"],
        ["Ambon", "Maluku", "pemerintah", "PBB", "Ridwan"],
        ["konflik", "Maluku", "PBB", "agenda"]
    ]
    
    adj_matrix = [
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.5, 0.4, 0.6],
        [0.2, 0.5, 1.0, 0.7, 0.3],
        [0.3, 0.4, 0.7, 1.0, 0.5],
        [0.1, 0.6, 0.3, 0.5, 1.0]
    ]
    
    # Buat PDF
    doc = SimpleDocTemplate("adjacency_matrix_gnn_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=20,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    )
    
    # Cover page
    story.append(Paragraph("LAPORAN ADJACENCY MATRIX DAN SIMPLE GNN CALCULATION", title_style))
    story.append(Spacer(1, 50))
    story.append(Paragraph("Text Summarization dengan Graph Neural Network", subtitle_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Dibuat untuk analisis similarity antar kalimat dan perhitungan GNN", normal_style))
    story.append(PageBreak())
    
    # Halaman 1: Kalimat-kalimat
    story.append(Paragraph("1. DATA KALIMAT", subtitle_style))
    story.append(Spacer(1, 20))
    
    for i, sentence in enumerate(sentences):
        story.append(Paragraph(f"<b>Kalimat {i+1}:</b>", styles['Heading3']))
        story.append(Paragraph(sentence, normal_style))
        story.append(Paragraph(f"<b>Kata Kunci:</b> {', '.join(keywords[i])}", normal_style))
        story.append(Spacer(1, 15))
    
    story.append(PageBreak())
    
    # Halaman 2: Adjacency Matrix
    story.append(Paragraph("2. ADJACENCY MATRIX", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Header untuk adjacency matrix
    header = ['', 'K1', 'K2', 'K3', 'K4', 'K5']
    matrix_data = [header]
    
    for i, row in enumerate(adj_matrix):
        matrix_row = [f'K{i+1}']
        for val in row:
            matrix_row.append(f'{val:.1f}')
        matrix_data.append(matrix_row)
    
    # Buat tabel adjacency matrix
    adj_table = Table(matrix_data, colWidths=[0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    adj_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    
    story.append(adj_table)
    story.append(Spacer(1, 20))
    
    # Penjelasan adjacency matrix
    story.append(Paragraph("3. TABEL DETAIL SIMILARITY", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Buat tabel detail similarity
    similarity_data = [
        ['Pasangan', 'Similarity', 'Kata Sama', 'Kata Berbeda', 'Keterangan']
    ]
    
    for i in range(5):
        for j in range(5):
            if i == j:
                similarity_data.append([
                    f'K{i+1} ↔ K{j+1}',
                    '1.000',
                    'Identik',
                    '-',
                    'Kalimat sama'
                ])
            else:
                similarity = adj_matrix[i][j]
                common_words = set(keywords[i]) & set(keywords[j])
                diff_words_i = set(keywords[i]) - set(keywords[j])
                diff_words_j = set(keywords[j]) - set(keywords[i])
                
                common_str = ', '.join(common_words) if common_words else 'Tidak ada'
                diff_str = f"K{i+1}:{','.join(diff_words_i)} | K{j+1}:{','.join(diff_words_j)}"
                
                if similarity >= 0.6:
                    keterangan = "Sangat similar"
                elif similarity >= 0.4:
                    keterangan = "Similar"
                elif similarity >= 0.2:
                    keterangan = "Sedikit similar"
                else:
                    keterangan = "Tidak similar"
                
                similarity_data.append([
                    f'K{i+1} ↔ K{j+1}',
                    f'{similarity:.3f}',
                    common_str,
                    diff_str,
                    keterangan
                ])
    
    # Buat tabel similarity (gunakan landscape untuk tabel besar)
    similarity_table = Table(similarity_data, colWidths=[1.2*inch, 0.8*inch, 1.5*inch, 2.5*inch, 1*inch])
    similarity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(similarity_table)
    story.append(PageBreak())
    
    # Halaman 3: Interpretasi
    story.append(Paragraph("4. INTERPRETASI SIMILARITY", subtitle_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Range Similarity:</b>", styles['Heading3']))
    story.append(Paragraph("• 0.8 - 1.0: Sangat sangat similar (hampir identik)", normal_style))
    story.append(Paragraph("• 0.6 - 0.8: Sangat similar (banyak kata yang sama)", normal_style))
    story.append(Paragraph("• 0.4 - 0.6: Similar (sedang, beberapa kata sama)", normal_style))
    story.append(Paragraph("• 0.2 - 0.4: Sedikit similar (sedikit kata yang sama)", normal_style))
    story.append(Paragraph("• 0.0 - 0.2: Tidak similar (hampir tidak ada kata yang sama)", normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Koneksi Terkuat:</b>", styles['Heading3']))
    story.append(Paragraph("• K3 ↔ K4: 0.7 (Sangat similar - 3 kata sama: Ambon, Maluku, PBB)", normal_style))
    story.append(Paragraph("• K2 ↔ K5: 0.6 (Sangat similar - 2 kata sama: konflik, Maluku)", normal_style))
    story.append(Paragraph("• K4 ↔ K5: 0.5 (Similar - 2 kata sama: Maluku, PBB)", normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Koneksi Terlemah:</b>", styles['Heading3']))
    story.append(Paragraph("• K1 ↔ K5: 0.1 (Tidak similar - hanya 1 kata sama: konflik)", normal_style))
    story.append(Paragraph("• K1 ↔ K3: 0.2 (Sedikit similar - 1 kata sama: Ambon)", normal_style))
    story.append(Paragraph("• K1 ↔ K4: 0.3 (Sedikit similar - 1 kata sama: Ambon)", normal_style))
    
    story.append(PageBreak())
    
    # Halaman 4: SimpleGNN Calculation
    story.append(Paragraph("5. SIMPLE GNN CALCULATION", subtitle_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Langkah-langkah Kalkulasi:</b>", styles['Heading3']))
    story.append(Paragraph("1. <b>Message Passing:</b> Setiap kalimat mengirim pesan ke tetangganya", normal_style))
    story.append(Paragraph("   Pesan = adjacency_weight × neighbor_features", normal_style))
    story.append(Paragraph("2. <b>Aggregation:</b> Menggabungkan semua pesan yang diterima (SUM)", normal_style))
    story.append(Paragraph("3. <b>Update:</b> New_Features = Original_Features + Aggregated_Messages", normal_style))
    story.append(Paragraph("4. <b>Scoring:</b> Score = Sum dari semua features yang diupdate", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Node Features Table
    story.append(Paragraph("<b>Node Features (TF-IDF Vectors):</b>", styles['Heading3']))
    
    node_features = [
        [1.0, 1.0, 1.0, 1.0],  # Kalimat 1: [Liputan6, Ambon, konflik, pemerintah]
        [0.0, 1.0, 1.0, 0.0],  # Kalimat 2: [Ambon, konflik]
        [0.0, 1.0, 0.0, 0.0],  # Kalimat 3: [Ambon]
        [0.0, 1.0, 0.0, 1.0],  # Kalimat 4: [Ambon, pemerintah]
        [0.0, 0.0, 1.0, 0.0]   # Kalimat 5: [konflik]
    ]
    
    features_header = ['Kalimat', 'F1', 'F2', 'F3', 'F4', 'Keterangan']
    features_data = [features_header]
    
    feature_names = ['Liputan6', 'Ambon', 'konflik', 'pemerintah']
    for i, features in enumerate(node_features):
        features_data.append([
            f'K{i+1}',
            f'{features[0]:.1f}',
            f'{features[1]:.1f}',
            f'{features[2]:.1f}',
            f'{features[3]:.1f}',
            f'[{", ".join([feature_names[j] for j in range(4) if features[j] > 0])}]'
        ])
    
    features_table = Table(features_data, colWidths=[0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 2*inch])
    features_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(features_table)
    story.append(Spacer(1, 20))
    
    # Contoh perhitungan
    story.append(Paragraph("<b>Contoh Perhitungan Manual:</b>", styles['Heading3']))
    story.append(Paragraph("Node 1 menerima pesan dari:", normal_style))
    story.append(Paragraph("• Node 2: 0.4 × [0.0, 1.0, 1.0, 0.0] = [0.0, 0.4, 0.4, 0.0]", normal_style))
    story.append(Paragraph("• Node 3: 0.2 × [0.0, 1.0, 0.0, 0.0] = [0.0, 0.2, 0.0, 0.0]", normal_style))
    story.append(Paragraph("• Node 4: 0.3 × [0.0, 1.0, 0.0, 1.0] = [0.0, 0.3, 0.0, 0.3]", normal_style))
    story.append(Paragraph("• Node 5: 0.1 × [0.0, 0.0, 1.0, 0.0] = [0.0, 0.0, 0.1, 0.0]", normal_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Aggregated: [0.0, 0.9, 0.5, 0.3]", normal_style))
    story.append(Paragraph("Updated: [1.0, 1.0, 1.0, 1.0] + [0.0, 0.9, 0.5, 0.3] = [1.0, 1.9, 1.5, 1.3]", normal_style))
    story.append(Paragraph("Final Score: 1.0 + 1.9 + 1.5 + 1.3 = 5.7", normal_style))
    
    story.append(PageBreak())
    
    # Halaman 5: Hasil Ranking
    story.append(Paragraph("6. HASIL RANKING DAN KESIMPULAN", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Ranking table
    ranking_data = [
        ['Rank', 'Kalimat', 'Score', 'Alasan Score Tinggi']
    ]
    
    # Simulasi scores (dari perhitungan sebelumnya)
    scores = [5.7, 5.5, 4.5, 5.2, 3.9]
    ranked_indices = sorted(range(5), key=lambda i: scores[i], reverse=True)
    
    for rank, idx in enumerate(ranked_indices):
        if rank == 0:
            alasan = "Banyak koneksi + features lengkap"
        elif rank == 1:
            alasan = "Koneksi kuat dengan kalimat lain"
        elif rank == 2:
            alasan = "Koneksi sedang, features cukup"
        elif rank == 3:
            alasan = "Sedikit koneksi, features terbatas"
        else:
            alasan = "Koneksi lemah, features minimal"
        
        ranking_data.append([
            f'Rank {rank+1}',
            f'Kalimat {idx+1}',
            f'{scores[idx]:.1f}',
            alasan
        ])
    
    ranking_table = Table(ranking_data, colWidths=[1*inch, 1.5*inch, 1*inch, 3*inch])
    ranking_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(ranking_table)
    story.append(Spacer(1, 20))
    
    # Kesimpulan
    story.append(Paragraph("<b>Kesimpulan:</b>", styles['Heading3']))
    story.append(Paragraph("• SimpleGNN menggunakan kalkulasi sederhana namun efektif", normal_style))
    story.append(Paragraph("• Kalimat dengan banyak koneksi dan similarity tinggi mendapat score tinggi", normal_style))
    story.append(Paragraph("• Graph structure mempengaruhi ranking kalimat secara signifikan", normal_style))
    story.append(Paragraph("• Metode ini transparan dan mudah diinterpretasi", normal_style))
    
    # Build PDF
    doc.build(story)
    print("✅ PDF report berhasil dibuat: adjacency_matrix_gnn_report.pdf")

if __name__ == "__main__":
    create_pdf_report() 