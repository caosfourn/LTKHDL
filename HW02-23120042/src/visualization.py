import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cấu hình chung cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

def plot_numerical_distribution(data_array, col_name):
    """
    Vẽ Histogram và Boxplot cho biến số.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histogram + KDE
    sns.histplot(data_array, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Histogram: {col_name}')
    
    # Boxplot
    sns.boxplot(x=data_array, ax=axes[1], color='salmon')
    axes[1].set_title(f'Boxplot: {col_name}')
    
    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(unique_vals, counts, col_name, top_n=10):
    """
    Vẽ Barplot cho biến phân loại.
    """
    plt.figure(figsize=(10, 5))
    # Chỉ lấy top_n giá trị
    sns.barplot(x=counts[:top_n], y=unique_vals[:top_n], color='teal')
    plt.title(f'Distribution of {col_name} (Top {top_n})')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.show()

def plot_missing_data(headers, missing_pcts, is_missing_matrix):
    """
    Vẽ biểu đồ tỉ lệ thiếu và heatmap vị trí thiếu.
    """
    # Lọc ra các cột có dữ liệu thiếu để vẽ cho gọn
    cols_with_missing = []
    pcts_with_missing = []
    
    for i, pct in enumerate(missing_pcts):
        if pct > 0:
            cols_with_missing.append(headers[i])
            pcts_with_missing.append(pct)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Biểu đồ 1: Bar chart tỉ lệ thiếu
    if len(cols_with_missing) > 0:
        sns.barplot(x=pcts_with_missing, y=cols_with_missing, ax=axes[0], color='salmon')
        axes[0].set_title('Tỉ lệ dữ liệu thiếu theo cột (%)')
        axes[0].set_xlabel('Phần trăm (%)')
    else:
        axes[0].text(0.5, 0.5, 'Không có dữ liệu thiếu', ha='center')

    # Biểu đồ 2: Heatmap
    sns.heatmap(is_missing_matrix, cbar=False, yticklabels=False, cmap='binary', ax=axes[1])
    axes[1].set_title('Bản đồ vị trí dữ liệu thiếu (Trắng: Thiếu)')
    axes[1].set_xlabel('Cột')
    axes[1].set_ylabel('Dòng')
    
    # Gán nhãn trục X cho heatmap (cẩn thận nếu quá nhiều cột)
    if len(headers) <= 20:
        axes[1].set_xticks(np.arange(len(headers)) + 0.5)
        axes[1].set_xticklabels(headers, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(corr_matrix, labels):
    """
    Vẽ Heatmap ma trận tương quan.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_group_means(groups, means, group_col_name, target_col_name='target'):
    """
    Vẽ Barplot giá trị trung bình của target theo nhóm.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(x=means, y=groups, color='skyblue')
    plt.title(f'Average {target_col_name} by {group_col_name}')
    plt.xlabel(f'Mean {target_col_name}')
    plt.tight_layout()
    plt.show()

def plot_gender_pie_charts(sizes_male, sizes_female, total_male, total_female):
    """
    Vẽ 2 biểu đồ tròn so sánh Nam/Nữ (Câu hỏi 1).
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    colors = ["#ef8a62", "#67a9cf"] 
    labels_legend = ['Đổi việc', 'Không đổi']

    # Nam
    wedges1, _, _ = ax[0].pie(sizes_male, autopct="%1.1f%%", startangle=90, colors=colors)
    ax[0].set_title(f"Tỉ lệ đổi việc của NAM\n(Tổng: {total_male})")

    # Nữ
    wedges2, _, _ = ax[1].pie(sizes_female, autopct="%1.1f%%", startangle=90, colors=colors)
    ax[1].set_title(f"Tỉ lệ đổi việc của NỮ\n(Tổng: {total_female})")

    fig.legend(wedges1, labels_legend, title="Trạng thái", loc="center right", bbox_to_anchor=(0.98, 0.5))
    plt.subplots_adjust(right=0.85)
    plt.show()

def plot_cdi_vs_training_hours(cdi_data, hours_data):
    """
    Vẽ biểu đồ đường + hồi quy giữa CDI và Training Hours (Câu hỏi 2).
    """
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Vẽ lineplot
    sns.lineplot(x=cdi_data, y=hours_data, color='teal', linewidth=2.5, errorbar='ci')
    
    # Vẽ regplot
    sns.regplot(x=cdi_data, y=hours_data, scatter=False, color='orange', 
                line_kws={'linestyle':'--'}, label='Xu hướng tuyến tính')

    plt.title('XU HƯỚNG ĐÀO TẠO THEO CHỈ SỐ PHÁT TRIỂN CỦA THÀNH PHỐ (CDI)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Chỉ số phát triển thành phố (CDI)', fontsize=12)
    plt.ylabel('Số giờ đào tạo trung bình', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_experience_vs_target(exp_labels, counts, percentages):
    """
    Vẽ biểu đồ kết hợp cột (số lượng) và đường (tỷ lệ) cho câu hỏi 3.
    """
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="white")

    # Trục 1: Barplot (Số lượng)
    ax1 = sns.barplot(x=exp_labels, y=counts, color="#67a9cf", alpha=0.6, label='Số lượng mẫu')
    ax1.set_ylabel('Số lượng ứng viên', fontsize=14, color="gray")
    ax1.tick_params(axis='y', labelcolor="gray")
    ax1.set_xlabel('Kinh nghiệm (năm)', fontsize=14)

    # Trục 2: Lineplot (Tỷ lệ %)
    ax2 = ax1.twinx()
    sns.lineplot(x=exp_labels, y=percentages, ax=ax2, color="#ef8a62", linewidth=3, 
                 marker='o', markersize=8, label='Tỷ lệ muốn thay đổi (%)')
    
    ax2.set_ylabel('Tỷ lệ muốn thay đổi (%)', fontsize=14, color="#ef8a62")
    ax2.tick_params(axis='y', labelcolor="#ef8a62")
    ax2.set_ylim(0, 100)

    plt.title('Tương quan giữa Kinh nghiệm và Tỷ lệ muốn thay đổi công việc', fontsize=18)
    
    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

    plt.show()