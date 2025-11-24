import numpy as np

def load_data(file_path):
    """Đọc dữ liệu và headers từ file CSV."""
    # Đọc toàn bộ dữ liệu dưới dạng chuỗi
    data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    # Đọc header
    headers = np.genfromtxt(file_path, delimiter=',', dtype=str, max_rows=1)
    # Xóa các dấu ngoặc kép hoặc khoảng trắng thừa trong header nếu có
    headers = np.char.strip(headers, '"')
    headers = np.char.strip(headers, "'")
    return data, headers

def identify_columns(headers):
    """Phân loại index các cột (Vectorized)."""
    
    # 1. Tìm index của các cột đặc biệt
    target_idx = np.where(headers == 'target')[0][0]
    exp_idx = np.where(headers == 'experience')[0][0]
    job_idx = np.where(headers == 'last_new_job')[0][0]
    
    # 2. Tìm index nhóm Numerical 
    num_names = ['city_development_index', 'training_hours']
    # Sử dụng np.isin để tìm vị trí các cột numerical
    num_mask = np.isin(headers, num_names)
    numerical_cols = np.where(num_mask)[0].tolist()
    
    # 3. Tìm nhóm Categorical (Logic loại trừ)
    # Tạo mask cho các cột cần loại bỏ: enrollee_id, target, experience, last_new_job, và các cột numerical
    id_mask = (headers == 'enrollee_id')
    special_mask = (headers == 'experience') | (headers == 'last_new_job') | (headers == 'target')
    
    # Tổng hợp mask các cột "Không phải categorical"
    exclude_mask = id_mask | special_mask | num_mask
    
    # Đảo ngược mask để lấy Categorical
    cat_mask = ~exclude_mask
    
    # Lấy index
    categorical_cols = np.where(cat_mask)[0].tolist()
    
    return numerical_cols, categorical_cols, target_idx, exp_idx, job_idx

def process_numerical_data(data, numerical_cols, exp_idx, job_idx):
    """Chuyển đổi dữ liệu chuỗi sang số và xử lý missing values (NO LOOPS)."""
    
    # 1. Tách dữ liệu cần xử lý
    cols_to_process = numerical_cols + [exp_idx, job_idx]
    raw_sub_matrix = data[:, cols_to_process] # Fancy Indexing
    
    # 2. Xử lý các ký tự đặc biệt (Vectorized String Replace)
    # np.char.replace tạo bản sao mới
    cleaned_matrix = np.char.replace(raw_sub_matrix, '>20', '21')
    cleaned_matrix = np.char.replace(cleaned_matrix, '<1', '0')
    cleaned_matrix = np.char.replace(cleaned_matrix, '>4', '5')
    cleaned_matrix = np.char.replace(cleaned_matrix, 'never', '0')
    
    # Xử lý các biến thể của NaN
    cleaned_matrix[cleaned_matrix == ''] = 'nan'
    cleaned_matrix[cleaned_matrix == 'NA'] = 'nan'
    cleaned_matrix[cleaned_matrix == 'NULL'] = 'nan'
    cleaned_matrix[cleaned_matrix == 'None'] = 'nan'

    # 3. Chuyển đổi sang float đồng loạt
    numerical_data = cleaned_matrix.astype(float)

    # 4. Impute Missing Values bằng Median (Vectorized)
    # Tính median theo cột, bỏ qua nan
    col_medians = np.nanmedian(numerical_data, axis=0)
    
    # Tìm vị trí nan trong ma trận (trả về tuple index dòng, index cột)
    inds = np.where(np.isnan(numerical_data))
    
    # Gán giá trị median tương ứng vào vị trí nan
    # inds[1] là mảng các chỉ số cột tương ứng với từng vị trí nan
    if len(inds[0]) > 0:
        numerical_data[inds] = col_medians[inds[1]]
    
    return numerical_data, cols_to_process

def process_categorical_data(data, categorical_cols):
    """Xử lý missing values cho categorical."""
    categorical_cleaned = data.copy()
    # Loop qua các cột (số lượng ít) là chấp nhận được
    for _, col_idx in enumerate(categorical_cols):
        col = categorical_cleaned[:, col_idx]
        missing_mask = (col == '') | (col == 'NA') | (col == 'NULL') | (col == 'None')
        categorical_cleaned[missing_mask, col_idx] = "Unknown"
    return categorical_cleaned

def handle_outliers(numerical_data, headers, col_indices):
    """Phát hiện và Winsorize outliers (Vectorized)."""
    numerical_winsorized = numerical_data.copy()
    
    print("--- Phát hiện Outliers ---")
    
    # 1. Tính toán thống kê theo cột (Broadcasting)
    Q1 = np.percentile(numerical_data, 25, axis=0)
    Q3 = np.percentile(numerical_data, 75, axis=0)
    IQR = Q3 - Q1
    
    # 2. Tính cận trên/dưới 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 3. Tạo mask cho outliers 
    outliers_mask = (numerical_data < lower_bound) | (numerical_data > upper_bound)
    
    # In thông tin
    outliers_count = np.sum(outliers_mask, axis=0)
    for j, count in enumerate(outliers_count):
        if count > 0:
            print(f"Column {headers[col_indices[j]]}: {count} outliers")
            
    # 4. Winsorize (Vectorized Logic)
    w_lower = np.percentile(numerical_data, 5, axis=0)
    w_upper = np.percentile(numerical_data, 95, axis=0)
    
    numerical_winsorized = np.where(numerical_data < lower_bound, w_lower, numerical_winsorized)
    numerical_winsorized = np.where(numerical_data > upper_bound, w_upper, numerical_winsorized)

    return numerical_winsorized

def normalize_data(data):
    """Min-Max Normalization (Vectorized)."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    ranges = max_vals - min_vals
    # Tránh chia cho 0 nếu max == min
    ranges[ranges == 0] = 1 
    
    min_max_normalized = (data - min_vals) / ranges
    
    return min_max_normalized

def encode_features(numerical_data, categorical_data, headers, numerical_cols_idx, categorical_cols_idx):
    """Thực hiện Ordinal Encoding và One-Hot Encoding (Optimized)."""
    
    # 1. Helper function thay thế cho np.vectorize
    def fast_ordinal_encode(data_col, mapping_dict):
        # Chuyển keys và vals sang array
        keys = np.array(list(mapping_dict.keys()))
        vals = np.array(list(mapping_dict.values()))
        
        # Sắp xếp keys để dùng searchsorted
        sorter = np.argsort(keys)
        keys = keys[sorter]
        vals = vals[sorter]
        
        # Tìm vị trí chèn
        idx = np.searchsorted(keys, data_col)
        
        # Xử lý các phần tử không tìm thấy (tránh index out of bounds)
        idx[idx == len(keys)] = 0 
        
        # Kiểm tra lại xem giá trị tại idx có khớp với data_col không
        mask = keys[idx] == data_col 
        
        # Trả về giá trị mapping, nếu không khớp (unknown key) trả về 0
        return np.where(mask, vals[idx], 0)

    # Định nghĩa Mapping 
    edu_mapping = {'Unknown': 0, 'Primary School': 1, 'High School': 2, 'Graduate': 3, 'Masters': 4, 'Phd': 5}
    company_size_mapping = {'Unknown': 0, '<10': 1, '10/49': 2, '50-99': 3, '100-500': 4, '500-999': 5, '1000-4999': 6, '5000-9999': 7, '10000+': 8}

    # Tìm index và Encode
    edu_header_idx = np.where(headers == 'education_level')[0][0]
    company_header_idx = np.where(headers == 'company_size')[0][0]
    
    # Tìm index tương đối trong mảng categorical_cols_idx
    edu_rel_idx = categorical_cols_idx.index(edu_header_idx)
    comp_rel_idx = categorical_cols_idx.index(company_header_idx)

    # --- Áp dụng hàm fast_ordinal_encode ---
    edu_encoded = fast_ordinal_encode(categorical_data[:, edu_rel_idx], edu_mapping)
    comp_encoded = fast_ordinal_encode(categorical_data[:, comp_rel_idx], company_size_mapping)
    
    # Normalize Ordinal 
    def simple_norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else arr
        
    edu_encoded = simple_norm(edu_encoded)
    comp_encoded = simple_norm(comp_encoded)

    # 2. One-Hot Encoding
    encoded_cols = []
    encoded_header_names = []
    
    print("--- Đang thực hiện Encoding ---")
    
    indices_to_skip = [edu_rel_idx, comp_rel_idx]
    
    # Iterate qua danh sách index CỦA mảng categorical (từ 0 đến len-1)
    for i in range(categorical_data.shape[1]):
        if i in indices_to_skip: continue
            
        # Lấy tên cột gốc từ categorical_cols_idx
        original_col_idx = categorical_cols_idx[i]
        header_name = headers[original_col_idx]
        
        col_data = categorical_data[:, i]
        uniques = np.unique(col_data)
        
        # Kỹ thuật One-Hot nhanh: Broadcasting comparison
        # Tạo ma trận one-hot (n_samples, n_uniques)
        one_hot_matrix = (col_data[:, None] == uniques[None, :]).astype(float)
        
        encoded_cols.append(one_hot_matrix)
        
        for u in uniques:
             clean_u = str(u).replace(" ", "_").replace(",", "")
             encoded_header_names.append(f"{header_name}_{clean_u}")

    # 3. Gộp tất cả
    X_ordinal = np.hstack((edu_encoded.reshape(-1, 1), comp_encoded.reshape(-1, 1)))
    ordinal_headers = ['education_level_ord', 'company_size_ord']
    numeric_header_names = [headers[i] for i in numerical_cols_idx]

    if encoded_cols:
        X_onehot = np.hstack(encoded_cols) 
        X_final = np.hstack((numerical_data, X_ordinal, X_onehot))
        final_headers = numeric_header_names + ordinal_headers + encoded_header_names
    else:
        X_final = np.hstack((numerical_data, X_ordinal))
        final_headers = numeric_header_names + ordinal_headers

    return X_final, final_headers

def save_processed_data(X, y, headers, output_file="processed_data.csv"):
    """Lưu dữ liệu ra file CSV."""
    data_ready = np.hstack((X, y.reshape(-1, 1)))
    final_headers = headers + ['target']
    header_str = ",".join(final_headers)
    
    # Lưu với định dạng %.6f để gọn gàng
    np.savetxt(output_file, data_ready, delimiter=",", header=header_str, fmt='%.6f', comments='')
    print(f"Đã lưu file tại: {output_file}")
    print(f"Kích thước dữ liệu: {data_ready.shape}")