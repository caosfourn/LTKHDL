# Dự đoán khả năng có thay đổi công việc Khoa học dữ liệu

## Giới thiệu

### Mô tả bài toán
Một công ty hoạt động trong lĩnh vực Big Data và Data Science muốn tuyển dụng các nhà khoa học dữ liệu. Tuy nhiên, họ gặp vấn đề khi các ứng viên sau khi được đào tạo lại có xu hướng rời bỏ công ty để tìm việc mới.

Dự án này xây dựng một mô hình máy học để dự đoán xác suất một ứng viên sẽ **tìm kiếm việc làm mới (1)** hay **ở lại công ty (0)**, dựa trên thông tin nhân khẩu học, trình độ học vấn và kinh nghiệm làm việc của họ.

### Động lực và Ứng dụng thực tế
* **Giảm chi phí:** Giúp công ty tập trung nguồn lực đào tạo vào những ứng viên có cam kết lâu dài.
* **Hoạch định nhân sự:** Dự báo biến động nhân sự để có kế hoạch tuyển dụng kịp thời.
* **Ứng dụng kỹ thuật:** Dự án tập trung vào việc **triển khai các thuật toán Machine Learning từ đầu (from scratch) sử dụng thư viện NumPy** để hiểu sâu về toán học và tối ưu hóa vectorization.

### Mục tiêu cụ thể
* Xử lý và làm sạch dữ liệu thô.
* Xây dựng các mô hình: Logistic Regression, KNN, Decision Tree, Random Forest bằng **NumPy**.
* Đạt độ chính xác (Accuracy) trên 75% trên tập kiểm thử.

---

## Dataset

* **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists (Kaggle)](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
* **Kích thước:** 19,158 dòng và 14 cột.
* **Mô tả các features chính:**
    * `enrollee_id`: ID định danh ứng viên (được loại bỏ khi training).
    * `city_development_index` (CDI): Chỉ số phát triển của thành phố (Numerical).
    * `gender`: Giới tính.
    * `relevent_experience`: Kinh nghiệm liên quan.
    * `enrolled_university`: Loại hình đại học đã đăng ký.
    * `education_level`: Trình độ học vấn (Graduate, Masters, PhD...).
    * `major_discipline`: Chuyên ngành.
    * `experience`: Số năm kinh nghiệm.
    * `company_size`: Quy mô công ty hiện tại.
    * `company_type`: Loại hình công ty.
    * `last_new_job`: Số năm kể từ công việc trước đó.
    * `training_hours`: Số giờ đào tạo đã hoàn thành.
    * `target`: **0** (Không tìm việc) hoặc **1** (Tìm việc).

---

## Phương pháp

Dự án tập trung vào việc sử dụng **NumPy** để thao tác dữ liệu và xây dựng thuật toán.

### 1. Quy trình xử lý dữ liệu (`data_processing.py`)
* **Data Cleaning:**
    * Xử lý giá trị thiếu (Missing Values): Điền `Median` cho biến số và `Unknown` cho biến phân loại.
    * Xử lý chuỗi: Chuẩn hóa các giá trị như `>20` thành `21`, `<1` thành `0`.
* **Outlier Handling:** Sử dụng phương pháp IQR để phát hiện và Winsorize outliers (thay thế bằng giá trị biên phân vị thứ 5 và 95).
* **Scaling:** Sử dụng Min-Max Normalization để đưa dữ liệu về khoảng [0, 1].
* **Encoding:**
    * *Ordinal Encoding:* Áp dụng cho `education_level` và `company_size` (có thứ tự).
    * *One-Hot Encoding:* Áp dụng cho các biến phân loại còn lại bằng kỹ thuật Broadcasting của NumPy.

### 2. Thuật toán và Công thức Toán học (`models.py`)
Các thuật toán được code tay hoàn toàn bằng NumPy.

#### a. Logistic Regression
Sử dụng hàm Sigmoid và tối ưu hóa bằng Gradient Descent (SGD/Adam).
* **Dự đoán:** $\hat{y} = \sigma(Xw + b)$ với $\sigma(z) = \frac{1}{1 + e^{-z}}$
* **Hàm mất mát (Log Loss):**
    $$J(w, b) = -\frac{1}{m} \sum [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$
* **Cập nhật trọng số (NumPy):**
    Sử dụng `np.dot` hoặc `np.einsum` để tính gradient:
    $$dw = \frac{1}{m} X^T (\hat{y} - y)$$

#### b. K-Nearest Neighbors (KNN)
* **Khoảng cách Euclidean (Vectorized):** Thay vì dùng vòng lặp, sử dụng công thức khai triển hằng đẳng thức để tính ma trận khoảng cách nhanh chóng:
    $$|x - y|^2 = |x|^2 + |y|^2 - 2x \cdot y$$
    *Implement:* `dists_sq = X_sq + train_sq - 2 * np.dot(X, X_train.T)`

#### c. Decision Tree & Random Forest
* **Tiêu chí chia (Split Criteria):** Sử dụng Information Gain dựa trên Entropy.
    $$Entropy(S) = - \sum p_i \log_2 p_i$$
* **Dự đoán:** Duyệt cây và lấy giá trị xuất hiện nhiều nhất (Majority Voting) bằng `np.bincount`.

---

## Installation & Setup

Yêu cầu: Python 3.8+ và các thư viện trong `requirements.txt`.

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/username/project-name.git](https://github.com/username/project-name.git)
    cd project-name
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý: Dự án chủ yếu dùng `numpy`, `matplotlib`, `seaborn`.*

---

## Usage

Dự án được chia thành các bước chạy tuần tự trong thư mục `notebooks/`:

1.  **Khám phá dữ liệu (EDA):**
    Chạy file `notebooks/01_data_exploration.ipynb` để xem phân tích thống kê, biểu đồ phân phối và tương quan.
    
2.  **Tiền xử lý dữ liệu:**
    Chạy file `notebooks/02_preprocessing.ipynb`.
    * Input: `data/raw/aug_train.csv`
    * Output: `data/processed/processed_data.csv` (Dữ liệu sạch, đã mã hóa).

3.  **Huấn luyện và Đánh giá:**
    Chạy file `notebooks/03_modeling.ipynb` để huấn luyện các model và xem kết quả so sánh.

---

## Results

Kết quả đánh giá trên tập kiểm thử (Test Set):

| Model               | Accuracy   | Precision | Recall | F1-Score | Thời gian chạy |
|---------------------|------------|-----------|--------|----------|----------------|
| **Random Forest**   | **77.32%** | 0.60      | 0.36   | **0.45** | 0.78s          |
| Logistic Regression | 77.26%     | -         | -      | -        | 4.53s          |
| Decision Tree       | 77.08%     | -         | -      | -        | 2.32s          |
| KNN                 | 76.74%     |           | -      | -        | 1.28s          |

* **Nhận xét:** Random Forest cho kết quả tốt nhất về độ chính xác.
* **Vấn đề:** Recall và F1-Score còn thấp do dữ liệu bị mất cân bằng (Imbalanced Dataset - Nhóm không nghỉ việc chiếm đa số).

**Trực quan hóa:**
(Xem chi tiết các biểu đồ phân phối và ma trận nhầm lẫn trong `notebooks/01_data_exploration.ipynb` và `03_modeling.ipynb`).

---

## Project Structure

```text
project-name/
├── README.md                   # Tài liệu dự án
├── requirements.txt            # Thư viện cần thiết
├── data/
│   ├── raw/                    # Dữ liệu gốc (aug_train.csv)
│   └── processed/              # Dữ liệu sau khi xử lý
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Phân tích, trực quan hóa (EDA)
│   ├── 02_preprocessing.ipynb     # Làm sạch, chuẩn hóa, mã hóa
│   └── 03_modeling.ipynb          # Huấn luyện, đánh giá model
└── src/
    ├── __init__.py
    ├── data_processing.py      # Các hàm xử lý dữ liệu (Numpy)
    ├── visualization.py        # Các hàm vẽ biểu đồ
    └── models.py               # Code thuật toán ML từ đầu (Numpy)
```
## Challenges & Solutions

Trong quá trình xây dựng các thuật toán Machine Learning từ đầu (from scratch) với **NumPy**, em đã gặp và giải quyết các thách thức sau:

* **1. Tối ưu hóa tốc độ tính toán (Vectorization):**
    * *Vấn đề:* Việc sử dụng vòng lặp `for` trong Python để tính khoảng cách (trong KNN) hoặc tính Gradient (trong Logistic Regression) rất chậm trên tập dữ liệu lớn (~20,000 dòng).
    * *Giải pháp:* Thay thế hoàn toàn vòng lặp bằng kỹ thuật **Broadcasting** và **Vectorization** của NumPy.
        * Ví dụ: Trong KNN, thay vì tính từng điểm, nhóm áp dụng công thức $||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b$ để tính toàn bộ ma trận khoảng cách cùng lúc.
        * Trong Logistic Regression, sử dụng `np.einsum` (Einstein summation) để tính tích vô hướng nhanh hơn `np.dot` thông thường.

* **2. Xử lý sự khác biệt về chiều (Shape Mismatch):**
    * *Vấn đề:* Thường xuyên gặp lỗi `ValueError` khi thực hiện các phép toán giữa vector hàng, vector cột và ma trận trọng số.
    * *Giải pháp:* Kiểm soát chặt chẽ chiều dữ liệu bằng `np.reshape(-1, 1)` và `np.newaxis` để đảm bảo tính tương thích khi Broadcasting.

* **3. Dữ liệu mất cân bằng (Imbalanced Dataset):**
    * *Vấn đề:* Tỉ lệ người nghỉ việc (Target=1) chỉ chiếm khoảng 25%, khiến mô hình có xu hướng dự đoán nghiêng về lớp 0 (Accuracy cao nhưng Recall thấp).
    * *Giải pháp:* Không chỉ dựa vào Accuracy, nhóm sử dụng thêm **Confusion Matrix**, **Precision** và **Recall** để đánh giá thực chất hiệu quả của mô hình.

## Future Improvements

Để nâng cao hiệu suất của dự án, các hướng phát triển tiếp theo bao gồm:

1.  **Cân bằng dữ liệu:** Cài đặt thuật toán **SMOTE** (Synthetic Minority Over-sampling Technique) từ đầu hoặc thêm tham số `class_weights` vào hàm Loss để cải thiện khả năng dự đoán lớp thiểu số.
2.  **Tối ưu Hyperparameters:** Xây dựng hàm **Grid Search** để tự động tìm kiếm các tham số tốt nhất (ví dụ: số lượng cây `n_estimators` trong Random Forest, độ sâu `max_depth` trong Decision Tree).
3.  **Feature Engineering:**
    * Thử nghiệm giảm chiều dữ liệu bằng PCA (Principal Component Analysis).
    * Loại bỏ các đặc trưng nhiễu hoặc ít tương quan (như `training_hours`).

## Contributors

Bài tập được thực hiện bởi:

Huỳnh Đặng Ngọc Hân
MSSV: 23120042

## Contact

Mọi ý kiến đóng góp hoặc thắc mắc, vui lòng liên hệ:
* **Email:** huynhdnhannd@gmail.com
* **GitHub:** [Link GitHub của bạn]

## License

Dự án này được phân phối dưới giấy phép **MIT License**.
