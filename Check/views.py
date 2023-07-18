from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import seaborn as sns
from django.http import HttpResponse

# Create your views here.

def import_model(request):
    #Chèn đường dẫn file
    data_path = 'Check/Modified_SQL_Dataset.csv'

    #Đọc data
    df = pd.read_csv(data_path)

    #Trích xuất dữ liệu
    X = df['Query'].tolist()
    Y = df['Label'].tolist()

    #Mã hóa dữ liệu
    encoder = CountVectorizer(ngram_range=(1, 1))
    encoder.fit(X)

    if request.method == 'POST':

        # Đường dẫn đến file pickle
        pickle_file_path = 'Check/SVMModel.pkl'

        # Đọc dữ liệu từ file pickle
        with open(pickle_file_path, 'rb') as f:
            model_data = pickle.load(f)

        READ_Label = {
            1: "Positive (Dương tính)",
            0: "Negative (Âm tính)"
        }

        input_text = request.POST.get('input_text', '')  # Lấy giá trị từ request.POST nếu cần

        input_encoder = encoder.transform([input_text])
        label = model_data.predict(input_encoder)

        result_label = READ_Label[label[0]]

        # Trả về kết quả dưới dạng context để hiển thị trên templates
        context = {
            'result_label': result_label
        }
        return render(request, 'predict.html', context)
    # Trả về templates gửi yêu cầu nhập câu dự đoán
    return render(request, 'predict.html')


import base64
def visualization(request):
    # Đường dẫn của hai hình ảnh
    image_data_path = 'Check/visualization/image_data.png'
    image_SVM_path = 'Check/visualization/image_SVM.png'

    # Chuyển đổi hình ảnh 1 thành base64
    with open(image_data_path, 'rb') as image1_file:
        image1_data = image1_file.read()
        image1_base64 = base64.b64encode(image1_data).decode('utf-8')

    # Chuyển đổi hình ảnh 2 thành base64
    with open(image_SVM_path, 'rb') as image2_file:
        image2_data = image2_file.read()
        image2_base64 = base64.b64encode(image2_data).decode('utf-8')

    context = {
        'image1_base64': image1_base64,
        'image2_base64': image2_base64,
    }

    # Trả về template và truyền dữ liệu ảnh base64
    return render(request, 'visualization.html', context)