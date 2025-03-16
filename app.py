import faiss
import json
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS, cross_origin
import os
from classifier import load_model, predict, TransformerClassifier
from productQuery import productsResponse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend truy cập API

# 🔹 Load API Key của Gemini
genai.configure(api_key="AIzaSyAmLesw2keGhIrZPMEyYJUs1PUqIidIWFU")
model = genai.GenerativeModel("gemini-2.0-flash")
classify_model = load_model("training/model.pkl")

# Lịch sử trò chuyện (tối đa 5 tin nhắn)
chat_history = []

with open("data/store.json", "r", encoding="utf-8") as f:
    intro_info = json.load(f)

# Load dữ liệu từ JSON
def load_terms(filename="data/terms.json"):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def classify_question(user_input):
    text = user_input
    result = predict(text, classify_model)
    print(result)
    return result

# Tạo FAISS Index
def build_faiss_index(data):
    texts = [entry["title"] + " - " + entry["content"] for entry in data]
    embeddings = model_emb.encode(texts, convert_to_numpy=True)

    # Tạo FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, texts, data

# Tìm kiếm điều khoản liên quan
def search_terms(query, index, texts, data, top_k=1):
    query_embedding = model_emb.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(data[idx])  # Lấy nội dung từ dữ liệu gốc

    return results


@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    global chat_history  # Sử dụng biến toàn cục để lưu lịch sử trò chuyện
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    category = classify_question(user_message)
    
    # Thêm tin nhắn mới vào lịch sử
    chat_history.append(f"Người dùng: {user_message}")
    
    # Nếu lịch sử vượt quá 5 tin nhắn, xóa tin nhắn cũ nhất
    if len(chat_history) > 4:
        chat_history.pop(0)
    # Ghép lịch sử tin nhắn vào prompt
    history_context = "\n".join(chat_history)
    
    if category == "product":
        try:
            response = productsResponse(history_context, user_message)
            print(response)
            final_prompt = f"Lịch sử trò chuyện:\n{history_context}\nCâu hỏi của khách hàng: {user_message}\n Thông tin tìm được: {response}\nTrả lời một cách thân thiện như một nhân viên bán hàng."
            response = model.generate_content(final_prompt)
            chat_history.append(f"Bot: {response.text.replace("\n", "<br>")}")
            return jsonify({"reply": response.text.replace("\n", "<br>")})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif category == "store_info":
        context = (
            f" Tên cửa hàng: {intro_info['name']}\n"
            f" Địa chỉ: {intro_info['address']}\n"
            f" Thời gian mở cửa: {intro_info['open_hours']}\n"
            f" Liên hệ: {intro_info['contact']}\n"
            f" {intro_info['description']}"
        )
        final_prompt = f"Lịch sử trò chuyện:\n{history_context}\nCâu hỏi của khách hàng: {user_message}\n Thông tin tìm được: {context}\nTrả lời một cách thân thiện như một nhân viên bán hàng."
        response = model.generate_content(final_prompt)
        chat_history.append(f"Bot: {response.text.replace("\n", "<br>")}")
        return jsonify({"reply": response.text.replace("\n", "<br>")})
    
    elif category == "else":
        final_prompt = f"Lịch sử trò chuyện:\n{history_context}\nCâu hỏi của khách hàng: {user_message}\n Hãy trả lời họ như là một người bán hàng vui vẻ và thân thiện hài hước."
        response = model.generate_content(final_prompt)
        chat_history.append(f"Bot: {response.text.replace("\n", "<br>")}")
        return jsonify({"reply": response.text.replace("\n", "<br>")})
    
    elif category == "terms":
        context = search_terms(user_message, faiss_index, faiss_texts, faiss_data)
        final_prompt = f"Lịch sử trò chuyện:\n{history_context}\nCâu hỏi của khách hàng: {user_message}\n Thông tin tìm được: {context}\nTrả lời một cách thân thiện như một nhân viên bán hàng."
        response = model.generate_content(final_prompt)
        chat_history.append(f"Bot: {response.text.replace("\n", "<br>")}")
        return jsonify({"reply": response.text.replace("\n", "<br>")})
    
    if len(chat_history) > 4:
        chat_history.pop(0)

if __name__ == "__main__":
    data = load_terms()
    faiss_index, faiss_texts, faiss_data = build_faiss_index(data)
    app.run(debug=True, port=5001)
