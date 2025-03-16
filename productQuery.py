from neo4j import GraphDatabase
import google.generativeai as genai
import json
import threading

# Thông tin kết nối Neo4j
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "hqiineo4jay"

# Singleton Neo4j Connection
class Neo4jConnection:
    _instance = None
    _lock = threading.Lock()  # Để tránh tạo nhiều kết nối cùng lúc khi chạy đa luồng

    def __new__(cls, uri, user, password):
        with cls._lock:  # Đảm bảo chỉ một luồng có thể khởi tạo kết nối
            if cls._instance is None:
                cls._instance = super(Neo4jConnection, cls).__new__(cls)
                cls._instance._driver = GraphDatabase.driver(uri, auth=(user, password), keep_alive=True)
                print("✅ Neo4j Connection Initialized!")  # Chỉ in một lần
            return cls._instance

    def query(self, cypher_query):
        with self._driver.session() as session:
            return session.run(cypher_query).data()

    def close(self):
        self._driver.close()

# Khởi tạo kết nối toàn cục
neo4j_conn = Neo4jConnection(NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)

# Cấu hình Gemini API
genai.configure(api_key="AIzaSyAmLesw2keGhIrZPMEyYJUs1PUqIidIWFU")
model = genai.GenerativeModel("gemini-2.0-flash")

# Đọc schema
with open("data/schema.json", "r", encoding="utf-8") as f:
    raw_schema = json.load(f)

schema = {
    "nodes": raw_schema["nodes"],
    "relationships": list(set(rel["type"] for rel in raw_schema["relationships"]))
}

def generate_cypher_prompt(chathistory,text):
    CYPHER_GENERATION_TEMPLATE = f"""Nhiệm vụ: Tạo một truy vấn Cypher để truy vấn cơ sở dữ liệu đồ thị neo4j.
        Hướng dẫn:
        - Phân tích câu hỏi và trích xuất các thành phần đồ thị liên quan một cách linh hoạt. Sử dụng thông tin này để xây dựng truy vấn Cypher.
        - Chỉ sử dụng các loại quan hệ và thuộc tính từ sơ đồ đã cung cấp. Không bao gồm bất kỳ loại quan hệ hoặc thuộc tính nào khác.
        - Sơ đồ được xây dựng dựa trên cấu trúc đồ thị với các nút và quan hệ như sau:
        {json.dumps(schema, indent=4, ensure_ascii=False)}
        - Chỉ trả về truy vấn Cypher đã được tạo trong phản hồi của bạn. Không bao gồm giải thích, chú thích hoặc bất kỳ văn bản bổ sung nào khác. TÔI NHẮC LẠI NÓ CHỈ LÀ CÂU TRUY VẤN MÀ TÔI CÓ THỂ LẤY TOÀN BỘ VĂN BẢN CỦA BẠN ĐỂ CHẠY NHƯ 1 CÂU TRUY VẤN MÀ KHÔNG CHỨA VĂN BẢN KHÔNG PHẢI TRUY VẤN KHÔNG CẦN THIẾT
        - Đảm bảo truy vấn Cypher phản hồi chính xác câu hỏi được đưa ra theo đúng sơ đồ.
        
        Ví dụ:
        #tôi muốn tìm sản phẩm áo khoác dạ dài
        MATCH (CLOTH)-[:CÓ_MÔ_TẢ]->(DES)
        WHERE CLOTH.name = 'Áo khoác dạ dài'
        RETURN DES

        #tôi muốn mua quần tây nam 
        MATCH (p)-[r]->(related)
        WHERE p.name = "Quần tây nam"
        RETURN p.name AS SanPham, type(r) AS MoiQuanHe, related.name AS GiaTri;

        #tôi muốn mua đồ phù hợp với dạo phố
        MATCH (p)-[:PHÙ_HỢP_VỚI]->(related)
        WHERE related.name = "Dạo phố"
        RETURN p

        #tôi muốn mua quần jean skinny
        MATCH (p)-[r]->(related)
        WHERE p.name = "Quần jean skinny"
        RETURN p.name AS SanPham, type(r) AS MoiQuanHe, related.name AS GiaTri;
        
        #áo sát nách thể thao có giá bao nhiêu
        MATCH (a)-[:CÓ_GIÁ]->(b)
        WHERE a.name = "Áo sát nách thể thao"
        RETURN b.name

        #Áo sát nách thể thao có giá bao nhiêu
        MATCH (a)-[:CÓ_GIÁ]->(b)
        WHERE a.name = "Áo sát nách thể thao"
        RETURN b.name
        
        #có áo polo nào thuộc brand Lacoste không
        MATCH (p)-[:THUỘC_THƯƠNG_HIỆU]->(brand) WHERE brand.name = "Lacoste" AND p.name CONTAINS "Áo polo"
        return p.name

        #có sản phẩm nào của hãng zara không
        MATCH (p)-[:THUỘC_THƯƠNG_HIỆU]->(related)
        WHERE related.name = "Zara"
        RETURN p

        #are there any product of zara?
        MATCH (p)-[:THUỘC_THƯƠNG_HIỆU]->(brand)
        WHERE brand.name = "Zara"
        RETURN p

        #giới thiệu sản phẩm váy suông midi đi
        MATCH (p)-[r]->(related)
        WHERE p.name = "Váy suông midi"
        RETURN p.name AS SanPham, type(r) AS MoiQuanHe, related.name AS GiaTri;
        
        Lịch sử trò chuyện của bạn với khách hàng như sau(có thể chưa có):{chathistory}
        câu cần bạn sinh truy vấn là:
        {text}
        """

    response = model.generate_content(CYPHER_GENERATION_TEMPLATE)
    return response.text 

def clean_cypher_code(code: str) -> str:
    return code.replace("```cypher", "").replace("```", "").strip()

def productsResponse(chathistory,user_message):
    cypher_result = generate_cypher_prompt(chathistory,user_message)
    cypher_result = clean_cypher_code(cypher_result)
    result = neo4j_conn.query(cypher_result)
    return result

# Chỉ chạy đoạn này nếu file được chạy độc lập
if __name__ == "__main__":
    print("✅ Connected to Neo4j database!")
