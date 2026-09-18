import json
import math
import os
import random

import streamlit as st

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# 1. CẤU HÌNH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUIZ_FILE = os.path.join(BASE_DIR, "quiz_data.json")
CRISIS_FILE = os.path.join(BASE_DIR, "crises.json")
SAVE_FILE = os.path.join(BASE_DIR, "save_game.json")


def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "").strip()

LEVEL_NAMES = {
    1: "Level 1 · Cơ bản",
    2: "Level 2 · Sơ cấp",
    3: "Level 3 · Trung cấp",
    4: "Level 4 · Nâng cao",
    5: "Level 5 · Chuyên gia",
}
LEVEL_ICONS = {1: "🔰", 2: "⭐", 3: "🏆", 4: "🚀", 5: "👑"}


@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY or genai is None:
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Không thể khởi tạo Gemini:", e)
        return None


client = get_gemini_client()


# ============================================================
# 2. HÀM HỖ TRỢ GEMINI
# ============================================================

def call_gemini(prompt):
    if client is None:
        return None, "Chưa có GEMINI_API_KEY hoặc Gemini chưa được khởi tạo."
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini không trả về nội dung."
        return text.strip(), None
    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            return None, "Gemini đang hết quota (429 RESOURCE_EXHAUSTED). Hệ thống đã chuyển sang chế độ Offline."
        return None, f"Lỗi Gemini: {error_text}"


def extract_json_from_text(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def safe_score(value, default=5):
    try:
        value = float(value)
        return round(max(0, min(10, value)), 1)
    except (ValueError, TypeError):
        return float(default)


def format_vnd(amount):
    """Định dạng số tiền theo kiểu Việt Nam, ví dụ 12.500.000 ₫"""
    try:
        amount = int(round(float(amount)))
    except (ValueError, TypeError):
        amount = 0
    sign = "-" if amount < 0 else ""
    return f"{sign}{abs(amount):,}".replace(",", ".") + " ₫"


# ============================================================
# 3. OFFLINE AI TUTOR / CRISIS (giữ nguyên logic)
# ============================================================

def generate_offline_tutor_response(user_query):
    query = user_query.lower()

    if any(w in query for w in ["giá", "pricing", "price", "tăng giá", "giảm giá"]):
        return ("Gợi ý Offline:\nKhi quyết định giá trong du lịch, nên cân nhắc chi phí, "
                "nhu cầu khách hàng, giá của đối thủ và giá trị mà khách nhận được.\n\n"
                "Có thể sử dụng giá linh hoạt theo mùa cao điểm và thấp điểm.")

    if any(w in query for w in ["khách", "customer", "phàn nàn", "khiếu nại", "dịch vụ", "service"]):
        return ("Gợi ý Offline:\nHãy xác định vấn đề của khách trước, sau đó đưa ra phương án "
                "giải quyết hợp lý.\n\nQuy trình cơ bản:\n1. Lắng nghe khách hàng.\n"
                "2. Xác định nguyên nhân.\n3. Đưa ra giải pháp.\n4. Theo dõi mức độ hài lòng sau xử lý.")

    if any(w in query for w in ["khủng hoảng", "crisis", "cháy", "tai nạn", "rủi ro", "nguy hiểm"]):
        return ("Gợi ý Offline:\nTrong khủng hoảng du lịch, ưu tiên đầu tiên là an toàn của khách "
                "và nhân viên.\n\nSau đó cần:\n1. Xác minh thông tin.\n2. Kiểm soát tình hình.\n"
                "3. Giao tiếp minh bạch.\n4. Đưa ra phương án phục hồi.")

    if any(w in query for w in ["môi trường", "xanh", "bền vững", "sustainability", "rác", "eco"]):
        return ("Gợi ý Offline:\nDu lịch bền vững cần cân bằng giữa lợi ích kinh tế, trải nghiệm "
                "khách hàng và bảo vệ môi trường.\n\nGiảm nhựa dùng một lần, tiết kiệm năng lượng "
                "và hợp tác với cộng đồng địa phương là những giải pháp khả thi.")

    if any(w in query for w in ["nhân viên", "nhân sự", "employee", "staff", "hr", "đào tạo", "tuyển"]):
        return ("Gợi ý Offline:\nQuản trị nhân sự trong du lịch nên chú ý đến tuyển dụng, đào tạo, "
                "động lực làm việc và chất lượng dịch vụ.\n\nNhân viên hài lòng thường tạo ra "
                "trải nghiệm khách hàng tốt hơn.")

    return ("Gợi ý Offline:\nHãy thử phân tích vấn đề theo 4 bước:\n1. Xác định vấn đề.\n"
            "2. Xác định nguyên nhân.\n3. So sánh các phương án.\n4. Dự đoán hậu quả trước khi quyết định.\n\n"
            "Nếu có nhiều bên liên quan, hãy cân nhắc lợi ích của khách hàng, nhân viên, doanh nghiệp và cộng đồng.")


def generate_offline_crisis_evaluation(user_solution, crisis):
    solution = user_solution.lower()
    scores = {
        "decision_making": 5, "risk_management": 5, "customer_service": 5,
        "financial_management": 5, "reputation_management": 5, "feasibility": 5,
    }
    strengths, weaknesses = [], []

    checks = [
        ("decision_making", ["ưu tiên", "quyết định", "giải quyết", "phương án", "kế hoạch", "xử lý"],
         "Bạn đã thể hiện khả năng đưa ra quyết định.", "Quyết định chưa được trình bày đủ rõ ràng.", True),
        ("risk_management", ["an toàn", "rủi ro", "kiểm tra", "phòng", "bảo vệ", "khẩn cấp"],
         "Bạn có chú ý đến quản trị rủi ro và an toàn.", "Cần phân tích rủi ro và phương án dự phòng rõ hơn.", True),
        ("customer_service", ["khách", "bồi thường", "xin lỗi", "hỗ trợ", "hoàn tiền", "đổi", "chăm sóc"],
         "Bạn có quan tâm đến trải nghiệm khách hàng.", "Cần nói rõ hơn cách bảo vệ quyền lợi khách hàng.", True),
        ("financial_management", ["chi phí", "ngân sách", "tiền", "doanh thu", "lợi nhuận", "tiết kiệm", "chi"],
         "Bạn đã cân nhắc yếu tố tài chính.", "Chưa đề cập nhiều đến tác động tài chính.", False),
        ("reputation_management", ["uy tín", "thương hiệu", "truyền thông", "minh bạch", "thông báo", "mạng xã hội", "danh tiếng"],
         "Bạn có chú ý đến uy tín và hình ảnh doanh nghiệp.", "Nên có kế hoạch truyền thông để bảo vệ uy tín.", False),
        ("feasibility", ["nhân viên", "thời gian", "nguồn lực", "có thể", "thực hiện", "phối hợp", "đối tác"],
         "Giải pháp có xét đến khả năng triển khai thực tế.", "Nên làm rõ nguồn lực và cách triển khai.", False),
    ]

    for key, keywords, good_msg, bad_msg, penalize in checks:
        if any(w in solution for w in keywords):
            scores[key] += 2
            strengths.append(good_msg)
        else:
            if penalize:
                scores[key] -= 1
            weaknesses.append(bad_msg)

    for key in scores:
        scores[key] = safe_score(scores[key])

    if not strengths:
        strengths.append("Bạn đã đưa ra một hướng xử lý ban đầu.")
    if not weaknesses:
        weaknesses.append("Có thể bổ sung thêm phương án dự phòng.")

    feedback = ("Đây là đánh giá Offline vì Gemini hiện không khả dụng. Khi có thể sử dụng Gemini, "
                "hệ thống sẽ đánh giá câu trả lời theo ngữ cảnh cụ thể của tình huống.")

    return {**scores, "strengths": strengths, "weaknesses": weaknesses, "feedback": feedback}


# ============================================================
# 4. MODEL GAME
# ============================================================

class TourismGameModel:

    CRISIS_WEIGHTS = {
        "decision_making": 0.25, "risk_management": 0.20, "customer_service": 0.20,
        "financial_management": 0.15, "reputation_management": 0.10, "feasibility": 0.10,
    }

    def __init__(self):
        self.xp = 0
        self.level_name = "Tourism Beginner"
        self.budget = 50_000_000
        self.staff_count = 5
        self.market_index = 100.0
        self.market_history = [100.0]
        self.turn = 0

        self.stats = {
            "customer_satisfaction": 75,
            "business_reputation": 70,
            "employee_satisfaction": 65,
            "financial_health": 72,
            "sustainability": 80,
        }

        self.history_log = []
        self.used_crises = []

        self.quiz_data = self.load_json(QUIZ_FILE, [])
        self.crises = self.load_json(CRISIS_FILE, [])

        # Cấu trúc: { level: { set_no: [câu hỏi...] } }
        self.quiz_structure = self.build_quiz_structure()
        # Tiến trình: { "1": {"sets": {"1": {...}, "2": {...}}, "completed": False}, ... }
        self.quiz_progress = self._init_quiz_progress()

        self.load_progress()

    @staticmethod
    def load_json(path, default):
        try:
            if not os.path.exists(path):
                return default
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Lỗi đọc {path}: {e}")
            return default

    # ---- QUIZ: LEVEL > NHIỀU BỘ (SET) ----

    @staticmethod
    def _split_into_sets(questions, max_sets=3):
        """Chia 1 danh sách câu hỏi thành tối đa max_sets bộ đều nhau."""
        if not questions:
            return {}
        num_sets = min(max_sets, len(questions)) or 1
        chunk = max(1, math.ceil(len(questions) / num_sets))
        sets = {}
        for index, q in enumerate(questions):
            set_no = min(num_sets, index // chunk + 1)
            sets.setdefault(set_no, []).append(q)
        return sets

    def build_quiz_structure(self):
        """Trả về { level(1-5): { set_no: [câu hỏi] } }.
        Ưu tiên trường "level" + "set" có sẵn trong quiz_data.json;
        nếu thiếu, tự suy ra để mỗi level có tối đa 3 bộ."""

        structure = {i: {} for i in range(1, 6)}
        if not self.quiz_data:
            return structure

        has_level = any(isinstance(q, dict) and "level" in q for q in self.quiz_data)
        has_set = any(isinstance(q, dict) and "set" in q for q in self.quiz_data)

        def clamp_level(v):
            try:
                return max(1, min(5, int(v)))
            except (ValueError, TypeError):
                return 1

        if has_level and has_set:
            for q in self.quiz_data:
                lvl = clamp_level(q.get("level", 1))
                try:
                    set_no = max(1, int(q.get("set", 1)))
                except (ValueError, TypeError):
                    set_no = 1
                structure[lvl].setdefault(set_no, []).append(q)
        elif has_level:
            by_level = {i: [] for i in range(1, 6)}
            for q in self.quiz_data:
                by_level[clamp_level(q.get("level", 1))].append(q)
            for lvl, qs in by_level.items():
                structure[lvl] = self._split_into_sets(qs)
        else:
            n = len(self.quiz_data)
            chunk = max(1, math.ceil(n / 5))
            by_level = {i: [] for i in range(1, 6)}
            for index, q in enumerate(self.quiz_data):
                by_level[min(5, index // chunk + 1)].append(q)
            for lvl, qs in by_level.items():
                structure[lvl] = self._split_into_sets(qs)

        return structure

    def _init_quiz_progress(self):
        progress = {}
        for level in range(1, 6):
            set_ids = self.get_set_ids(level)
            progress[str(level)] = {
                "sets": {str(s): {"correct": 0, "total": 0, "completed": False} for s in set_ids},
                "completed": False,
            }
        return progress

    def get_set_ids(self, level):
        return sorted(self.quiz_structure.get(level, {}).keys())

    def get_set_questions(self, level, set_no):
        return self.quiz_structure.get(level, {}).get(set_no, [])

    def is_level_unlocked(self, level):
        if level == 1:
            return True
        prev = self.quiz_progress.get(str(level - 1), {})
        return prev.get("completed", False)

    def get_set_progress(self, level, set_no):
        lvl_prog = self.quiz_progress.get(str(level), {})
        return lvl_prog.get("sets", {}).get(str(set_no), {"correct": 0, "total": 0, "completed": False})

    def record_quiz_answer(self, level, set_no, correct):
        lvl_key, set_key = str(level), str(set_no)
        lvl_prog = self.quiz_progress.setdefault(lvl_key, {"sets": {}, "completed": False})
        set_prog = lvl_prog["sets"].setdefault(set_key, {"correct": 0, "total": 0, "completed": False})
        set_prog["total"] += 1
        if correct:
            set_prog["correct"] += 1
        self.save_progress()

    def complete_set(self, level, set_no):
        """Đánh dấu 1 bộ đã hoàn thành. Nếu TẤT CẢ các bộ trong level đã
        hoàn thành thì đánh dấu cả level hoàn thành (mở khoá level tiếp theo).
        Trả về True nếu đây là lần đầu cả LEVEL được hoàn thành."""
        lvl_key, set_key = str(level), str(set_no)
        lvl_prog = self.quiz_progress.setdefault(lvl_key, {"sets": {}, "completed": False})
        set_prog = lvl_prog["sets"].setdefault(set_key, {"correct": 0, "total": 0, "completed": False})
        set_prog["completed"] = True

        was_level_completed = lvl_prog["completed"]
        set_ids = self.get_set_ids(level)
        all_sets_done = bool(set_ids) and all(
            lvl_prog["sets"].get(str(s), {}).get("completed", False) for s in set_ids
        )
        if all_sets_done:
            lvl_prog["completed"] = True

        self.save_progress()
        return (not was_level_completed) and lvl_prog["completed"]

    # ---- SAVE / LOAD ----

    def save_progress(self):
        data = {
            "xp": self.xp, "level_name": self.level_name, "budget": self.budget,
            "staff_count": self.staff_count, "stats": self.stats,
            "history_log": self.history_log, "quiz_progress": self.quiz_progress,
            "market_index": self.market_index, "market_history": self.market_history,
            "turn": self.turn,
        }
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print("Lỗi lưu game:", e)

    def load_progress(self):
        if not os.path.exists(SAVE_FILE):
            return
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            self.xp = data.get("xp", self.xp)
            self.level_name = data.get("level_name", self.level_name)
            self.budget = data.get("budget", self.budget)
            self.staff_count = data.get("staff_count", self.staff_count)
            self.market_index = data.get("market_index", self.market_index)
            self.market_history = data.get("market_history", self.market_history)
            self.turn = data.get("turn", self.turn)
            saved_stats = data.get("stats", {})
            for key in self.stats:
                if key in saved_stats:
                    self.stats[key] = saved_stats[key]
            self.history_log = data.get("history_log", [])
            saved_quiz_progress = data.get("quiz_progress", {})
            for lvl_key, lvl_val in saved_quiz_progress.items():
                if lvl_key not in self.quiz_progress or not isinstance(lvl_val, dict):
                    continue
                self.quiz_progress[lvl_key]["completed"] = lvl_val.get("completed", False)
                for set_key, set_val in lvl_val.get("sets", {}).items():
                    if set_key in self.quiz_progress[lvl_key]["sets"] and isinstance(set_val, dict):
                        self.quiz_progress[lvl_key]["sets"][set_key].update(set_val)
            self.update_level_only()
        except Exception as e:
            print("Lỗi load save:", e)

    # ---- LEVEL DOANH NGHIỆP ----

    def update_level_only(self):
        if self.xp >= 1500:
            self.level_name = "Tourism Expert"
        elif self.xp >= 1000:
            self.level_name = "Tourism Strategist"
        elif self.xp >= 600:
            self.level_name = "Tourism Manager"
        elif self.xp >= 300:
            self.level_name = "Tourism Planner"
        elif self.xp >= 100:
            self.level_name = "Tourism Trainee"
        else:
            self.level_name = "Tourism Beginner"

    def update_xp(self, amount):
        self.xp = max(0, self.xp + amount)
        old_level = self.level_name
        self.update_level_only()
        self.save_progress()
        return old_level != self.level_name

    def log_activity(self, text):
        self.history_log.append(text)
        self.save_progress()

    def calculate_crisis_score(self, evaluation):
        total = 0
        for criterion, weight in self.CRISIS_WEIGHTS.items():
            total += safe_score(evaluation.get(criterion, 5)) * weight
        return round(total, 2)

    def update_business_stats(self, evaluation):
        mapping = {
            "customer_service": "customer_satisfaction",
            "reputation_management": "business_reputation",
            "financial_management": "financial_health",
            "risk_management": "sustainability",
            "decision_making": "employee_satisfaction",
        }
        total_score = self.calculate_crisis_score(evaluation)
        for criterion, stat_name in mapping.items():
            score = safe_score(evaluation.get(criterion, 5))
            delta = (score - 5) * 2
            current = self.stats.get(stat_name, 50) + delta
            self.stats[stat_name] = round(max(0, min(100, current)), 1)
        self.log_activity(f"Quản lý khủng hoảng: điểm tổng {total_score}/10")
        return total_score


def normalize_evaluation(evaluation):
    if not isinstance(evaluation, dict):
        evaluation = {}
    result = {}
    criteria = [
        "decision_making", "risk_management", "customer_service",
        "financial_management", "reputation_management", "feasibility",
    ]
    for criterion in criteria:
        result[criterion] = safe_score(evaluation.get(criterion, 5))

    strengths = evaluation.get("strengths", [])
    weaknesses = evaluation.get("weaknesses", [])
    feedback = evaluation.get("feedback", "")

    if isinstance(strengths, str):
        strengths = [strengths]
    elif not isinstance(strengths, list):
        strengths = []

    if isinstance(weaknesses, str):
        weaknesses = [weaknesses]
    elif not isinstance(weaknesses, list):
        weaknesses = []

    if not isinstance(feedback, str):
        feedback = str(feedback)

    result["strengths"] = strengths
    result["weaknesses"] = weaknesses
    result["feedback"] = feedback
    return result


# ============================================================
# 5. THEME — "ĐỊA LÝ + KINH DOANH"
#    Navy đại dương / Vàng gold doanh nhân / Kem cát du lịch
# ============================================================

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #F8F3E7 0%, #F4EFE1 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2545 0%, #13315C 100%);
    }
    section[data-testid="stSidebar"] * { color: #F1EAD6 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(241,234,214,0.2); }

    /* Nút trong sidebar: nền đủ tối để chữ kem sáng luôn rõ, có viền nhẹ
       phân biệt với nền sidebar; trạng thái đang chọn (primary) nổi bật vàng gold */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(241, 234, 214, 0.08) !important;
        border: 1px solid rgba(241, 234, 214, 0.35) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(212, 160, 23, 0.25) !important;
        border-color: #D4A017 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #D4A017 !important;
        border: 1px solid #D4A017 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
        color: #0B2545 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #E8B923 !important;
    }

    h1, h2, h3 { font-family: 'Poppins', sans-serif !important; color: #0B2545 !important; }

    .hero-banner {
        background: linear-gradient(120deg, #0B2545 0%, #13315C 55%, #0F766E 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(11,37,69,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "🌍";
        position: absolute;
        right: 18px;
        top: 8px;
        font-size: 90px;
        opacity: 0.12;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        color: #FDF6E3;
        font-size: 30px;
        font-weight: 700;
        margin: 0;
    }
    .hero-subtitle {
        color: #D4A017;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .topbar {
        display: flex;
        gap: 12px;
        margin-bottom: 18px;
    }
    .topbar-item {
        flex: 1;
        background: #FFFFFF;
        border-radius: 12px;
        padding: 10px 14px;
        border-left: 4px solid #D4A017;
        box-shadow: 0 2px 8px rgba(11,37,69,0.08);
    }
    .topbar-label { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .topbar-value { font-size: 18px; font-weight: 700; color: #0B2545; font-family: 'Poppins', sans-serif; }

    .biz-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        border-left: 5px solid #0F766E;
        box-shadow: 0 3px 12px rgba(11,37,69,0.08);
        margin-bottom: 16px;
    }
    .biz-card.gold { border-left-color: #D4A017; }
    .biz-card.locked { border-left-color: #9CA3AF; opacity: 0.65; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-done { background: #DCFCE7; color: #15803D; }
    .badge-progress { background: #FEF3C7; color: #B45309; }
    .badge-locked { background: #E5E7EB; color: #6B7280; }

    /* Nút bấm trong vùng nội dung chính: mặc định phải NHÌN RÕ ngay
       (nền trắng, viền + chữ navy đậm) — chỉ đổi màu khi hover/bấm. */
    [data-testid="stAppViewContainer"] .stButton > button {
        background: #FFFFFF !important;
        color: #0B2545 !important;
        border: 2px solid #0B2545 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out;
    }
    [data-testid="stAppViewContainer"] .stButton > button:hover {
        background: #0B2545 !important;
        color: #FDF6E3 !important;
        border-color: #0B2545 !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button:disabled {
        background: #E5E7EB !important;
        color: #9CA3AF !important;
        border-color: #E5E7EB !important;
    }
    /* Nút primary (Thực hiện, nav đang chọn...): nền navy sẵn, hover sang vàng gold */
    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"] {
        background: #0B2545 !important;
        color: #FDF6E3 !important;
        border-color: #0B2545 !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover {
        background: #D4A017 !important;
        color: #0B2545 !important;
        border-color: #D4A017 !important;
    }
    /* Đảm bảo chữ trong nút luôn ăn theo màu của nút (không bị luật ép màu chữ chung ghi đè) */
    [data-testid="stAppViewContainer"] .stButton > button [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] .stButton > button [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }

    .stat-caption { color: #6B7280; font-size: 12px; margin-bottom: -6px; }

    /* Ép chữ trong vùng nội dung chính luôn là màu tối, bất kể nền vàng/trắng/kem
       và bất kể người dùng đang bật theme sáng hay tối của Streamlit.
       (Không áp dụng cho sidebar nền navy, và không đụng tới các span/div đã
       tự set màu riêng như hero-title, badge, topbar-value...) */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricDelta"] {
        color: #1F2933 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
        color: #4B5563 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
        color: #1F2933 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def hero_banner(title, subtitle):
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🧭 {title}</div>
        <div class="hero-subtitle">🗺️ ĐỊA LÝ &nbsp;·&nbsp; 💼 KINH DOANH &nbsp;·&nbsp; ✈️ DU LỊCH — {subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def topbar():
    m = st.session_state.model
    items = [
        ("⭐ XP", m.xp),
        ("💼 Level", m.level_name),
        ("💰 Ngân sách", format_vnd(m.budget)),
        ("👥 Nhân sự", m.staff_count),
        ("📈 Chỉ số TT", f"{m.market_index:.1f}"),
    ]
    html = '<div class="topbar">'
    for label, value in items:
        html += f'<div class="topbar-item"><div class="topbar-label">{label}</div><div class="topbar-value">{value}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def stat_row(stats_dict):
    labels = {
        "customer_satisfaction": "🧳 Customer",
        "business_reputation": "🏛️ Reputation",
        "employee_satisfaction": "🧑‍💼 Employee",
        "financial_health": "📊 Financial",
        "sustainability": "🌱 Sustainability",
    }
    cols = st.columns(len(labels))
    for col, (key, label) in zip(cols, labels.items()):
        with col:
            st.markdown(f'<div class="stat-caption">{label}</div>', unsafe_allow_html=True)
            st.progress(int(round(stats_dict[key])) / 100)
            st.caption(f"{round(stats_dict[key])}/100")


# ============================================================
# 6. KHỞI TẠO SESSION STATE
# ============================================================

def init_state():
    if "model" not in st.session_state:
        st.session_state.model = TourismGameModel()
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "current_level" not in st.session_state:
        st.session_state.current_level = None
    if "current_set" not in st.session_state:
        st.session_state.current_set = None
    if "level_questions" not in st.session_state:
        st.session_state.level_questions = []
    if "level_pos" not in st.session_state:
        st.session_state.level_pos = 0
    if "level_correct" not in st.session_state:
        st.session_state.level_correct = 0
    if "answered" not in st.session_state:
        st.session_state.answered = False
    if "last_correct" not in st.session_state:
        st.session_state.last_correct = None
    if "current_crisis" not in st.session_state:
        st.session_state.current_crisis = None
    if "crisis_result" not in st.session_state:
        st.session_state.crisis_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_turn_result" not in st.session_state:
        st.session_state.last_turn_result = None


def go(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# 7. SIDEBAR ĐIỀU HƯỚNG
# ============================================================

def sidebar_nav():
    m = st.session_state.model
    with st.sidebar:
        st.markdown("### 🧭 AI Tourism Business")
        st.caption("Simulator — Địa lý & Kinh doanh du lịch")
        st.markdown("---")

        nav_items = [
            ("home", "🏠 Trang chủ"),
            ("quiz_levels", "📚 Quiz (5 Level)"),
            ("ai_tutor", "🤝 AI Tutor"),
            ("crisis", "🌪️ Crisis Management"),
            ("operations", "🏢 Vận hành Doanh nghiệp"),
            ("history", "📜 Lịch sử hoạt động"),
        ]
        current_top = st.session_state.page if st.session_state.page in [p for p, _ in nav_items] else \
            ("quiz_levels" if st.session_state.page in ("quiz_sets", "quiz_play", "quiz_result") else st.session_state.page)

        for key, label in nav_items:
            is_active = key == current_top
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                if key == "home":
                    st.session_state.current_crisis = None
                go(key)

        st.markdown("---")
        st.caption(f"Gemini model: `{GEMINI_MODEL}`")
        st.caption("🟢 Đã kết nối Gemini" if client else "🟡 Chế độ Offline")

        if st.button("💾 Save Game", use_container_width=True):
            m.save_progress()
            st.toast("Đã lưu tiến trình!", icon="💾")


# ============================================================
# 8. TRANG CHỦ
# ============================================================

def page_home():
    m = st.session_state.model
    hero_banner("AI TOURISM BUSINESS SIMULATOR", "Bảng điều khiển doanh nghiệp")
    topbar()

    st.markdown("#### 📊 Chỉ số doanh nghiệp")
    with st.container(border=True):
        stat_row(m.stats)

    st.markdown("#### 🗂️ Trung tâm điều hành")
    cols = st.columns(3)
    cards = [
        ("📚 Quiz 5 Level", "Ôn tập kiến thức du lịch & kinh doanh theo cấp độ.", "quiz_levels"),
        ("🤝 AI Tutor", "Hỏi đáp cùng trợ lý AI về quản trị du lịch.", "ai_tutor"),
        ("🌪️ Crisis Management", "Xử lý tình huống khủng hoảng thực tế.", "crisis"),
        ("🏢 Vận hành Doanh nghiệp", "Tuyển/đào tạo nhân sự & phân bổ ngân sách — 1 hệ thống thống nhất.", "operations"),
        ("📜 Lịch sử", "Xem lại các quyết định đã thực hiện.", "history"),
    ]
    for i, (title, desc, page_key) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="biz-card gold">
                <b>{title}</b><br>
                <span style="color:#6B7280;font-size:13px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Mở →", key=f"home_open_{page_key}", use_container_width=True):
                go(page_key)

    with st.expander("🎲 Sự kiện ngẫu nhiên (Random Event)"):
        if st.button("Kích hoạt sự kiện"):
            trigger_random_event()


def trigger_random_event():
    m = st.session_state.model
    events = [
        {"text": "Một travel blogger nổi tiếng đăng bài tích cực về doanh nghiệp.",
         "budget_change": -3_000_000, "reputation_change": 5},
        {"text": "Một khách hàng đăng bài phàn nàn trên mạng xã hội.",
         "budget_change": -5_000_000, "reputation_change": -2},
        {"text": "Một đối tác địa phương đề xuất hợp tác quảng bá.",
         "budget_change": 8_000_000, "reputation_change": 10},
    ]
    event = random.choice(events)
    m.budget = max(0, m.budget + event["budget_change"])
    m.stats["business_reputation"] = max(0, min(100, m.stats["business_reputation"] + event["reputation_change"]))
    m.log_activity(f"Random Event: {event['text']}")
    st.success(f"{event['text']}  |  Ngân sách {event['budget_change']:+,}đ  |  Uy tín {event['reputation_change']:+}")


# ============================================================
# 9. QUIZ — CHỌN LEVEL → CHỌN BỘ (SET) → CHƠI
# ============================================================

def page_quiz_levels():
    m = st.session_state.model
    hero_banner("QUIZ NGHIỆP VỤ", "Hoàn thành TẤT CẢ các bộ trong 1 level để mở khoá level tiếp theo")
    topbar()

    if not m.quiz_data:
        st.warning("Không tìm thấy quiz_data.json.")
        return

    cols = st.columns(5)
    for level in range(1, 6):
        set_ids = m.get_set_ids(level)
        lvl_progress = m.quiz_progress.get(str(level), {"sets": {}, "completed": False})
        unlocked = m.is_level_unlocked(level)
        sets_done = sum(1 for s in set_ids if lvl_progress["sets"].get(str(s), {}).get("completed", False))
        total_sets = len(set_ids)

        if lvl_progress["completed"]:
            badge = '<span class="badge badge-done">✓ Hoàn thành</span>'
        elif sets_done > 0:
            badge = '<span class="badge badge-progress">Đang làm</span>'
        elif not unlocked:
            badge = '<span class="badge badge-locked">🔒 Đã khoá</span>'
        else:
            badge = '<span class="badge badge-locked">Chưa bắt đầu</span>'

        card_class = "biz-card" if unlocked else "biz-card locked"
        with cols[level - 1]:
            st.markdown(f"""
            <div class="{card_class}" style="text-align:center;">
                <div style="font-size:26px;">{LEVEL_ICONS.get(level, "📘") if unlocked else "🔒"}</div>
                <b>{LEVEL_NAMES.get(level)}</b><br>
                <span style="color:#6B7280;font-size:12px;">{total_sets} bộ quiz</span><br>
                {badge}<br><br>
                <span style="font-size:11px;color:#6B7280;">Đã hoàn thành {sets_done}/{total_sets} bộ</span>
            </div>
            """, unsafe_allow_html=True)

            disabled = (not unlocked) or (total_sets == 0)
            if st.button("Xem các bộ →", key=f"open_lvl_{level}", use_container_width=True, disabled=disabled):
                st.session_state.current_level = level
                go("quiz_sets")


def page_quiz_sets():
    m = st.session_state.model
    level = st.session_state.current_level
    if level is None or not m.is_level_unlocked(level):
        go("quiz_levels")
        return

    set_ids = m.get_set_ids(level)
    hero_banner(f"CHỌN BỘ QUIZ — {LEVEL_NAMES.get(level, '')}", f"{len(set_ids)} bộ · làm hết cả 3 bộ để mở khoá level tiếp theo")
    topbar()

    if not set_ids:
        st.warning("Level này chưa có bộ quiz nào.")
        st.button("← Chọn level khác", on_click=lambda: go("quiz_levels"))
        return

    cols = st.columns(len(set_ids))
    for i, set_no in enumerate(set_ids):
        questions = m.get_set_questions(level, set_no)
        prog = m.get_set_progress(level, set_no)

        if prog["completed"]:
            badge = '<span class="badge badge-done">✓ Hoàn thành</span>'
        elif prog["total"] > 0:
            badge = '<span class="badge badge-progress">Đang làm</span>'
        else:
            badge = '<span class="badge badge-locked">Chưa bắt đầu</span>'

        with cols[i]:
            st.markdown(f"""
            <div class="biz-card" style="text-align:center;">
                <div style="font-size:22px;">📦</div>
                <b>Bộ {set_no}</b><br>
                <span style="color:#6B7280;font-size:12px;">{len(questions)} câu hỏi</span><br>
                {badge}<br><br>
                <span style="font-size:11px;color:#6B7280;">{prog['correct']}/{prog['total']} đúng</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Bắt đầu bộ này", key=f"start_set_{level}_{set_no}", use_container_width=True,
                         disabled=not questions):
                start_quiz_set(level, set_no)

    if st.button("← Chọn level khác"):
        go("quiz_levels")


def start_quiz_set(level, set_no):
    m = st.session_state.model
    questions = list(m.get_set_questions(level, set_no))
    random.shuffle(questions)
    st.session_state.current_level = level
    st.session_state.current_set = set_no
    st.session_state.level_questions = questions
    st.session_state.level_pos = 0
    st.session_state.level_correct = 0
    st.session_state.answered = False
    go("quiz_play")


def page_quiz_play():
    m = st.session_state.model
    level = st.session_state.current_level
    set_no = st.session_state.current_set
    questions = st.session_state.level_questions
    pos = st.session_state.level_pos
    total = len(questions)

    if not questions or pos >= total:
        go("quiz_sets")
        return

    hero_banner(f"QUIZ — {LEVEL_NAMES.get(level, '')} · Bộ {set_no}", f"Câu {pos + 1}/{total}")
    topbar()
    st.progress((pos + (1 if st.session_state.answered else 0)) / total)

    question = questions[pos]
    with st.container(border=True):
        st.markdown(f"##### 🗺️ {question.get('question', 'Question')}")
        options = question.get("options", [])
        selected = st.radio("Chọn đáp án:", options, key=f"quiz_radio_{level}_{set_no}_{pos}",
                             index=None, disabled=st.session_state.answered)

        if not st.session_state.answered:
            if st.button("✅ Nộp câu trả lời", type="primary"):
                if selected is None:
                    st.warning("Hãy chọn một đáp án.")
                else:
                    selected_index = options.index(selected)
                    correct_answer = question.get("answer", question.get("correct_answer", 0))
                    is_correct = str(selected_index) == str(correct_answer)

                    m.record_quiz_answer(level, set_no, is_correct)
                    st.session_state.answered = True
                    st.session_state.last_correct = is_correct

                    if is_correct:
                        st.session_state.level_correct += 1
                        m.update_xp(30)
                        m.log_activity(f"Trả lời đúng 1 câu Quiz {LEVEL_NAMES.get(level)} - Bộ {set_no} (+30 XP).")
                    st.rerun()
        else:
            if st.session_state.last_correct:
                st.success(f"✓ Chính xác! {question.get('explanation', '')}  (+30 XP)")
            else:
                st.error(f"✗ Chưa chính xác. {question.get('explanation', '')}")

            label = "Câu tiếp theo →" if pos + 1 < total else "Xem kết quả bộ này →"
            if st.button(label, type="primary"):
                st.session_state.level_pos += 1
                st.session_state.answered = False
                if st.session_state.level_pos >= total:
                    go("quiz_result")
                else:
                    st.rerun()

    if st.button("← Chọn bộ khác"):
        go("quiz_sets")


def page_quiz_result():
    m = st.session_state.model
    level = st.session_state.current_level
    set_no = st.session_state.current_set
    total = len(st.session_state.level_questions)
    correct = st.session_state.level_correct

    level_just_completed = m.complete_set(level, set_no)

    m.update_xp(20)
    m.log_activity(f"Hoàn thành Bộ {set_no} — {LEVEL_NAMES.get(level)} (+20 XP).")

    bonus_xp = 0
    if level_just_completed:
        bonus_xp = 80
        m.update_xp(bonus_xp)
        m.log_activity(f"🎉 Hoàn thành TẤT CẢ các bộ của {LEVEL_NAMES.get(level)} (+{bonus_xp} XP).")

    hero_banner(f"HOÀN THÀNH BỘ {set_no}", LEVEL_NAMES.get(level, ""))
    topbar()

    set_ids = m.get_set_ids(level)
    lvl_progress = m.quiz_progress.get(str(level), {"sets": {}, "completed": False})
    sets_done = sum(1 for s in set_ids if lvl_progress["sets"].get(str(s), {}).get("completed", False))

    with st.container(border=True):
        st.markdown(f"### Kết quả bộ {set_no}: {correct}/{total} câu đúng")
        st.progress(correct / total if total else 0)
        st.caption(f"Tiến trình level: đã hoàn thành {sets_done}/{len(set_ids)} bộ")
        st.progress(sets_done / len(set_ids) if set_ids else 0)

        if level_just_completed:
            st.success(f"🎉 Bạn vừa hoàn thành TOÀN BỘ {LEVEL_NAMES.get(level)}! (+{bonus_xp} XP thưởng)")
            next_level = level + 1
            if next_level <= 5:
                st.info(f"🔓 Level {next_level} đã được mở khoá!")
            else:
                st.info("🏆 Bạn đã hoàn thành toàn bộ 5 level!")
        elif lvl_progress["completed"]:
            st.info("Bạn đã hoàn thành toàn bộ level này từ trước — có thể ôn lại bất kỳ bộ nào.")
        else:
            st.info(f"Hoàn thành thêm {len(set_ids) - sets_done} bộ nữa để mở khoá level tiếp theo.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📦 Các bộ khác trong level", use_container_width=True):
                go("quiz_sets")
        with c2:
            if st.button("📚 Xem tất cả level", use_container_width=True):
                go("quiz_levels")
        with c3:
            next_level = level + 1
            if next_level <= 5 and m.is_level_unlocked(next_level):
                if st.button(f"Chơi Level {next_level} →", type="primary", use_container_width=True):
                    st.session_state.current_level = next_level
                    go("quiz_sets")


# ============================================================
# 10. AI TUTOR
# ============================================================

def page_ai_tutor():
    hero_banner("AI TOURISM TUTOR", "Trợ lý AI về quản trị du lịch & kinh doanh")
    topbar()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else "🧑‍💼"):
            st.markdown(msg["content"])

    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🧭"):
            st.markdown("Xin chào! Bạn có thể hỏi tôi về quản trị du lịch, khách hàng, "
                        "nhân sự, marketing, tài chính hoặc xử lý khủng hoảng.")

    query = st.chat_input("Đặt câu hỏi cho AI Tutor...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🧭"):
            with st.spinner("Đang xử lý..."):
                if client is None:
                    answer = generate_offline_tutor_response(query)
                    error = "offline"
                else:
                    prompt = f"""
Bạn là AI Tutor chuyên về Quản trị Du lịch và Lữ hành.
Hãy trả lời câu hỏi của học sinh bằng tiếng Việt, rõ ràng, dễ hiểu, phù hợp học sinh THPT.
Nếu có thể, hãy đưa ví dụ thực tế trong ngành du lịch.

Câu hỏi:
{query}
"""
                    answer, error = call_gemini(prompt)
                    if answer is None:
                        answer = generate_offline_tutor_response(query)

            if error and error != "offline":
                st.warning(f"⚠️ Gemini không khả dụng: {error}\n\n→ Đang dùng Offline Tutor.")
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        m = st.session_state.model
        m.update_xp(10)
        m.log_activity("Sử dụng AI Tutor (+10 XP).")
        st.rerun()


# ============================================================
# 11. CRISIS MANAGEMENT
# ============================================================

def page_crisis():
    m = st.session_state.model
    hero_banner("CRISIS MANAGEMENT", "Xử lý tình huống khủng hoảng du lịch thực tế")
    topbar()

    if not m.crises:
        st.warning("Không tìm thấy crises.json.")
        return

    if st.session_state.current_crisis is None:
        load_random_crisis()

    crisis = st.session_state.current_crisis

    with st.container(border=True):
        st.markdown(f"### 🌪️ {crisis.get('title', 'Tourism Crisis')}")
        st.caption(f"📍 Topic: {crisis.get('topic', 'General')}  ·  🎯 Difficulty: {crisis.get('difficulty', 'Medium')}")
        st.markdown(crisis.get("description", ""))

        solution = st.text_area("💼 Quyết định của bạn:", height=150, key="crisis_solution_input")

        c1, c2 = st.columns([1, 1])
        with c1:
            evaluate_clicked = st.button("📊 Evaluate Solution", type="primary", use_container_width=True)
        with c2:
            if st.button("🔄 Đổi tình huống khác", use_container_width=True):
                load_random_crisis()
                st.session_state.crisis_result = None
                st.rerun()

        if evaluate_clicked:
            if not solution.strip():
                st.warning("Hãy nhập quyết định và cách giải quyết.")
            else:
                with st.spinner("Đang đánh giá..."):
                    evaluate_crisis_solution(solution, crisis)
                st.rerun()

    if st.session_state.crisis_result:
        render_crisis_result(st.session_state.crisis_result)


def load_random_crisis():
    m = st.session_state.model
    available = [c for c in m.crises if c not in m.used_crises]
    if not available:
        m.used_crises = []
        available = m.crises
    crisis = random.choice(available)
    m.used_crises.append(crisis)
    st.session_state.current_crisis = crisis
    st.session_state.crisis_result = None


def evaluate_crisis_solution(user_solution, crisis):
    evaluation, error = None, None

    if client is None:
        evaluation = generate_offline_crisis_evaluation(user_solution, crisis)
        error = "offline"
    else:
        prompt = f"""
Bạn là chuyên gia đánh giá năng lực quản trị trong ngành Du lịch và Lữ hành.
Hãy đánh giá quyết định của học sinh trong tình huống sau.

TÌNH HUỐNG:
{json.dumps(crisis, ensure_ascii=False, indent=2)}

CÂU TRẢ LỜI CỦA HỌC SINH:
{user_solution}

Hãy đánh giá 6 tiêu chí (số nguyên 0-10): decision_making, risk_management,
customer_service, financial_management, reputation_management, feasibility.

Trả về ĐÚNG JSON:
{{
    "decision_making": 0, "risk_management": 0, "customer_service": 0,
    "financial_management": 0, "reputation_management": 0, "feasibility": 0,
    "strengths": ["..."], "weaknesses": ["..."], "feedback": "..."
}}
Không thêm Markdown, không thêm ```json, không giải thích bên ngoài JSON.
"""
        raw_response, error = call_gemini(prompt)
        if raw_response:
            evaluation = extract_json_from_text(raw_response)
            if evaluation is None:
                error = "Gemini trả về dữ liệu không đúng JSON."
        if evaluation is None:
            evaluation = generate_offline_crisis_evaluation(user_solution, crisis)

    evaluation = normalize_evaluation(evaluation)
    m = st.session_state.model
    total_score = m.update_business_stats(evaluation)
    m.update_xp(100)
    m.log_activity(f"Hoàn thành Crisis Management (+100 XP, score {total_score}/10).")

    st.session_state.crisis_result = {
        "evaluation": evaluation, "error": error, "total_score": total_score,
    }


def render_crisis_result(result):
    evaluation = result["evaluation"]
    error = result["error"]
    total_score = result["total_score"]

    with st.container(border=True):
        if error and error != "offline":
            st.warning(f"⚠️ Gemini không khả dụng: {error}\n\nHệ thống đã chuyển sang Offline Evaluation.")
        elif error == "offline":
            st.info("ℹ️ Đang dùng bộ đánh giá Offline vì Gemini hiện không khả dụng.")

        st.markdown(f"### 🎯 Điểm tổng: {total_score}/10")
        st.progress(total_score / 10)

        labels = {
            "decision_making": ("Decision Making", "25%"),
            "risk_management": ("Risk Management", "20%"),
            "customer_service": ("Customer Service", "20%"),
            "financial_management": ("Financial Management", "15%"),
            "reputation_management": ("Reputation Management", "10%"),
            "feasibility": ("Feasibility", "10%"),
        }
        cols = st.columns(3)
        for i, (key, (label, weight)) in enumerate(labels.items()):
            with cols[i % 3]:
                st.metric(f"{label} ({weight})", f"{evaluation[key]}/10")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Điểm mạnh:**")
            for item in evaluation["strengths"]:
                st.markdown(f"- {item}")
        with c2:
            st.markdown("**⚠️ Điểm cần cải thiện:**")
            for item in evaluation["weaknesses"]:
                st.markdown(f"- {item}")

        st.markdown("**💬 Feedback:**")
        st.info(evaluation["feedback"])
        st.success("+100 XP")


# ============================================================
# 12. KHUNG HÀNH ĐỘNG DÙNG CHUNG CHO HR & BUDGET
# ============================================================

def action_card(name, cost, effect_text, stat_label, stat_value, budget, on_click_key, handler):
    afford = budget >= cost
    card_class = "biz-card" if afford else "biz-card locked"
    with st.container(border=False):
        st.markdown(f"""
        <div class="{card_class}">
            <b>{name}</b><br>
            <span style="color:{'#0B2545' if afford else '#DC2626'};font-size:13px;">
                💵 Chi phí: {format_vnd(cost)}{'' if afford else ' (không đủ ngân sách)'}
            </span><br>
            <span style="color:#6B7280;font-size:12px;">{effect_text}</span>
        </div>
        """, unsafe_allow_html=True)
        if stat_label:
            st.caption(f"{stat_label}: {round(stat_value)}/100")
            st.progress(round(stat_value) / 100)
        if st.button("Thực hiện", key=on_click_key, disabled=not afford,
                     type="primary", use_container_width=True):
            handler()
            st.rerun()


# ============================================================
# 13. VẬN HÀNH DOANH NGHIỆP — HR + BUDGET GỘP THÀNH 1 HỆ THỐNG
#     (Mọi quyết định đầu tư nguồn lực nằm chung 1 nơi, có chung
#     mục tiêu và chung "vòng giá trị" để người chơi thấy rõ
#     nhân sự và tài chính liên kết với nhau như thế nào.)
# ============================================================

def page_operations():
    m = st.session_state.model
    hero_banner("TRUNG TÂM VẬN HÀNH DOANH NGHIỆP", "Đầu tư Nhân sự & Tài chính, theo dõi thị trường theo từng lượt")
    topbar()

    # ---- Mục đích của hệ thống — trả lời "tạo ra để làm gì" ----
    st.markdown("""
    <div class="biz-card gold">
        <b>🎯 Mục đích của Trung tâm Vận hành</b><br>
        <span style="color:#374151;font-size:13px;line-height:1.6;">
        Đây là nơi bạn ra <b>quyết định đầu tư nguồn lực</b> cho doanh nghiệp du lịch của mình,
        rồi bấm <b>Kết thúc lượt</b> để bước sang tuần kinh doanh tiếp theo. Mỗi lượt, chỉ số thị
        trường du lịch (biến động như một biểu đồ chứng khoán) sẽ thay đổi, và một sự kiện ngẫu
        nhiên có thể xảy ra — ví dụ đoàn thanh tra bất chợt, đoàn khách lớn, thời tiết xấu, hay đối
        tác ngỏ lời hợp tác. Vượt qua sự kiện thành công sẽ mang lại tiền thưởng.
        </span><br><br>
        <span style="color:#0B2545;font-size:13px;font-weight:600;">
        🧑‍💼 Tuyển &amp; đào tạo nhân sự → 📢 Marketing &amp; nâng cấp cơ sở → 🔚 Kết thúc lượt →
        📈 Thị trường biến động + 🎲 Sự kiện ngẫu nhiên → 💰 Doanh thu &amp; hệ quả → 🔁 Tái đầu tư
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Tổng quan tài nguyên dùng chung cho các nhóm quyết định ----
    with st.container(border=True):
        st.markdown("#### 📋 Tổng quan tài nguyên hiện có")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💵 Ngân sách", format_vnd(m.budget))
        c2.metric("👥 Nhân sự", m.staff_count)
        c3.metric("😊 Hài lòng NV", f"{round(m.stats['employee_satisfaction'])}/100")
        c4.metric("📊 Sức khoẻ TC", f"{round(m.stats['financial_health'])}/100")
        c5.metric("📈 Chỉ số thị trường", f"{m.market_index:.1f}")

    tab_hr, tab_budget, tab_market = st.tabs([
        "🧑‍💼 NHÂN SỰ — Hire & Train",
        "💰 TÀI CHÍNH — Marketing & Nâng cấp",
        "📈 THỊ TRƯỜNG — Kết thúc lượt",
    ])

    with tab_hr:
        st.caption("Đầu tư vào con người: nhân sự nhiều & giỏi hơn → phục vụ khách tốt hơn.")
        col1, col2 = st.columns(2)
        with col1:
            action_card(
                "👤 Hire Staff — Tuyển thêm nhân viên", 5_000_000,
                "Tăng 1 nhân viên và +3 Employee Satisfaction. Nhân sự càng nhiều, doanh thu tự động "
                "mỗi lượt càng cao.",
                "Employee Satisfaction", m.stats["employee_satisfaction"], m.budget,
                "hr_hire", hire_staff,
            )
        with col2:
            action_card(
                "🎓 Train Staff — Đào tạo nhân viên", 8_000_000,
                "Tăng +8 Employee Satisfaction và +5 Customer Satisfaction. Nâng cao chất lượng "
                "phục vụ lâu dài, không cần tuyển thêm người.",
                "Customer Satisfaction", m.stats["customer_satisfaction"], m.budget,
                "hr_train", train_staff,
            )

    with tab_budget:
        st.caption("Đầu tư tài chính: marketing & cơ sở vật chất → uy tín và độ bền vững tăng → dễ "
                    "vượt qua các sự kiện ngẫu nhiên hơn (ví dụ thanh tra bất chợt).")
        col1, col2 = st.columns(2)
        with col1:
            action_card(
                "📢 Marketing Campaign", 10_000_000,
                "Tăng +7 Business Reputation và +5 Customer Satisfaction. Thu hút thêm khách hàng mới.",
                "Business Reputation", m.stats["business_reputation"], m.budget,
                "budget_marketing", marketing_campaign,
            )
        with col2:
            action_card(
                "🏗️ Facility Upgrade", 15_000_000,
                "Tăng +10 Sustainability và +8 Customer Satisfaction. Cải thiện cơ sở vật chất lâu dài.",
                "Sustainability", m.stats["sustainability"], m.budget,
                "budget_facility", facility_upgrade,
            )
        st.info("💡 Doanh thu không còn thu thủ công nữa — sang tab **Thị trường**, mỗi lần "
                "**Kết thúc lượt** doanh thu sẽ tự động được cộng dựa theo số nhân sự và chỉ số thị trường.")

    with tab_market:
        st.caption("Chỉ số thị trường du lịch biến động qua từng lượt, giống một biểu đồ chứng khoán. "
                    "Kết thúc lượt để thu doanh thu tự động và đối mặt với một sự kiện ngẫu nhiên.")

        prev_index = m.market_history[-2] if len(m.market_history) >= 2 else m.market_history[-1]
        delta = m.market_index - prev_index
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("📈 Chỉ số thị trường", f"{m.market_index:.1f}", delta=f"{delta:+.1f}")
            st.metric("🔁 Lượt hiện tại", m.turn)
        with c2:
            st.line_chart(m.market_history, height=200)

        if st.button("🔚 Kết thúc lượt (Sang tuần mới)", type="primary", use_container_width=True):
            event, outcome, revenue = advance_turn()
            st.session_state.last_turn_result = {"event": event, "outcome": outcome, "revenue": revenue}
            st.rerun()

        result = st.session_state.last_turn_result
        if result:
            event, outcome, revenue = result["event"], result["outcome"], result["revenue"]
            with st.container(border=True):
                st.markdown(f"#### {event['icon']} {event['name']} — Lượt {m.turn}")
                if outcome["success"]:
                    st.success(outcome["message"])
                else:
                    st.error(outcome["message"])
                st.markdown(f"💰 Doanh thu tự động lượt này: **{format_vnd(revenue)}** "
                            f"(chỉ số thị trường {m.market_index:.1f})")


# ---- Sự kiện ngẫu nhiên trong Trung tâm Vận hành ----

def event_inspection(m):
    score = (m.stats["sustainability"] + m.stats["business_reputation"]) / 2
    if score >= 65:
        reward = random.randint(2, 4) * 1_000_000
        return {"success": True, "budget_delta": reward, "stat_deltas": {"business_reputation": 2},
                "message": f"Đoàn thanh tra đánh giá tốt về cơ sở vật chất & uy tín, thưởng {format_vnd(reward)}."}
    fine = random.randint(4, 8) * 1_000_000
    return {"success": False, "budget_delta": -fine, "stat_deltas": {"business_reputation": -5},
            "message": f"Đoàn thanh tra bất chợt phát hiện thiếu sót, doanh nghiệp bị phạt {format_vnd(fine)}."}


def event_big_group(m):
    if m.stats["customer_satisfaction"] >= 60:
        bonus = random.randint(3, 8) * 1_000_000
        return {"success": True, "budget_delta": bonus, "stat_deltas": {"customer_satisfaction": 2},
                "message": f"Một đoàn khách lớn rất hài lòng với dịch vụ, mang lại {format_vnd(bonus)}."}
    loss = random.randint(1, 2) * 1_000_000
    return {"success": False, "budget_delta": -loss, "stat_deltas": {"customer_satisfaction": -3},
            "message": f"Đoàn khách lớn huỷ tour vì dịch vụ chưa tốt, thiệt hại {format_vnd(loss)}."}


def event_bad_weather(m):
    loss = random.randint(2, 6) * 1_000_000
    return {"success": False, "budget_delta": -loss, "stat_deltas": {},
            "message": f"Thời tiết xấu khiến một số tour bị huỷ, thiệt hại {format_vnd(loss)}."}


def event_market_boom(m):
    return {"success": True, "budget_delta": 0, "stat_deltas": {"business_reputation": 1},
            "message": "Thị trường du lịch tăng trưởng nóng, chỉ số thị trường được đẩy lên rõ rệt."}


def event_market_crisis(m):
    loss = random.randint(1, 3) * 1_000_000
    return {"success": False, "budget_delta": -loss, "stat_deltas": {},
            "message": f"Biến động kinh tế khiến doanh thu tạm thời giảm {format_vnd(loss)}."}


def event_partner_offer(m):
    if m.stats["business_reputation"] >= 70:
        bonus = random.randint(6, 12) * 1_000_000
        return {"success": True, "budget_delta": bonus, "stat_deltas": {},
                "message": f"Ký kết hợp tác thành công với một đối tác chiến lược, nhận {format_vnd(bonus)}."}
    return {"success": False, "budget_delta": 0, "stat_deltas": {},
            "message": "Một đối tác ngỏ lời hợp tác nhưng từ chối vì uy tín doanh nghiệp chưa đủ cao."}


OPERATIONS_EVENTS = [
    {"icon": "🕵️", "name": "Thanh tra bất chợt", "handler": event_inspection},
    {"icon": "👨‍👩‍👧‍👦", "name": "Đoàn khách lớn đặt tour", "handler": event_big_group},
    {"icon": "🌧️", "name": "Thời tiết xấu", "handler": event_bad_weather},
    {"icon": "📈", "name": "Thị trường tăng trưởng nóng", "handler": event_market_boom},
    {"icon": "📉", "name": "Biến động kinh tế", "handler": event_market_crisis},
    {"icon": "🤝", "name": "Đối tác ngỏ lời hợp tác", "handler": event_partner_offer},
]


def advance_turn():
    """Vòng lặp chính của Trung tâm Vận hành: mỗi lượt thị trường biến động
    như chứng khoán, 1 sự kiện ngẫu nhiên xảy ra, và doanh thu được tự động
    thu về dựa trên số nhân sự và chỉ số thị trường."""
    m = st.session_state.model
    m.turn += 1

    baseline = random.uniform(-6, 6)
    reputation_bonus = (m.stats["business_reputation"] - 50) / 50 * 2
    sustain_bonus = (m.stats["sustainability"] - 50) / 50 * 2
    change_pct = baseline + reputation_bonus + sustain_bonus

    event = random.choice(OPERATIONS_EVENTS)
    outcome = event["handler"](m)

    if event["name"] == "Thị trường tăng trưởng nóng":
        change_pct += random.uniform(8, 18)
    elif event["name"] == "Biến động kinh tế":
        change_pct -= random.uniform(8, 18)

    m.market_index = max(10.0, round(m.market_index * (1 + change_pct / 100), 1))
    m.market_history.append(m.market_index)
    if len(m.market_history) > 30:
        m.market_history = m.market_history[-30:]

    m.budget = max(0, m.budget + outcome["budget_delta"])
    for key, delta in outcome.get("stat_deltas", {}).items():
        m.stats[key] = round(max(0, min(100, m.stats[key] + delta)), 1)

    base_revenue = m.staff_count * 2_500_000
    revenue = round(base_revenue * (m.market_index / 100))
    m.budget += revenue
    m.stats["financial_health"] = round(min(100, m.stats["financial_health"] + 1), 1)

    m.log_activity(f"[Lượt {m.turn}] {event['icon']} {event['name']}: {outcome['message']}")
    m.log_activity(f"[Lượt {m.turn}] 💰 Doanh thu tự động: {format_vnd(revenue)} (chỉ số thị trường {m.market_index}).")
    m.update_xp(15)
    m.save_progress()

    return event, outcome, revenue


def hire_staff():
    m = st.session_state.model
    m.budget -= 5_000_000
    m.staff_count += 1
    m.stats["employee_satisfaction"] = min(100, m.stats["employee_satisfaction"] + 3)
    m.log_activity("Tuyển thêm 1 nhân viên.")
    st.toast("Đã tuyển thêm 1 nhân viên! Staff +1, Employee Satisfaction +3", icon="👤")


def train_staff():
    m = st.session_state.model
    m.budget -= 8_000_000
    m.stats["employee_satisfaction"] = min(100, m.stats["employee_satisfaction"] + 8)
    m.stats["customer_satisfaction"] = min(100, m.stats["customer_satisfaction"] + 5)
    m.log_activity("Đào tạo nhân viên.")
    st.toast("Đã đào tạo nhân viên! Employee +8, Customer +5", icon="🎓")


def marketing_campaign():
    m = st.session_state.model
    m.budget -= 10_000_000
    m.stats["business_reputation"] = min(100, m.stats["business_reputation"] + 7)
    m.stats["customer_satisfaction"] = min(100, m.stats["customer_satisfaction"] + 5)
    m.log_activity("Thực hiện chiến dịch Marketing.")
    st.toast("Marketing thành công! Reputation +7, Customer +5", icon="📢")


def facility_upgrade():
    m = st.session_state.model
    m.budget -= 15_000_000
    m.stats["sustainability"] = min(100, m.stats["sustainability"] + 10)
    m.stats["customer_satisfaction"] = min(100, m.stats["customer_satisfaction"] + 8)
    m.log_activity("Nâng cấp cơ sở vật chất.")
    st.toast("Nâng cấp thành công! Sustainability +10, Customer +8", icon="🏗️")


# ============================================================
# 15. LỊCH SỬ
# ============================================================

def page_history():
    m = st.session_state.model
    hero_banner("LỊCH SỬ HOẠT ĐỘNG", "Nhật ký quyết định kinh doanh")
    topbar()

    with st.container(border=True):
        if not m.history_log:
            st.info("Chưa có hoạt động nào.")
        else:
            for i, item in enumerate(reversed(m.history_log), 1):
                st.markdown(f"**{i}.** {item}")


# ============================================================
# 16. MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="AI Tourism Business Simulator",
        page_icon="🧭",
        layout="wide",
    )
    inject_theme()
    init_state()
    sidebar_nav()

    page = st.session_state.page
    if page == "home":
        page_home()
    elif page == "quiz_levels":
        page_quiz_levels()
    elif page == "quiz_sets":
        page_quiz_sets()
    elif page == "quiz_play":
        page_quiz_play()
    elif page == "quiz_result":
        page_quiz_result()
    elif page == "ai_tutor":
        page_ai_tutor()
    elif page == "crisis":
        page_crisis()
    elif page == "operations":
        page_operations()
    elif page == "history":
        page_history()
    else:
        page_home()


if __name__ == "__main__":
    main()
