import json
import math
import os
import random
import threading
import time

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


# Giữ nguyên model hiện tại của project
GEMINI_MODEL = get_secret(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

GEMINI_API_KEY = get_secret(
    "GEMINI_API_KEY",
    ""
).strip()


# Gemini không được phép làm UI chờ vô hạn
GEMINI_TIMEOUT = 15

# Khi gặp 429, tạm ngừng gọi Gemini trong khoảng thời gian này.
# Trong thời gian đó hệ thống dùng Offline Mode.
GEMINI_COOLDOWN_SECONDS = 60

_gemini_blocked_until = 0.0


LEVEL_NAMES = {
    1: "Level 1 · Cơ bản",
    2: "Level 2 · Sơ cấp",
    3: "Level 3 · Trung cấp",
    4: "Level 4 · Nâng cao",
    5: "Level 5 · Chuyên gia",
}

LEVEL_ICONS = {
    1: "🔰",
    2: "⭐",
    3: "🏆",
    4: "🚀",
    5: "👑",
}


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None

    if genai is None:
        return None

    try:
        return genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception as e:
        print("Không thể khởi tạo Gemini:", e)
        return None


client = get_gemini_client()


# ============================================================
# 3. GEMINI HELPER
# ============================================================

def gemini_is_in_cooldown():
    global _gemini_blocked_until

    return time.monotonic() < _gemini_blocked_until


def set_gemini_cooldown():
    global _gemini_blocked_until

    _gemini_blocked_until = (
        time.monotonic() + GEMINI_COOLDOWN_SECONDS
    )


def remaining_gemini_cooldown():
    if not gemini_is_in_cooldown():
        return 0

    return max(
        0,
        int(_gemini_blocked_until - time.monotonic())
    )


def _gemini_worker(prompt, result):
    """
    Hàm chạy Gemini trong thread riêng.

    Thread daemon giúp Streamlit không phải chờ vô hạn
    nếu API bị treo hoặc mạng có vấn đề.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        text = getattr(response, "text", None)

        if not text:
            result["error"] = "Gemini không trả về nội dung."
            return

        result["text"] = text.strip()

    except Exception as e:
        result["error"] = str(e)


def call_gemini(prompt, timeout=GEMINI_TIMEOUT):
    """
    Gọi Gemini an toàn.

    - Có timeout.
    - Không retry khi 429.
    - 429 -> Offline Mode.
    - Timeout -> Offline Mode.
    - Lỗi API -> Offline Mode.
    """

    if client is None:
        return None, "Gemini chưa được khởi tạo."

    if gemini_is_in_cooldown():
        seconds = remaining_gemini_cooldown()

        return (
            None,
            f"Gemini đang tạm nghỉ do quota 429. "
            f"Hệ thống sẽ dùng Offline Mode trong khoảng {seconds} giây."
        )

    result = {
        "text": None,
        "error": None
    }

    thread = threading.Thread(
        target=_gemini_worker,
        args=(prompt, result),
        daemon=True
    )

    thread.start()
    thread.join(timeout)

    # API chạy quá lâu
    if thread.is_alive():
        return (
            None,
            f"Gemini phản hồi quá {timeout} giây. "
            f"Hệ thống đã chuyển sang Offline Mode."
        )

    if result["error"]:
        error_text = result["error"]

        error_lower = error_text.lower()

        # -----------------------------
        # 429 / QUOTA
        # -----------------------------
        if (
            "429" in error_text
            or "resource_exhausted" in error_lower
            or "quota" in error_lower
            or "rate limit" in error_lower
        ):
            set_gemini_cooldown()

            return (
                None,
                "Gemini đang hết quota "
                "(429 RESOURCE_EXHAUSTED). "
                "Hệ thống đã chuyển sang Offline Mode."
            )

        # -----------------------------
        # AUTH
        # -----------------------------
        if (
            "401" in error_text
            or "unauthenticated" in error_lower
            or "authentication" in error_lower
            or "api key" in error_lower
        ):
            return (
                None,
                "Gemini gặp lỗi xác thực API key. "
                "Hệ thống đã chuyển sang Offline Mode."
            )

        # -----------------------------
        # 403
        # -----------------------------
        if "403" in error_text:
            return (
                None,
                "Gemini từ chối quyền truy cập API. "
                "Hệ thống đã chuyển sang Offline Mode."
            )

        return None, f"Lỗi Gemini: {error_text}"

    if not result["text"]:
        return None, "Gemini không trả về nội dung."

    return result["text"], None


# ============================================================
# 4. JSON PARSER
# ============================================================

def extract_json_from_text(text):
    if not text:
        return None

    text = text.strip()

    # Trường hợp Gemini trả:
    # ```json
    # {...}
    # ```
    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Thử JSON nguyên bản
    try:
        return json.loads(text)
    except Exception:
        pass

    # Tìm phần {...}
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            pass

    return None


def safe_score(value, default=5):
    try:
        value = float(value)

        return round(
            max(0, min(10, value)),
            1
        )

    except (ValueError, TypeError):
        return float(default)


# ============================================================
# 5. OFFLINE AI TUTOR
# ============================================================

def generate_offline_tutor_response(user_query):

    query = user_query.lower()

    if any(
        w in query
        for w in [
            "giá",
            "pricing",
            "price",
            "tăng giá",
            "giảm giá"
        ]
    ):
        return (
            "### 💡 Gợi ý Offline\n\n"
            "Khi quyết định giá trong du lịch, nên cân nhắc:\n\n"
            "1. Chi phí vận hành.\n"
            "2. Nhu cầu khách hàng.\n"
            "3. Giá của đối thủ cạnh tranh.\n"
            "4. Giá trị mà khách hàng nhận được.\n"
            "5. Mùa cao điểm và thấp điểm.\n\n"
            "Ví dụ: resort có thể áp dụng giá cao hơn vào mùa cao điểm "
            "và chương trình ưu đãi vào mùa thấp điểm."
        )

    if any(
        w in query
        for w in [
            "khách",
            "customer",
            "phàn nàn",
            "khiếu nại",
            "dịch vụ",
            "service"
        ]
    ):
        return (
            "### 💡 Gợi ý Offline\n\n"
            "Khi xử lý khách hàng không hài lòng, có thể đi theo quy trình:\n\n"
            "1. Lắng nghe khách hàng.\n"
            "2. Xác định nguyên nhân.\n"
            "3. Đưa ra giải pháp phù hợp.\n"
            "4. Xem xét bồi thường nếu cần.\n"
            "5. Theo dõi mức độ hài lòng sau xử lý.\n\n"
            "Mục tiêu không chỉ là giải quyết khiếu nại mà còn "
            "giữ được niềm tin của khách hàng."
        )

    if any(
        w in query
        for w in [
            "khủng hoảng",
            "crisis",
            "cháy",
            "tai nạn",
            "rủi ro",
            "nguy hiểm"
        ]
    ):
        return (
            "### 💡 Gợi ý Offline\n\n"
            "Trong khủng hoảng du lịch, ưu tiên đầu tiên là "
            "an toàn của khách hàng và nhân viên.\n\n"
            "Sau đó:\n"
            "1. Xác minh thông tin.\n"
            "2. Kiểm soát tình hình.\n"
            "3. Giao tiếp minh bạch.\n"
            "4. Phối hợp với các bên liên quan.\n"
            "5. Đưa ra phương án phục hồi."
        )

    if any(
        w in query
        for w in [
            "môi trường",
            "xanh",
            "bền vững",
            "sustainability",
            "rác",
            "eco"
        ]
    ):
        return (
            "### 💡 Gợi ý Offline\n\n"
            "Du lịch bền vững cần cân bằng giữa:\n\n"
            "- Lợi ích kinh tế.\n"
            "- Trải nghiệm khách hàng.\n"
            "- Bảo vệ môi trường.\n"
            "- Lợi ích của cộng đồng địa phương.\n\n"
            "Một số giải pháp: giảm nhựa dùng một lần, "
            "tiết kiệm năng lượng, giảm chất thải và hợp tác "
            "với cộng đồng địa phương."
        )

    if any(
        w in query
        for w in [
            "nhân viên",
            "nhân sự",
            "employee",
            "staff",
            "hr",
            "đào tạo",
            "tuyển"
        ]
    ):
        return (
            "### 💡 Gợi ý Offline\n\n"
            "Quản trị nhân sự trong du lịch thường chú ý đến:\n\n"
            "1. Tuyển dụng.\n"
            "2. Đào tạo.\n"
            "3. Phân công công việc.\n"
            "4. Đánh giá hiệu quả.\n"
            "5. Tạo động lực.\n"
            "6. Giữ chân nhân viên.\n\n"
            "Chất lượng nhân sự có thể ảnh hưởng trực tiếp "
            "đến chất lượng trải nghiệm khách hàng."
        )

    return (
        "### 💡 Gợi ý Offline\n\n"
        "Hãy thử phân tích vấn đề theo 4 bước:\n\n"
        "1. **Xác định vấn đề.**\n"
        "2. **Xác định nguyên nhân.**\n"
        "3. **So sánh các phương án.**\n"
        "4. **Dự đoán hậu quả trước khi quyết định.**\n\n"
        "Nếu có nhiều bên liên quan, hãy cân nhắc lợi ích "
        "của khách hàng, nhân viên, doanh nghiệp và cộng đồng."
    )


# ============================================================
# 6. OFFLINE CRISIS EVALUATION
# ============================================================

def generate_offline_crisis_evaluation(
    user_solution,
    crisis
):
    solution = user_solution.lower()

    scores = {
        "decision_making": 5,
        "risk_management": 5,
        "customer_service": 5,
        "financial_management": 5,
        "reputation_management": 5,
        "feasibility": 5,
    }

    strengths = []
    weaknesses = []

    checks = [

        (
            "decision_making",
            [
                "ưu tiên",
                "quyết định",
                "giải quyết",
                "phương án",
                "kế hoạch",
                "xử lý"
            ],
            "Bạn đã thể hiện khả năng đưa ra quyết định.",
            "Quyết định chưa được trình bày đủ rõ ràng.",
            True
        ),

        (
            "risk_management",
            [
                "an toàn",
                "rủi ro",
                "kiểm tra",
                "phòng",
                "bảo vệ",
                "khẩn cấp"
            ],
            "Bạn có chú ý đến quản trị rủi ro và an toàn.",
            "Cần phân tích rủi ro và phương án dự phòng rõ hơn.",
            True
        ),

        (
            "customer_service",
            [
                "khách",
                "bồi thường",
                "xin lỗi",
                "hỗ trợ",
                "hoàn tiền",
                "đổi",
                "chăm sóc"
            ],
            "Bạn có quan tâm đến trải nghiệm khách hàng.",
            "Cần nói rõ hơn cách bảo vệ quyền lợi khách hàng.",
            True
        ),

        (
            "financial_management",
            [
                "chi phí",
                "ngân sách",
                "tiền",
                "doanh thu",
                "lợi nhuận",
                "tiết kiệm",
                "chi"
            ],
            "Bạn đã cân nhắc yếu tố tài chính.",
            "Chưa đề cập nhiều đến tác động tài chính.",
            False
        ),

        (
            "reputation_management",
            [
                "uy tín",
                "thương hiệu",
                "truyền thông",
                "minh bạch",
                "thông báo",
                "mạng xã hội",
                "danh tiếng"
            ],
            "Bạn có chú ý đến uy tín và hình ảnh doanh nghiệp.",
            "Nên có kế hoạch truyền thông để bảo vệ uy tín.",
            False
        ),

        (
            "feasibility",
            [
                "nhân viên",
                "thời gian",
                "nguồn lực",
                "có thể",
                "thực hiện",
                "phối hợp",
                "đối tác"
            ],
            "Giải pháp có xét đến khả năng triển khai thực tế.",
            "Nên làm rõ nguồn lực và cách triển khai.",
            False
        ),
    ]

    for (
        key,
        keywords,
        good_msg,
        bad_msg,
        penalize
    ) in checks:

        if any(
            word in solution
            for word in keywords
        ):
            scores[key] += 2
            strengths.append(good_msg)

        else:
            if penalize:
                scores[key] -= 1

            weaknesses.append(bad_msg)

    for key in scores:
        scores[key] = safe_score(
            scores[key]
        )

    if not strengths:
        strengths.append(
            "Bạn đã đưa ra một hướng xử lý ban đầu."
        )

    if not weaknesses:
        weaknesses.append(
            "Có thể bổ sung thêm phương án dự phòng."
        )

    feedback = (
        "Đây là đánh giá Offline vì Gemini hiện không khả dụng. "
        "Khi Gemini hoạt động, hệ thống sẽ đánh giá câu trả lời "
        "theo ngữ cảnh cụ thể của tình huống."
    )

    return {
        **scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "feedback": feedback
    }


# ============================================================
# 7. MODEL GAME
# ============================================================

class TourismGameModel:

    CRISIS_WEIGHTS = {
        "decision_making": 0.25,
        "risk_management": 0.20,
        "customer_service": 0.20,
        "financial_management": 0.15,
        "reputation_management": 0.10,
        "feasibility": 0.10,
    }

    def __init__(self):

        self.xp = 0

        self.level_name = (
            "Tourism Beginner"
        )

        self.budget = 5000
        self.staff_count = 5

        self.stats = {
            "customer_satisfaction": 75,
            "business_reputation": 70,
            "employee_satisfaction": 65,
            "financial_health": 72,
            "sustainability": 80,
        }

        self.history_log = []

        self.used_crises = []

        self.quiz_progress = {
            str(i): {
                "correct": 0,
                "total": 0,
                "completed": False
            }
            for i in range(1, 6)
        }

        self.quiz_data = self.load_json(
            QUIZ_FILE,
            []
        )

        self.crises = self.load_json(
            CRISIS_FILE,
            []
        )

        self.quiz_levels = (
            self.build_quiz_levels()
        )

        self.load_progress()

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def load_json(path, default):

        try:

            if not os.path.exists(path):
                return default

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception as e:

            print(
                f"Lỗi đọc {path}: {e}"
            )

            return default

    # ========================================================
    # QUIZ LEVEL
    # ========================================================

    def build_quiz_levels(self):

        levels = {
            i: []
            for i in range(1, 6)
        }

        if not self.quiz_data:
            return levels

        has_level_field = any(
            isinstance(q, dict)
            and "level" in q
            for q in self.quiz_data
        )

        if has_level_field:

            for q in self.quiz_data:

                try:
                    lvl = int(
                        q.get("level", 1)
                    )

                except (
                    ValueError,
                    TypeError
                ):
                    lvl = 1

                lvl = max(
                    1,
                    min(5, lvl)
                )

                levels[lvl].append(q)

        else:

            n = len(
                self.quiz_data
            )

            chunk = max(
                1,
                math.ceil(n / 5)
            )

            for index, q in enumerate(
                self.quiz_data
            ):

                lvl = min(
                    5,
                    index // chunk + 1
                )

                levels[lvl].append(q)

        return levels

    def is_level_unlocked(self, level):

        if level == 1:
            return True

        prev = self.quiz_progress.get(
            str(level - 1),
            {}
        )

        return prev.get(
            "completed",
            False
        )

    def record_quiz_answer(
        self,
        level,
        correct
    ):

        key = str(level)

        prog = self.quiz_progress.setdefault(
            key,
            {
                "correct": 0,
                "total": 0,
                "completed": False
            }
        )

        prog["total"] += 1

        if correct:
            prog["correct"] += 1

        self.save_progress()

    def complete_level(self, level):

        key = str(level)

        self.quiz_progress.setdefault(
            key,
            {
                "correct": 0,
                "total": 0,
                "completed": False
            }
        )

        self.quiz_progress[key][
            "completed"
        ] = True

        self.save_progress()

    # ========================================================
    # SAVE / LOAD
    # ========================================================

    def save_progress(self):

        data = {
            "xp": self.xp,
            "level_name": self.level_name,
            "budget": self.budget,
            "staff_count": self.staff_count,
            "stats": self.stats,
            "history_log": self.history_log,
            "quiz_progress": self.quiz_progress,
        }

        try:

            with open(
                SAVE_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=4
                )

        except Exception as e:

            print(
                "Lỗi lưu game:",
                e
            )

    def load_progress(self):

        if not os.path.exists(
            SAVE_FILE
        ):
            return

        try:

            with open(
                SAVE_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            self.xp = data.get(
                "xp",
                self.xp
            )

            self.level_name = data.get(
                "level_name",
                self.level_name
            )

            self.budget = data.get(
                "budget",
                self.budget
            )

            self.staff_count = data.get(
                "staff_count",
                self.staff_count
            )

            saved_stats = data.get(
                "stats",
                {}
            )

            for key in self.stats:

                if key in saved_stats:
                    self.stats[key] = saved_stats[key]

            self.history_log = data.get(
                "history_log",
                []
            )

            saved_quiz_progress = data.get(
                "quiz_progress",
                {}
            )

            for key in self.quiz_progress:

                if key in saved_quiz_progress:

                    self.quiz_progress[key].update(
                        saved_quiz_progress[key]
                    )

            self.update_level_only()

        except Exception as e:

            print(
                "Lỗi load save:",
                e
            )

    # ========================================================
    # LEVEL
    # ========================================================

    def update_level_only(self):

        if self.xp >= 1500:

            self.level_name = (
                "Tourism Expert"
            )

        elif self.xp >= 1000:

            self.level_name = (
                "Tourism Strategist"
            )

        elif self.xp >= 600:

            self.level_name = (
                "Tourism Manager"
            )

        elif self.xp >= 300:

            self.level_name = (
                "Tourism Planner"
            )

        elif self.xp >= 100:

            self.level_name = (
                "Tourism Trainee"
            )

        else:

            self.level_name = (
                "Tourism Beginner"
            )

    def update_xp(self, amount):

        self.xp = max(
            0,
            self.xp + amount
        )

        old_level = self.level_name

        self.update_level_only()

        self.save_progress()

        return (
            old_level != self.level_name
        )

    def log_activity(self, text):

        self.history_log.append(
            text
        )

        self.save_progress()

    # ========================================================
    # CRISIS SCORE
    # ========================================================

    def calculate_crisis_score(
        self,
        evaluation
    ):

        total = 0

        for (
            criterion,
            weight
        ) in self.CRISIS_WEIGHTS.items():

            total += (
                safe_score(
                    evaluation.get(
                        criterion,
                        5
                    )
                )
                * weight
            )

        return round(
            total,
            2
        )

    def update_business_stats(
        self,
        evaluation
    ):

        mapping = {
            "customer_service":
                "customer_satisfaction",

            "reputation_management":
                "business_reputation",

            "financial_management":
                "financial_health",

            "risk_management":
                "sustainability",

            "decision_making":
                "employee_satisfaction",
        }

        total_score = (
            self.calculate_crisis_score(
                evaluation
            )
        )

        for (
            criterion,
            stat_name
        ) in mapping.items():

            score = safe_score(
                evaluation.get(
                    criterion,
                    5
                )
            )

            delta = (
                score - 5
            ) * 2

            current = (
                self.stats.get(
                    stat_name,
                    50
                )
                + delta
            )

            self.stats[stat_name] = round(
                max(
                    0,
                    min(
                        100,
                        current
                    )
                ),
                1
            )

        self.log_activity(
            f"Quản lý khủng hoảng: "
            f"điểm tổng {total_score}/10"
        )

        return total_score


# ============================================================
# 8. NORMALIZE AI EVALUATION
# ============================================================

def normalize_evaluation(
    evaluation
):

    if not isinstance(
        evaluation,
        dict
    ):
        evaluation = {}

    result = {}

    criteria = [
        "decision_making",
        "risk_management",
        "customer_service",
        "financial_management",
        "reputation_management",
        "feasibility",
    ]

    for criterion in criteria:

        result[criterion] = safe_score(
            evaluation.get(
                criterion,
                5
            )
        )

    strengths = evaluation.get(
        "strengths",
        []
    )

    weaknesses = evaluation.get(
        "weaknesses",
        []
    )

    feedback = evaluation.get(
        "feedback",
        ""
    )

    if isinstance(
        strengths,
        str
    ):
        strengths = [strengths]

    elif not isinstance(
        strengths,
        list
    ):
        strengths = []

    if isinstance(
        weaknesses,
        str
    ):
        weaknesses = [weaknesses]

    elif not isinstance(
        weaknesses,
        list
    ):
        weaknesses = []

    if not isinstance(
        feedback,
        str
    ):
        feedback = str(
            feedback
        )

    result["strengths"] = strengths
    result["weaknesses"] = weaknesses
    result["feedback"] = feedback

    return result


# ============================================================
# 9. QUIZ ANSWER CHECK
# ============================================================

def check_quiz_answer(
    selected,
    options,
    correct_answer
):
    """
    Hỗ trợ nhiều dạng answer trong quiz_data.json:

    "answer": 0
    "answer": "0"

    "answer": "A"
    "answer": "B"

    "answer": "A. ..."
    "answer": "B. ..."

    hoặc chính nội dung đáp án.
    """

    if selected is None:
        return False

    # --------------------------------------------------------
    # 1. Nếu answer là số
    # --------------------------------------------------------

    if isinstance(
        correct_answer,
        int
    ):
        return (
            options.index(selected)
            == correct_answer
        )

    correct_text = str(
        correct_answer
    ).strip()

    # --------------------------------------------------------
    # 2. Nếu answer là "0", "1", "2",...
    # --------------------------------------------------------

    if correct_text.isdigit():

        try:

            correct_index = int(
                correct_text
            )

            return (
                options.index(selected)
                == correct_index
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # 3. Nếu answer là A/B/C/D/E
    # --------------------------------------------------------

    letter = correct_text.upper()

    if len(letter) == 1 and letter in "ABCDE":

        correct_index = (
            ord(letter) - ord("A")
        )

        return (
            options.index(selected)
            == correct_index
        )

    # --------------------------------------------------------
    # 4. Nếu answer dạng "B. Nội dung"
    # --------------------------------------------------------

    if (
        len(correct_text) >= 2
        and correct_text[0].upper() in "ABCDE"
        and correct_text[1] in ".):"
    ):

        correct_index = (
            ord(
                correct_text[0].upper()
            )
            - ord("A")
        )

        return (
            options.index(selected)
            == correct_index
        )

    # --------------------------------------------------------
    # 5. So sánh trực tiếp nội dung
    # --------------------------------------------------------

    return (
        selected.strip().lower()
        == correct_text.lower()
    )


# ============================================================
# 10. THEME
# ============================================================

def inject_theme():

    st.markdown(
        """
        <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap'
        );

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
            linear-gradient(
                180deg,
                #F8F3E7 0%,
                #F4EFE1 100%
            );
        }

        section[data-testid="stSidebar"] {
            background:
            linear-gradient(
                180deg,
                #0B2545 0%,
                #13315C 100%
            );
        }

        section[data-testid="stSidebar"] * {
            color: #F1EAD6 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color:
            rgba(241,234,214,0.2);
        }

        h1, h2, h3 {
            font-family:
            'Poppins', sans-serif !important;

            color:
            #0B2545 !important;
        }

        .hero-banner {
            background:
            linear-gradient(
                120deg,
                #0B2545 0%,
                #13315C 55%,
                #0F766E 100%
            );

            border-radius: 18px;

            padding:
            28px 32px;

            margin-bottom:
            18px;

            box-shadow:
            0 8px 24px
            rgba(11,37,69,0.25);

            position:
            relative;

            overflow:
            hidden;
        }

        .hero-banner::after {
            content:
            "🌍";

            position:
            absolute;

            right:
            18px;

            top:
            8px;

            font-size:
            90px;

            opacity:
            0.12;
        }

        .hero-title {
            font-family:
            'Poppins',
            sans-serif;

            color:
            #FDF6E3;

            font-size:
            30px;

            font-weight:
            700;

            margin:
            0;
        }

        .hero-subtitle {
            color:
            #D4A017;

            font-size:
            14px;

            letter-spacing:
            1px;

            text-transform:
            uppercase;

            margin-top:
            4px;
        }

        .topbar {
            display:
            flex;

            gap:
            12px;

            margin-bottom:
            18px;
        }

        .topbar-item {
            flex:
            1;

            background:
            #FFFFFF;

            border-radius:
            12px;

            padding:
            10px 14px;

            border-left:
            4px solid #D4A017;

            box-shadow:
            0 2px 8px
            rgba(11,37,69,0.08);
        }

        .topbar-label {
            font-size:
            11px;

            color:
            #6B7280;

            text-transform:
            uppercase;

            letter-spacing:
            0.5px;
        }

        .topbar-value {
            font-size:
            18px;

            font-weight:
            700;

            color:
            #0B2545;

            font-family:
            'Poppins',
            sans-serif;
        }

        .biz-card {
            background:
            #FFFFFF;

            border-radius:
            14px;

            padding:
            18px 20px;

            border-left:
            5px solid #0F766E;

            box-shadow:
            0 3px 12px
            rgba(11,37,69,0.08);

            margin-bottom:
            16px;
        }

        .biz-card.gold {
            border-left-color:
            #D4A017;
        }

        .biz-card.locked {
            border-left-color:
            #9CA3AF;

            opacity:
            0.65;
        }

        .badge {
            display:
            inline-block;

            padding:
            3px 10px;

            border-radius:
            999px;

            font-size:
            11px;

            font-weight:
            600;

            letter-spacing:
            0.3px;
        }

        .badge-done {
            background:
            #DCFCE7;

            color:
            #15803D;
        }

        .badge-progress {
            background:
            #FEF3C7;

            color:
            #B45309;
        }

        .badge-locked {
            background:
            #E5E7EB;

            color:
            #6B7280;
        }

        .stButton > button {
            border-radius:
            10px;

            font-weight:
            600;

            border:
            none;
        }

        .stButton > button[kind="primary"] {
            background:
            #0B2545;

            color:
            #FDF6E3;
        }

        .stat-caption {
            color:
            #6B7280;

            font-size:
            12px;

            margin-bottom:
            -6px;
        }

        [data-testid="stAppViewContainer"]
        [data-testid="stMarkdownContainer"],

        [data-testid="stAppViewContainer"]
        [data-testid="stMetricLabel"],

        [data-testid="stAppViewContainer"]
        [data-testid="stMetricValue"],

        [data-testid="stAppViewContainer"]
        [data-testid="stMetricDelta"] {

            color:
            #1F2933 !important;
        }

        [data-testid="stAppViewContainer"]
        [data-testid="stCaptionContainer"] {

            color:
            #4B5563 !important;
        }

        [data-testid="stAppViewContainer"]
        [data-testid="stWidgetLabel"] p {

            color:
            #1F2933 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def hero_banner(
    title,
    subtitle
):

    st.markdown(
        f"""
        <div class="hero-banner">

            <div class="hero-title">
                🧭 {title}
            </div>

            <div class="hero-subtitle">
                🗺️ ĐỊA LÝ
                &nbsp;·&nbsp;
                💼 KINH DOANH
                &nbsp;·&nbsp;
                ✈️ DU LỊCH
                — {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def topbar():

    m = st.session_state.model

    items = [
        ("⭐ XP", m.xp),
        ("💼 Level", m.level_name),
        ("💰 Budget", f"${m.budget}"),
        ("👥 Staff", m.staff_count),
    ]

    html = (
        '<div class="topbar">'
    )

    for label, value in items:

        html += f"""
        <div class="topbar-item">

            <div class="topbar-label">
                {label}
            </div>

            <div class="topbar-value">
                {value}
            </div>

        </div>
        """

    html += "</div>"

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def stat_row(stats_dict):

    labels = {
        "customer_satisfaction":
            "🧳 Customer",

        "business_reputation":
            "🏛️ Reputation",

        "employee_satisfaction":
            "🧑‍💼 Employee",

        "financial_health":
            "📊 Financial",

        "sustainability":
            "🌱 Sustainability",
    }

    cols = st.columns(
        len(labels)
    )

    for col, (
        key,
        label
    ) in zip(
        cols,
        labels.items()
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat-caption">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                int(
                    round(
                        stats_dict[key]
                    )
                ) / 100
            )

            st.caption(
                f"{round(stats_dict[key])}/100"
            )


# ============================================================
# 11. SESSION STATE
# ============================================================

def init_state():

    if "model" not in st.session_state:
        st.session_state.model = (
            TourismGameModel()
        )

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if "current_level" not in st.session_state:
        st.session_state.current_level = None

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


def go(page):

    st.session_state.page = page

    st.rerun()


# ============================================================
# 12. SIDEBAR
# ============================================================

def sidebar_nav():

    m = st.session_state.model

    with st.sidebar:

        st.markdown(
            "### 🧭 AI Tourism Business"
        )

        st.caption(
            "Simulator — Địa lý & Kinh doanh du lịch"
        )

        st.markdown("---")

        nav_items = [
            ("home", "🏠 Trang chủ"),
            ("quiz_levels", "📚 Quiz (5 Level)"),
            ("ai_tutor", "🤝 AI Tutor"),
            ("crisis", "🌪️ Crisis Management"),
            ("hr", "🧑‍💼 HR Management"),
            ("budget", "💰 Budget Allocation"),
            ("history", "📜 Lịch sử hoạt động"),
        ]

        current_page = st.session_state.page

        if current_page in (
            "quiz_play",
            "quiz_result"
        ):
            current_top = "quiz_levels"

        else:
            current_top = current_page

        for key, label in nav_items:

            is_active = (
                key == current_top
            )

            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type=(
                    "primary"
                    if is_active
                    else "secondary"
                )
            ):

                if key == "home":
                    st.session_state.current_crisis = None

                go(key)

        st.markdown("---")

        st.caption(
            f"Gemini model: `{GEMINI_MODEL}`"
        )

        if client:

            if gemini_is_in_cooldown():

                seconds = (
                    remaining_gemini_cooldown()
                )

                st.caption(
                    f"🟡 Gemini cooldown "
                    f"({seconds}s)"
                )

            else:

                st.caption(
                    "🟢 Đã kết nối Gemini"
                )

        else:

            st.caption(
                "🟡 Chế độ Offline"
            )

        if st.button(
            "💾 Save Game",
            use_container_width=True
        ):

            m.save_progress()

            st.toast(
                "Đã lưu tiến trình!",
                icon="💾"
            )


# ============================================================
# 13. HOME
# ============================================================

def page_home():

    m = st.session_state.model

    hero_banner(
        "AI TOURISM BUSINESS SIMULATOR",
        "Bảng điều khiển doanh nghiệp"
    )

    topbar()

    st.markdown(
        "#### 📊 Chỉ số doanh nghiệp"
    )

    with st.container(
        border=True
    ):

        stat_row(
            m.stats
        )

    st.markdown(
        "#### 🗂️ Trung tâm điều hành"
    )

    cols = st.columns(3)

    cards = [

        (
            "📚 Quiz 5 Level",
            "Ôn tập kiến thức du lịch & kinh doanh theo cấp độ.",
            "quiz_levels"
        ),

        (
            "🤝 AI Tutor",
            "Hỏi đáp cùng trợ lý AI về quản trị du lịch.",
            "ai_tutor"
        ),

        (
            "🌪️ Crisis Management",
            "Xử lý tình huống khủng hoảng thực tế.",
            "crisis"
        ),

        (
            "🧑‍💼 HR Management",
            "Tuyển dụng & đào tạo nhân sự.",
            "hr"
        ),

        (
            "💰 Budget Allocation",
            "Phân bổ ngân sách kinh doanh.",
            "budget"
        ),

        (
            "📜 Lịch sử",
            "Xem lại các quyết định đã thực hiện.",
            "history"
        ),
    ]

    for i, (
        title,
        desc,
        page_key
    ) in enumerate(cards):

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="biz-card gold">

                    <b>{title}</b><br>

                    <span style="
                        color:#6B7280;
                        font-size:13px;
                    ">
                        {desc}
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Mở →",
                key=f"home_open_{page_key}",
                use_container_width=True
            ):

                go(page_key)

    with st.expander(
        "🎲 Sự kiện ngẫu nhiên (Random Event)"
    ):

        if st.button(
            "Kích hoạt sự kiện"
        ):

            trigger_random_event()


def trigger_random_event():

    m = st.session_state.model

    events = [

        {
            "text":
                "Một travel blogger nổi tiếng đăng bài tích cực về doanh nghiệp.",
            "budget_change":
                -300,
            "reputation_change":
                5
        },

        {
            "text":
                "Một khách hàng đăng bài phàn nàn trên mạng xã hội.",
            "budget_change":
                -500,
            "reputation_change":
                -2
        },

        {
            "text":
                "Một đối tác địa phương đề xuất hợp tác quảng bá.",
            "budget_change":
                800,
            "reputation_change":
                10
        },
    ]

    event = random.choice(
        events
    )

    m.budget = max(
        0,
        m.budget
        + event["budget_change"]
    )

    m.stats[
        "business_reputation"
    ] = max(
        0,
        min(
            100,
            m.stats[
                "business_reputation"
            ]
            + event["reputation_change"]
        )
    )

    m.log_activity(
        f"Random Event: {event['text']}"
    )

    st.success(
        f"{event['text']}  | "
        f"Ngân sách {event['budget_change']:+}$  | "
        f"Uy tín {event['reputation_change']:+}"
    )


# ============================================================
# 14. QUIZ LEVEL
# ============================================================

def page_quiz_levels():

    m = st.session_state.model

    hero_banner(
        "QUIZ NGHIỆP VỤ",
        "Hoàn thành level để mở khoá level tiếp theo"
    )

    topbar()

    if not m.quiz_data:

        st.warning(
            "Không tìm thấy quiz_data.json."
        )

        return

    cols = st.columns(5)

    for level in range(1, 6):

        questions = (
            m.quiz_levels.get(
                level,
                []
            )
        )

        progress = (
            m.quiz_progress.get(
                str(level),
                {
                    "correct": 0,
                    "total": 0,
                    "completed": False
                }
            )
        )

        unlocked = (
            m.is_level_unlocked(
                level
            )
        )

        if progress["completed"]:

            badge = (
                '<span class="badge badge-done">'
                '✓ Hoàn thành'
                '</span>'
            )

        elif progress["total"] > 0:

            badge = (
                '<span class="badge badge-progress">'
                'Đang làm'
                '</span>'
            )

        elif not unlocked:

            badge = (
                '<span class="badge badge-locked">'
                '🔒 Đã khoá'
                '</span>'
            )

        else:

            badge = (
                '<span class="badge badge-locked">'
                'Chưa bắt đầu'
                '</span>'
            )

        card_class = (
            "biz-card"
            if unlocked
            else "biz-card locked"
        )

        with cols[level - 1]:

            st.markdown(
                f"""
                <div class="{card_class}"
                     style="text-align:center;">

                    <div style="
                        font-size:26px;
                    ">
                        {
                            LEVEL_ICONS.get(
                                level,
                                "📘"
                            )
                            if unlocked
                            else "🔒"
                        }
                    </div>

                    <b>
                        {LEVEL_NAMES.get(level)}
                    </b><br>

                    <span style="
                        color:#6B7280;
                        font-size:12px;
                    ">
                        {len(questions)} câu hỏi
                    </span><br>

                    {badge}<br><br>

                    <span style="
                        font-size:11px;
                        color:#6B7280;
                    ">
                        {progress['correct']}/
                        {progress['total']} đúng
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            disabled = (
                not unlocked
                or not questions
            )

            if st.button(
                "Bắt đầu",
                key=f"start_lvl_{level}",
                use_container_width=True,
                disabled=disabled
            ):

                start_quiz_level(
                    level
                )


def start_quiz_level(level):

    m = st.session_state.model

    questions = list(
        m.quiz_levels.get(
            level,
            []
        )
    )

    random.shuffle(
        questions
    )

    st.session_state.current_level = level
    st.session_state.level_questions = questions
    st.session_state.level_pos = 0
    st.session_state.level_correct = 0
    st.session_state.answered = False
    st.session_state.last_correct = None

    go("quiz_play")


# ============================================================
# 15. QUIZ PLAY
# ============================================================

def page_quiz_play():

    m = st.session_state.model

    level = (
        st.session_state.current_level
    )

    questions = (
        st.session_state.level_questions
    )

    pos = (
        st.session_state.level_pos
    )

    total = len(
        questions
    )

    if (
        not questions
        or pos >= total
    ):

        go("quiz_levels")

        return

    hero_banner(
        f"QUIZ — {LEVEL_NAMES.get(level, '')}",
        f"Câu {pos + 1}/{total}"
    )

    topbar()

    st.progress(
        (
            pos
            + (
                1
                if st.session_state.answered
                else 0
            )
        )
        / total
    )

    question = questions[pos]

    with st.container(
        border=True
    ):

        st.markdown(
            f"""
            ##### 🗺️
            {question.get(
                'question',
                'Question'
            )}
            """
        )

        options = question.get(
            "options",
            []
        )

        selected = st.radio(
            "Chọn đáp án:",
            options,
            key=f"quiz_radio_{level}_{pos}",
            index=None,
            disabled=(
                st.session_state.answered
            )
        )

        if not st.session_state.answered:

            if st.button(
                "✅ Nộp câu trả lời",
                type="primary"
            ):

                if selected is None:

                    st.warning(
                        "Hãy chọn một đáp án."
                    )

                else:

                    correct_answer = (
                        question.get(
                            "answer",
                            question.get(
                                "correct_answer",
                                0
                            )
                        )
                    )

                    is_correct = (
                        check_quiz_answer(
                            selected,
                            options,
                            correct_answer
                        )
                    )

                    m.record_quiz_answer(
                        level,
                        is_correct
                    )

                    st.session_state.answered = True

                    st.session_state.last_correct = (
                        is_correct
                    )

                    if is_correct:

                        st.session_state.level_correct += 1

                        m.update_xp(
                            30
                        )

                        m.log_activity(
                            f"Trả lời đúng 1 câu "
                            f"Quiz Level {level} "
                            f"(+30 XP)."
                        )

                    st.rerun()

        else:

            explanation = question.get(
                "explanation",
                "Chưa có giải thích."
            )

            if st.session_state.last_correct:

                st.success(
                    f"✓ Chính xác!\n\n"
                    f"{explanation}\n\n"
                    f"(+30 XP)"
                )

            else:

                st.error(
                    f"✗ Chưa chính xác.\n\n"
                    f"{explanation}"
                )

            # Nếu JSON có thêm phần giải thích
            # các đáp án khác
            wrong_explanations = question.get(
                "wrong_explanations",
                question.get(
                    "why_wrong",
                    None
                )
            )

            if wrong_explanations:

                with st.expander(
                    "📖 Vì sao các đáp án khác chưa phù hợp?"
                ):

                    if isinstance(
                        wrong_explanations,
                        dict
                    ):

                        for key, value in wrong_explanations.items():

                            st.markdown(
                                f"**{key}:** {value}"
                            )

                    else:

                        st.markdown(
                            str(
                                wrong_explanations
                            )
                        )

            label = (
                "Câu tiếp theo →"
                if pos + 1 < total
                else
                "Xem kết quả level →"
            )

            if st.button(
                label,
                type="primary"
            ):

                st.session_state.level_pos += 1

                st.session_state.answered = False

                st.session_state.last_correct = None

                if (
                    st.session_state.level_pos
                    >= total
                ):

                    go(
                        "quiz_result"
                    )

                else:

                    st.rerun()

    if st.button(
        "← Chọn level khác"
    ):

        go(
            "quiz_levels"
        )


# ============================================================
# 16. QUIZ RESULT
# ============================================================

def page_quiz_result():

    m = st.session_state.model

    level = (
        st.session_state.current_level
    )

    total = len(
        st.session_state.level_questions
    )

    correct = (
        st.session_state.level_correct
    )

    was_completed = (
        m.quiz_progress
        .get(
            str(level),
            {}
        )
        .get(
            "completed",
            False
        )
    )

    m.complete_level(
        level
    )

    bonus_xp = 0

    if not was_completed:

        bonus_xp = 50

        m.update_xp(
            bonus_xp
        )

        m.log_activity(
            f"Hoàn thành Quiz "
            f"{LEVEL_NAMES.get(level)} "
            f"(+{bonus_xp} XP)."
        )

    hero_banner(
        "HOÀN THÀNH LEVEL",
        LEVEL_NAMES.get(level, "")
    )

    topbar()

    with st.container(
        border=True
    ):

        st.markdown(
            f"""
            ### Kết quả:
            {correct}/{total} câu đúng
            """
        )

        st.progress(
            correct / total
            if total
            else 0
        )

        if bonus_xp:

            st.success(
                f"🎉 Lần đầu hoàn thành "
                f"level này: +{bonus_xp} XP thưởng!"
            )

        next_level = level + 1

        if next_level <= 5:

            st.info(
                f"Level {next_level} đã được mở khoá!"
                if not was_completed
                else
                f"Bạn đã mở khoá "
                f"Level {next_level} từ trước."
            )

        else:

            st.info(
                "🏆 Bạn đã hoàn thành "
                "toàn bộ 5 level!"
            )

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "📚 Xem tất cả level",
                use_container_width=True
            ):

                go(
                    "quiz_levels"
                )

        with c2:

            if (
                next_level <= 5
                and m.is_level_unlocked(
                    next_level
                )
            ):

                if st.button(
                    f"Chơi Level {next_level} →",
                    type="primary",
                    use_container_width=True
                ):

                    start_quiz_level(
                        next_level
                    )


# ============================================================
# 17. AI TUTOR
# ============================================================

def page_ai_tutor():

    hero_banner(
        "AI TOURISM TUTOR",
        "Trợ lý AI về quản trị du lịch & kinh doanh"
    )

    topbar()

    for msg in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            msg["role"],
            avatar=(
                "🧭"
                if msg["role"] == "assistant"
                else
                "🧑‍💼"
            )
        ):

            st.markdown(
                msg["content"]
            )

    if not st.session_state.chat_history:

        with st.chat_message(
            "assistant",
            avatar="🧭"
        ):

            st.markdown(
                "Xin chào! Bạn có thể hỏi tôi về "
                "quản trị du lịch, khách hàng, "
                "nhân sự, marketing, tài chính "
                "hoặc xử lý khủng hoảng."
            )

    query = st.chat_input(
        "Đặt câu hỏi cho AI Tutor..."
    )

    if query:

        st.session_state.chat_history.append(
            {
                "role":
                    "user",
                "content":
                    query
            }
        )

        with st.chat_message(
            "user",
            avatar="🧑‍💼"
        ):

            st.markdown(
                query
            )

        with st.chat_message(
            "assistant",
            avatar="🧭"
        ):

            with st.spinner(
                "Đang xử lý..."
            ):

                if client is None:

                    answer = (
                        generate_offline_tutor_response(
                            query
                        )
                    )

                    error = "offline"

                else:

                    prompt = f"""
Bạn là AI Tutor chuyên về
Quản trị Du lịch và Lữ hành.

Hãy trả lời câu hỏi của học sinh
bằng tiếng Việt.

Yêu cầu:
- Rõ ràng.
- Dễ hiểu.
- Phù hợp học sinh THPT.
- Không giải thích quá hàn lâm.
- Nếu phù hợp, đưa ví dụ thực tế trong ngành du lịch.
- Có thể đưa ra một câu hỏi nhỏ để học sinh tự kiểm tra.

CÂU HỎI:
{query}
"""

                    answer, error = call_gemini(
                        prompt
                    )

                    if answer is None:

                        answer = (
                            generate_offline_tutor_response(
                                query
                            )
                        )

                if (
                    error
                    and error != "offline"
                ):

                    st.warning(
                        f"⚠️ Gemini không khả dụng: "
                        f"{error}\n\n"
                        "→ Đang dùng Offline Tutor."
                    )

                st.markdown(
                    answer
                )

        st.session_state.chat_history.append(
            {
                "role":
                    "assistant",
                "content":
                    answer
            }
        )

        m = st.session_state.model

        m.update_xp(
            10
        )

        m.log_activity(
            "Sử dụng AI Tutor (+10 XP)."
        )

        st.rerun()


# ============================================================
# 18. CRISIS MANAGEMENT
# ============================================================

def page_crisis():

    m = st.session_state.model

    hero_banner(
        "CRISIS MANAGEMENT",
        "Xử lý tình huống khủng hoảng du lịch thực tế"
    )

    topbar()

    if not m.crises:

        st.warning(
            "Không tìm thấy crises.json."
        )

        return

    if (
        st.session_state.current_crisis
        is None
    ):

        load_random_crisis()

    crisis = (
        st.session_state.current_crisis
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f"""
            ### 🌪️
            {crisis.get(
                'title',
                'Tourism Crisis'
            )}
            """
        )

        st.caption(
            f"📍 Topic: "
            f"{crisis.get(
                'topic',
                'General'
            )}"
            f"  ·  🎯 Difficulty: "
            f"{crisis.get(
                'difficulty',
                'Medium'
            )}"
        )

        st.markdown(
            crisis.get(
                "description",
                ""
            )
        )

        solution = st.text_area(
            "💼 Quyết định của bạn:",
            height=150,
            key="crisis_solution_input"
        )

        c1, c2 = st.columns(
            [1, 1]
        )

        with c1:

            evaluate_clicked = st.button(
                "📊 Evaluate Solution",
                type="primary",
                use_container_width=True
            )

        with c2:

            if st.button(
                "🔄 Đổi tình huống khác",
                use_container_width=True
            ):

                load_random_crisis()

                st.session_state.crisis_result = None

                st.rerun()

        if evaluate_clicked:

            if not solution.strip():

                st.warning(
                    "Hãy nhập quyết định "
                    "và cách giải quyết."
                )

            else:

                with st.spinner(
                    "Đang đánh giá..."
                ):

                    evaluate_crisis_solution(
                        solution,
                        crisis
                    )

                st.rerun()

    if st.session_state.crisis_result:

        render_crisis_result(
            st.session_state.crisis_result
        )


def load_random_crisis():

    m = st.session_state.model

    available = [
        c
        for c in m.crises
        if c not in m.used_crises
    ]

    if not available:

        m.used_crises = []

        available = m.crises

    crisis = random.choice(
        available
    )

    m.used_crises.append(
        crisis
    )

    st.session_state.current_crisis = (
        crisis
    )

    st.session_state.crisis_result = None


# ============================================================
# 19. CRISIS EVALUATION
# ============================================================

def evaluate_crisis_solution(
    user_solution,
    crisis
):

    evaluation = None
    error = None

    if client is None:

        evaluation = (
            generate_offline_crisis_evaluation(
                user_solution,
                crisis
            )
        )

        error = "offline"

    else:

        prompt = f"""
Bạn là chuyên gia đánh giá năng lực
quản trị trong ngành Du lịch và Lữ hành.

Hãy đánh giá quyết định của học sinh
trong tình huống sau.

TÌNH HUỐNG:
{json.dumps(
    crisis,
    ensure_ascii=False,
    indent=2
)}

CÂU TRẢ LỜI CỦA HỌC SINH:
{user_solution}

Hãy đánh giá 6 tiêu chí
theo thang điểm 0-10:

1. decision_making
2. risk_management
3. customer_service
4. financial_management
5. reputation_management
6. feasibility

Đánh giá dựa trên nội dung thực tế
của câu trả lời và bối cảnh tình huống.

Trả về ĐÚNG JSON:

{{
    "decision_making": 0,
    "risk_management": 0,
    "customer_service": 0,
    "financial_management": 0,
    "reputation_management": 0,
    "feasibility": 0,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "feedback": "..."
}}

Không thêm Markdown.
Không thêm ```json.
Không giải thích bên ngoài JSON.
"""

        raw_response, error = (
            call_gemini(
                prompt
            )
        )

        if raw_response:

            evaluation = (
                extract_json_from_text(
                    raw_response
                )
            )

            if evaluation is None:

                error = (
                    "Gemini trả về dữ liệu "
                    "không đúng JSON."
                )

        if evaluation is None:

            evaluation = (
                generate_offline_crisis_evaluation(
                    user_solution,
                    crisis
                )
            )

    evaluation = normalize_evaluation(
        evaluation
    )

    m = st.session_state.model

    total_score = (
        m.update_business_stats(
            evaluation
        )
    )

    m.update_xp(
        100
    )

    m.log_activity(
        f"Hoàn thành Crisis Management "
        f"(+100 XP, score {total_score}/10)."
    )

    st.session_state.crisis_result = {
        "evaluation":
            evaluation,

        "error":
            error,

        "total_score":
            total_score,
    }


# ============================================================
# 20. CRISIS RESULT
# ============================================================

def render_crisis_result(
    result
):

    evaluation = result[
        "evaluation"
    ]

    error = result[
        "error"
    ]

    total_score = result[
        "total_score"
    ]

    with st.container(
        border=True
    ):

        if (
            error
            and error != "offline"
        ):

            st.warning(
                f"⚠️ Gemini không khả dụng: "
                f"{error}\n\n"
                "Hệ thống đã chuyển sang "
                "Offline Evaluation."
            )

        elif error == "offline":

            st.info(
                "ℹ️ Đang dùng bộ đánh giá "
                "Offline vì Gemini hiện "
                "không khả dụng."
            )

        st.markdown(
            f"""
            ### 🎯 Điểm tổng:
            {total_score}/10
            """
        )

        st.progress(
            total_score / 10
        )

        labels = {

            "decision_making":
                (
                    "Decision Making",
                    "25%"
                ),

            "risk_management":
                (
                    "Risk Management",
                    "20%"
                ),

            "customer_service":
                (
                    "Customer Service",
                    "20%"
                ),

            "financial_management":
                (
                    "Financial Management",
                    "15%"
                ),

            "reputation_management":
                (
                    "Reputation Management",
                    "10%"
                ),

            "feasibility":
                (
                    "Feasibility",
                    "10%"
                ),
        }

        cols = st.columns(3)

        for i, (
            key,
            (
                label,
                weight
            )
        ) in enumerate(
            labels.items()
        ):

            with cols[i % 3]:

                st.metric(
                    f"{label} ({weight})",
                    f"{evaluation[key]}/10"
                )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "**✅ Điểm mạnh:**"
            )

            for item in (
                evaluation["strengths"]
            ):

                st.markdown(
                    f"- {item}"
                )

        with c2:

            st.markdown(
                "**⚠️ Điểm cần cải thiện:**"
            )

            for item in (
                evaluation["weaknesses"]
            ):

                st.markdown(
                    f"- {item}"
                )

        st.markdown(
            "**💬 Feedback:**"
        )

        st.info(
            evaluation["feedback"]
        )

        st.success(
            "+100 XP"
        )


# ============================================================
# 21. ACTION CARD
# ============================================================

def action_card(
    name,
    cost,
    effect_text,
    stat_label,
    stat_value,
    budget,
    on_click_key,
    handler
):

    afford = (
        budget >= cost
    )

    card_class = (
        "biz-card"
        if afford
        else
        "biz-card locked"
    )

    with st.container(
        border=False
    ):

        st.markdown(
            f"""
            <div class="{card_class}">

                <b>{name}</b><br>

                <span style="
                    color:{
                        '#0B2545'
                        if afford
                        else
                        '#DC2626'
                    };
                    font-size:13px;
                ">

                    💵 Chi phí:
                    ${cost}

                    {
                        ''
                        if afford
                        else
                        ' (không đủ ngân sách)'
                    }

                </span><br>

                <span style="
                    color:#6B7280;
                    font-size:12px;
                ">

                    {effect_text}

                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        if stat_label:

            st.caption(
                f"{stat_label}: "
                f"{round(stat_value)}/100"
            )

            st.progress(
                round(stat_value) / 100
            )

        if st.button(
            "Thực hiện",
            key=on_click_key,
            disabled=not afford,
            type="primary",
            use_container_width=True
        ):

            handler()

            st.rerun()


# ============================================================
# 22. HR
# ============================================================

def page_hr():

    m = st.session_state.model

    hero_banner(
        "HR MANAGEMENT",
        "Quản trị nhân sự doanh nghiệp du lịch"
    )

    topbar()

    with st.container(
        border=True
    ):

        st.markdown(
            "#### 🧑‍💼 Tình trạng nhân sự"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "👥 Staff",
            m.staff_count
        )

        c2.metric(
            "😊 Employee Satisfaction",
            f"{round(
                m.stats['employee_satisfaction']
            )}/100"
        )

    col1, col2 = st.columns(2)

    with col1:

        action_card(
            "👤 Hire Staff — Tuyển thêm nhân viên",
            500,

            "Tăng 1 nhân viên và "
            "+3 Employee Satisfaction. "
            "Giúp tăng năng lực phục vụ khách hàng.",

            "Employee Satisfaction",

            m.stats[
                "employee_satisfaction"
            ],

            m.budget,

            "hr_hire",

            hire_staff
        )

    with col2:

        action_card(
            "🎓 Train Staff — Đào tạo nhân viên",
            800,

            "Tăng +8 Employee Satisfaction "
            "và +5 Customer Satisfaction. "
            "Nâng cao chất lượng phục vụ lâu dài.",

            "Customer Satisfaction",

            m.stats[
                "customer_satisfaction"
            ],

            m.budget,

            "hr_train",

            train_staff
        )


def hire_staff():

    m = st.session_state.model

    m.budget -= 500

    m.staff_count += 1

    m.stats[
        "employee_satisfaction"
    ] = min(
        100,
        m.stats[
            "employee_satisfaction"
        ] + 3
    )

    m.log_activity(
        "Tuyển thêm 1 nhân viên."
    )

    st.toast(
        "Đã tuyển thêm 1 nhân viên! "
        "Staff +1, Employee Satisfaction +3",
        icon="👤"
    )


def train_staff():

    m = st.session_state.model

    m.budget -= 800

    m.stats[
        "employee_satisfaction"
    ] = min(
        100,
        m.stats[
            "employee_satisfaction"
        ] + 8
    )

    m.stats[
        "customer_satisfaction"
    ] = min(
        100,
        m.stats[
            "customer_satisfaction"
        ] + 5
    )

    m.log_activity(
        "Đào tạo nhân viên."
    )

    st.toast(
        "Đã đào tạo nhân viên! "
        "Employee +8, Customer +5",
        icon="🎓"
    )


# ============================================================
# 23. BUDGET
# ============================================================

def page_budget():

    m = st.session_state.model

    hero_banner(
        "BUDGET ALLOCATION",
        "Phân bổ ngân sách kinh doanh du lịch"
    )

    topbar()

    with st.container(
        border=True
    ):

        st.markdown(
            "#### 💰 Tình trạng tài chính"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "💵 Budget",
            f"${m.budget}"
        )

        c2.metric(
            "🏛️ Reputation",
            f"{round(
                m.stats['business_reputation']
            )}/100"
        )

        c3.metric(
            "📊 Financial Health",
            f"{round(
                m.stats['financial_health']
            )}/100"
        )

    revenue_estimate = (
        m.staff_count * 250
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        action_card(
            "📢 Marketing Campaign",
            1000,

            "Tăng +7 Business Reputation "
            "và +5 Customer Satisfaction. "
            "Thu hút thêm khách hàng mới.",

            "Business Reputation",

            m.stats[
                "business_reputation"
            ],

            m.budget,

            "budget_marketing",

            marketing_campaign
        )

    with col2:

        action_card(
            "🏗️ Facility Upgrade",
            1500,

            "Tăng +10 Sustainability "
            "và +8 Customer Satisfaction. "
            "Cải thiện cơ sở vật chất lâu dài.",

            "Sustainability",

            m.stats[
                "sustainability"
            ],

            m.budget,

            "budget_facility",

            facility_upgrade
        )

    with col3:

        action_card(
            "💰 Collect Revenue",
            0,

            f"Thu về ước tính "
            f"${revenue_estimate} "
            f"(${250} x {m.staff_count} nhân viên) "
            f"và +3 Financial Health. "
            f"Không tốn chi phí.",

            "Financial Health",

            m.stats[
                "financial_health"
            ],

            m.budget,

            "budget_revenue",

            collect_revenue
        )


def marketing_campaign():

    m = st.session_state.model

    m.budget -= 1000

    m.stats[
        "business_reputation"
    ] = min(
        100,
        m.stats[
            "business_reputation"
        ] + 7
    )

    m.stats[
        "customer_satisfaction"
    ] = min(
        100,
        m.stats[
            "customer_satisfaction"
        ] + 5
    )

    m.log_activity(
        "Thực hiện chiến dịch Marketing."
    )

    st.toast(
        "Marketing thành công! "
        "Reputation +7, Customer +5",
        icon="📢"
    )


def facility_upgrade():

    m = st.session_state.model

    m.budget -= 1500

    m.stats[
        "sustainability"
    ] = min(
        100,
        m.stats[
            "sustainability"
        ] + 10
    )

    m.stats[
        "customer_satisfaction"
    ] = min(
        100,
        m.stats[
            "customer_satisfaction"
        ] + 8
    )

    m.log_activity(
        "Nâng cấp cơ sở vật chất."
    )

    st.toast(
        "Nâng cấp thành công! "
        "Sustainability +10, Customer +8",
        icon="🏗️"
    )


def collect_revenue():

    m = st.session_state.model

    revenue = (
        m.staff_count * 250
    )

    m.budget += revenue

    m.stats[
        "financial_health"
    ] = min(
        100,
        m.stats[
            "financial_health"
        ] + 3
    )

    m.log_activity(
        f"Thu doanh thu ${revenue}."
    )

    st.toast(
        f"Đã thu ${revenue} doanh thu! "
        "Financial Health +3",
        icon="💰"
    )


# ============================================================
# 24. HISTORY
# ============================================================

def page_history():

    m = st.session_state.model

    hero_banner(
        "LỊCH SỬ HOẠT ĐỘNG",
        "Nhật ký quyết định kinh doanh"
    )

    topbar()

    with st.container(
        border=True
    ):

        if not m.history_log:

            st.info(
                "Chưa có hoạt động nào."
            )

        else:

            for i, item in enumerate(
                reversed(
                    m.history_log
                ),
                1
            ):

                st.markdown(
                    f"**{i}.** {item}"
                )


# ============================================================
# 25. MAIN
# ============================================================

def main():

    st.set_page_config(
        page_title=
            "AI Tourism Business Simulator",

        page_icon=
            "🧭",

        layout=
            "wide"
    )

    inject_theme()

    init_state()

    sidebar_nav()

    page = (
        st.session_state.page
    )

    if page == "home":

        page_home()

    elif page == "quiz_levels":

        page_quiz_levels()

    elif page == "quiz_play":

        page_quiz_play()

    elif page == "quiz_result":

        page_quiz_result()

    elif page == "ai_tutor":

        page_ai_tutor()

    elif page == "crisis":

        page_crisis()

    elif page == "hr":

        page_hr()

    elif page == "budget":

        page_budget()

    elif page == "history":

        page_history()

    else:

        page_home()


if __name__ == "__main__":
    main()
