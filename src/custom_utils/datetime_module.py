from datetime import datetime


def now_str() -> str:
    """現在の日時を "YYYYMMDD_HHMMSS" 形式の文字列で取得"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")
