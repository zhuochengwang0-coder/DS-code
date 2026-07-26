"""
Steam《Elden Ring》评论采集脚本（可直接在 PyCharm 点击 Run 运行）

首次运行前，在 PyCharm Terminal 执行：
    pip install requests

采样规则：
    - Steam AppID: 1245620
    - 最新的英文评论（filter=recent, language=english）
    - 推荐与不推荐评论全部保留
    - Steam 购买与非 Steam 购买评论全部保留
    - 目标为 20,000 条唯一评论；若接口没有更多数据，则保存已取得的最大数量

输出文件位于本脚本同级目录的 steam_review_data 文件夹。
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# =========================
# 可在 PyCharm 中直接修改的配置
# =========================
GAME = "Elden Ring"
APPID = "1245620"
TARGET_REVIEWS = 20_000
LANGUAGE = "english"
FILTER = "recent"
NUM_PER_PAGE = 100
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 7
SAVE_EVERY_PAGES = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "steam_review_data"
CSV_PATH = OUTPUT_DIR / "elden_ring_reviews_20000_full_fields.csv"
STATE_PATH = OUTPUT_DIR / "elden_ring_reviews_20000_state.json"
METADATA_PATH = OUTPUT_DIR / "elden_ring_reviews_20000_metadata.json"
QUALITY_REPORT_PATH = OUTPUT_DIR / "elden_ring_reviews_20000_quality_report.json"

API_URL = f"https://store.steampowered.com/appreviews/{APPID}"

FIELDS = [
    "game",
    "appid",
    "recommendation_id",
    "language",
    "review",
    "voted_up",
    "playtime_at_review_hours",
    "playtime_forever_hours",
    "playtime_last_two_weeks_hours",
    "deck_playtime_at_review_hours",
    "timestamp_created",
    "timestamp_updated",
    "last_played",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "num_games_owned",
    "num_reviews_by_author",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "primarily_steam_deck",
    "scraped_at_utc",
    "review_word_count",
]


def utc_now_iso() -> str:
    """返回带时区的 UTC ISO 时间。"""
    return datetime.now(timezone.utc).isoformat()


def minutes_to_hours(value: Any) -> float | str:
    """Steam 游玩时长单位为分钟；缺失时留空，不把缺失误写成 0。"""
    if value is None or value == "":
        return ""
    try:
        return round(float(value) / 60.0, 4)
    except (TypeError, ValueError):
        return ""


def optional_number(value: Any) -> int | float | str:
    """保留数值；缺失值写为空白。"""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return ""


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Academic Steam review research collector/1.0",
            "Accept": "application/json",
        }
    )
    return session


def fetch_page(
    session: requests.Session,
    cursor: str,
) -> dict[str, Any]:
    params = {
        "json": 1,
        "filter": FILTER,
        "language": LANGUAGE,
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": NUM_PER_PAGE,
        "cursor": cursor,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("success") != 1:
                raise RuntimeError(
                    f"Steam API 返回 success={data.get('success')!r}"
                )
            return data
        except (requests.RequestException, ValueError, RuntimeError) as error:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"请求连续失败 {MAX_RETRIES + 1} 次：{error}"
                ) from error

            wait_seconds = min(60.0, 2**attempt + random.random())
            print(
                f"请求失败：{error}；{wait_seconds:.1f} 秒后进行第 "
                f"{attempt + 2} 次尝试。",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise AssertionError("不可到达的代码")


def normalize_review(review_data: dict[str, Any], scraped_at: str) -> dict[str, Any]:
    """把 Steam 的嵌套 JSON 映射为用户指定的 25 个 CSV 字段。"""
    author = review_data.get("author") or {}
    review_text = review_data.get("review") or ""

    return {
        "game": GAME,
        "appid": APPID,
        # ID 作为字符串保存，避免 Excel 对长整数产生精度损失。
        "recommendation_id": str(review_data.get("recommendationid", "")),
        "language": review_data.get("language", ""),
        "review": review_text,
        "voted_up": review_data.get("voted_up", ""),
        "playtime_at_review_hours": minutes_to_hours(
            author.get("playtime_at_review")
        ),
        "playtime_forever_hours": minutes_to_hours(author.get("playtime_forever")),
        "playtime_last_two_weeks_hours": minutes_to_hours(
            author.get("playtime_last_two_weeks")
        ),
        "deck_playtime_at_review_hours": minutes_to_hours(
            author.get("deck_playtime_at_review")
        ),
        "timestamp_created": optional_number(review_data.get("timestamp_created")),
        "timestamp_updated": optional_number(review_data.get("timestamp_updated")),
        "last_played": optional_number(author.get("last_played")),
        "votes_up": optional_number(review_data.get("votes_up")),
        "votes_funny": optional_number(review_data.get("votes_funny")),
        "weighted_vote_score": optional_number(
            review_data.get("weighted_vote_score")
        ),
        "comment_count": optional_number(review_data.get("comment_count")),
        "num_games_owned": optional_number(author.get("num_games_owned")),
        "num_reviews_by_author": optional_number(author.get("num_reviews")),
        "steam_purchase": review_data.get("steam_purchase", ""),
        "received_for_free": review_data.get("received_for_free", ""),
        "written_during_early_access": review_data.get(
            "written_during_early_access", ""
        ),
        "primarily_steam_deck": review_data.get("primarily_steam_deck", ""),
        "scraped_at_utc": scraped_at,
        # 这里使用空白字符分词；正式 NLP 清洗时可再采用更严格的 tokenizer。
        "review_word_count": len(review_text.split()),
    }


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """先写临时文件再替换，避免程序中断留下半个 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_existing_rows() -> tuple[list[dict[str, Any]], set[str]]:
    if not CSV_PATH.exists():
        return [], set()

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(
                "现有 CSV 的字段与本脚本不一致。请先重命名现有 CSV，"
                "或修改 CSV_PATH 后重新运行。"
            )
        rows = list(reader)

    unique_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        recommendation_id = row.get("recommendation_id", "")
        if recommendation_id and recommendation_id not in seen_ids:
            unique_rows.append(row)
            seen_ids.add(recommendation_id)

    return unique_rows, seen_ids


def load_resume_cursor(saved_row_count: int) -> str:
    """只有当状态文件与 CSV 行数一致时才使用已保存游标。"""
    if not STATE_PATH.exists():
        return "*"
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "*"

    if state.get("saved_unique_reviews") != saved_row_count:
        return "*"
    cursor = state.get("next_cursor")
    return str(cursor) if cursor else "*"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_final_reports(
    rows: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    pages_requested_this_run: int,
    stop_reason: str,
) -> None:
    language_counts = Counter(str(row.get("language", "")) for row in rows)
    missing_counts = {
        field: sum(row.get(field, "") in (None, "") for row in rows)
        for field in FIELDS
    }
    ids = [str(row.get("recommendation_id", "")) for row in rows]

    metadata = {
        "game": GAME,
        "appid": APPID,
        "target_unique_reviews": TARGET_REVIEWS,
        "collected_unique_reviews": len(rows),
        "sampling_method": "Consecutive most-recent reviews using Steam cursor pagination",
        "api_parameters": {
            "filter": FILTER,
            "language": LANGUAGE,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": NUM_PER_PAGE,
        },
        "fields": FIELDS,
        "time_units": {
            "playtime_at_review_hours": "hours",
            "playtime_forever_hours": "hours",
            "playtime_last_two_weeks_hours": "hours",
            "deck_playtime_at_review_hours": "hours",
        },
        "timestamp_units": {
            "timestamp_created": "Unix seconds UTC",
            "timestamp_updated": "Unix seconds UTC",
            "last_played": "Unix seconds UTC",
            "scraped_at_utc": "ISO 8601 UTC",
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "pages_requested_this_run": pages_requested_this_run,
        "stop_reason": stop_reason,
    }
    atomic_write_json(METADATA_PATH, metadata)

    quality_report = {
        "row_count": len(rows),
        "column_count": len(FIELDS),
        "unique_recommendation_id_count": len(set(ids)),
        "duplicate_recommendation_id_count": len(ids) - len(set(ids)),
        "language_counts": dict(language_counts),
        "empty_review_count": missing_counts["review"],
        "at_least_10_words_count": sum(
            int(float(row.get("review_word_count", 0) or 0)) >= 10 for row in rows
        ),
        "missing_value_counts": missing_counts,
        "csv_size_bytes": CSV_PATH.stat().st_size,
        "csv_sha256": file_sha256(CSV_PATH),
    }
    atomic_write_json(QUALITY_REPORT_PATH, quality_report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    rows, seen_ids = load_existing_rows()

    print(f"输出目录：{OUTPUT_DIR}")
    print(f"已存在的唯一评论：{len(seen_ids):,}")

    if len(seen_ids) >= TARGET_REVIEWS:
        rows = rows[:TARGET_REVIEWS]
        atomic_write_csv(CSV_PATH, rows)
        write_final_reports(
            rows=rows,
            started_at=started_at,
            finished_at=utc_now_iso(),
            pages_requested_this_run=0,
            stop_reason="target already reached",
        )
        print(f"目标已经完成：{len(rows):,} 条唯一评论。")
        return

    cursor = load_resume_cursor(len(rows))
    session = build_session()
    pages_requested = 0
    stop_reason = "unknown"

    try:
        while len(seen_ids) < TARGET_REVIEWS:
            data = fetch_page(session=session, cursor=cursor)
            pages_requested += 1
            reviews = data.get("reviews") or []
            next_cursor = data.get("cursor")

            if not reviews:
                stop_reason = "Steam API returned no more reviews"
                break

            scraped_at = utc_now_iso()
            for review_data in reviews:
                recommendation_id = str(review_data.get("recommendationid", ""))
                if not recommendation_id or recommendation_id in seen_ids:
                    continue
                rows.append(normalize_review(review_data, scraped_at))
                seen_ids.add(recommendation_id)
                if len(seen_ids) >= TARGET_REVIEWS:
                    break

            should_checkpoint = (
                pages_requested % SAVE_EVERY_PAGES == 0
                or len(seen_ids) >= TARGET_REVIEWS
                or not next_cursor
            )
            if should_checkpoint:
                atomic_write_csv(CSV_PATH, rows[:TARGET_REVIEWS])
                atomic_write_json(
                    STATE_PATH,
                    {
                        "saved_at_utc": utc_now_iso(),
                        "saved_unique_reviews": min(len(rows), TARGET_REVIEWS),
                        "next_cursor": next_cursor,
                    },
                )
                print(
                    f"已请求 {pages_requested:,} 页；"
                    f"已保存 {min(len(seen_ids), TARGET_REVIEWS):,}/"
                    f"{TARGET_REVIEWS:,} 条唯一评论。",
                    flush=True,
                )

            if not next_cursor or str(next_cursor) == cursor:
                stop_reason = "pagination cursor ended or repeated"
                break

            cursor = str(next_cursor)
            time.sleep(max(0.0, REQUEST_DELAY_SECONDS))
        else:
            stop_reason = "target reached"

    except KeyboardInterrupt:
        stop_reason = "interrupted by user; checkpoint saved"
        print("\n检测到手动中断，正在保存检查点……")
    finally:
        rows = rows[:TARGET_REVIEWS]
        atomic_write_csv(CSV_PATH, rows)
        atomic_write_json(
            STATE_PATH,
            {
                "saved_at_utc": utc_now_iso(),
                "saved_unique_reviews": len(rows),
                "next_cursor": cursor,
            },
        )
        write_final_reports(
            rows=rows,
            started_at=started_at,
            finished_at=utc_now_iso(),
            pages_requested_this_run=pages_requested,
            stop_reason=stop_reason,
        )
        session.close()

    print(f"完成：{len(rows):,} 条唯一评论。停止原因：{stop_reason}")
    print(f"CSV：{CSV_PATH}")
    print(f"元数据：{METADATA_PATH}")
    print(f"质量报告：{QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
