from bs4 import BeautifulSoup
import re

# 1. 读取网页
try:
    with open("publications.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, 'html.parser')
except Exception as e:
    print(f"❌ 读取文件出错: {e}")
    exit()

items = soup.find_all('div', class_='w3-cell-middle')
seen_records = set()

print(f"🔍 找到 {len(items)} 个数据块，开始解析...\n")

for item in items:
    # === 【关键修改】 ===
    # 先把 <br> 标签换成 "|||"，这样我们以此为界限切分
    for br in item.find_all("br"):
        br.replace_with("|||")

    # 获取纯文本 (不加 separator，这样作者名字的链接就会自动拼合在一起)
    full_text = item.get_text(strip=True)

    # 用 ||| 切分，rows[0]就是第一行，rows[1]就是Venue
    rows = full_text.split("|||")

    if len(rows) < 2:
        continue

    info_line = rows[0]  # 完整的第一行：Title by Author in 2023
    venue = rows[1]  # 第二行：Venue

    # 正则提取
    match = re.search(r'(.*) by (.*) in (\d{4})', info_line)

    if match:
        title = match.group(1).strip()
        authors = match.group(2).strip()
        year = match.group(3).strip()

        # 去重
        unique_id = (title, year)
        if unique_id not in seen_records:
            seen_records.add(unique_id)

            # 打印结果
            print(f"Title:   {title}")
            print(f"Venue:   {venue}")
            print(f"Year:    {year}")
            print(f"Authors: {authors}")
            print("-" * 50)

print(f"\n✅ 全部完成！共获取 {len(seen_records)} 条独特记录。")