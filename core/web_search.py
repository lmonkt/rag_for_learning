# core/web_search.py
import requests
import logging
from config import SERPAPI_KEY, SEARCH_ENGINE


def check_serpapi_key():
    return SERPAPI_KEY and SERPAPI_KEY.strip() != ""


def serpapi_search(query: str, num_results: int = 5) -> list:
    """执行SerpAPI搜索并返回结构化结果。"""
    # TODO: (改进方向) 多元数据源接入
    # 思路: 这里可以很容易地替换为其他搜索引擎API，如 Tavily, SearxNG, Google Custom Search API等。
    # 只需修改这里的API调用逻辑，并保持返回格式一致即可。
    if not check_serpapi_key():
        raise ValueError("未设置 SERPAPI_KEY。")

    params = {
        "engine": SEARCH_ENGINE,
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "hl": "zh-CN",
        "gl": "cn"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        search_data = response.json()

        # 解析结果
        results = []
        if "organic_results" in search_data:
            for item in search_data["organic_results"]:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                })
        return results
    except Exception as e:
        logging.error(f"网络搜索失败: {e}")
        return []