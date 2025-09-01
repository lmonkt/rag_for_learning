# core/web_search.py
import logging
import requests
from requests.adapters import HTTPAdapter
import importlib
from config import (
    SERPAPI_KEY, SEARCH_ENGINE, SERPAPI_URL, SEARCH_NUM_RESULTS,
    SEARCH_LANGUAGE, SEARCH_COUNTRY, SERPAPI_TIMEOUT,
    # 提供方与多引擎支持
    SEARCH_PROVIDER, BING_SEARCH_API_KEY, BING_SEARCH_API_URL,
    HTTP_RETRIES,
    # Google CSE
    GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX, GOOGLE_CSE_API_URL,
)

# 复用会话以提升网络可靠性
session = requests.Session()
session.mount('http://', HTTPAdapter(max_retries=HTTP_RETRIES))
session.mount('https://', HTTPAdapter(max_retries=HTTP_RETRIES))


def check_serpapi_key() -> bool:
    return bool(SERPAPI_KEY and SERPAPI_KEY.strip())


def check_bing_key() -> bool:
    return bool(BING_SEARCH_API_KEY and BING_SEARCH_API_KEY.strip())


def get_last_search_count() -> int | None:
    """返回最近一次 web_search 返回的结果数。"""
    return LAST_SEARCH_COUNT


def get_last_search_provider() -> str | None:
    """返回最近一次 web_search 实际使用的搜索提供方。"""
    return LAST_SEARCH_PROVIDER


def check_google_cse() -> bool:
    return bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)


def _parse_serpapi_results(payload: dict) -> list:
    results = []
    try:
        for item in payload.get("organic_results", []) or []:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
    except Exception as e:
        logging.error(f"解析 SerpAPI 结果失败: {e}")
    return results


def _parse_bing_results(payload: dict) -> list:
    results = []
    try:
        for item in (payload.get("webPages", {}) or {}).get("value", []) or []:
            results.append({
                "title": item.get("name"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
            })
    except Exception as e:
        logging.error(f"解析 Bing 结果失败: {e}")
    return results


def _parse_google_cse_results(payload: dict) -> list:
    results = []
    try:
        for item in payload.get("items", []) or []:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
    except Exception as e:
        logging.error(f"解析 Google CSE 结果失败: {e}")
    return results


def _map_ddg_region(lang: str) -> str:
    mapping = {
        "zh-CN": "cn-zh",
        "zh-TW": "tw-zh",
        "zh-HK": "hk-zh",
        "en-US": "us-en",
        "en-GB": "uk-en",
        "ja-JP": "jp-ja",
    }
    return mapping.get((lang or "").strip(), "wt-wt")


def _load_ddgs():
    """动态加载 DDG 客户端，优先 ddgs，回退 duckduckgo_search。"""
    try:
        mod = importlib.import_module("ddgs")
        cls = getattr(mod, "DDGS", None)
        if cls is not None:
            return cls, "ddgs"
    except Exception:
        pass
    try:
        mod = importlib.import_module("duckduckgo_search")
        cls = getattr(mod, "DDGS", None)
        if cls is not None:
            return cls, "duckduckgo_search"
    except Exception:
        pass
    return None, None


def _ddg_search(query: str, num_results: int | None = None) -> list:
    if num_results is None:
        num_results = SEARCH_NUM_RESULTS

    DDGS, using_pkg = _load_ddgs()
    if DDGS is None:
        logging.warning("未安装 ddgs/duckduckgo-search，无法使用 DuckDuckGo 搜索。")
        return []

    region = _map_ddg_region(SEARCH_LANGUAGE)
    results = []
    try:
        with DDGS(timeout=SERPAPI_TIMEOUT) as ddgs:
            for item in ddgs.text(query, region=region, safesearch="moderate", timelimit=None, max_results=num_results):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("href"),
                    "snippet": item.get("body"),
                })
        if using_pkg == "duckduckgo_search":
            logging.debug("提示：建议将依赖从 'duckduckgo-search' 迁移到 'ddgs' 以消除弃用告警。")
        return results
    except Exception as e:
        logging.error(f"DuckDuckGo 搜索失败: {e}")
        return []


def _serpapi_search(query: str, num_results: int | None = None) -> list:
    if not check_serpapi_key():
        raise ValueError("未设置 SERPAPI_KEY。")

    if num_results is None:
        num_results = SEARCH_NUM_RESULTS

    params = {
        "engine": SEARCH_ENGINE,
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "hl": SEARCH_LANGUAGE,
        "gl": SEARCH_COUNTRY,
    }
    try:
        resp = session.get(SERPAPI_URL, params=params, timeout=SERPAPI_TIMEOUT)
        resp.raise_for_status()
        return _parse_serpapi_results(resp.json())
    except Exception as e:
        logging.error(f"SerpAPI 搜索失败: {e}")
        return []


def _bing_search(query: str, num_results: int | None = None) -> list:
    if not check_bing_key():
        raise ValueError("未设置 BING_SEARCH_API_KEY。")

    if num_results is None:
        num_results = SEARCH_NUM_RESULTS

    mkt = (SEARCH_LANGUAGE or "zh-CN").strip()
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    params = {"q": query, "count": num_results, "mkt": mkt}
    try:
        resp = session.get(BING_SEARCH_API_URL, headers=headers, params=params, timeout=SERPAPI_TIMEOUT)
        resp.raise_for_status()
        return _parse_bing_results(resp.json())
    except Exception as e:
        logging.error(f"Bing 搜索失败: {e}")
        return []


def _google_cse_search(query: str, num_results: int | None = None) -> list:
    if not check_google_cse():
        raise ValueError("未配置 GOOGLE_CSE_API_KEY 或 GOOGLE_CSE_CX。")

    if num_results is None:
        num_results = SEARCH_NUM_RESULTS

    params = {
        "q": query,
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "num": max(1, min(10, num_results)),  # CSE 单次最多10条
        "safe": "off",
        "hl": (SEARCH_LANGUAGE or "zh-CN"),
    }
    try:
        resp = session.get(GOOGLE_CSE_API_URL, params=params, timeout=SERPAPI_TIMEOUT)
        resp.raise_for_status()
        return _parse_google_cse_results(resp.json())
    except Exception as e:
        logging.error(f"Google CSE 搜索失败: {e}")
        return []


# 全局变量用于跟踪搜索结果
LAST_SEARCH_PROVIDER = None
LAST_SEARCH_COUNT = 0


def web_search(query: str, num_results: int | None = None) -> list:
    """统一网络搜索入口：SerpAPI / Bing / Google CSE / DuckDuckGo。

    优先级：
    - provider=serpapi 且配置密钥 -> SerpAPI
    - provider=bing 且配置密钥 -> Bing
    - provider=google_cse 且配置密钥 -> Google CSE
    - provider=duckduckgo -> DuckDuckGo
    - provider=auto -> 依次尝试 SerpAPI -> Bing -> Google CSE -> DuckDuckGo
    """
    global LAST_SEARCH_PROVIDER, LAST_SEARCH_COUNT

    provider = (SEARCH_PROVIDER or "auto").lower()

    if provider == "serpapi":
        if check_serpapi_key():
            results = _serpapi_search(query, num_results)
            LAST_SEARCH_PROVIDER = "serpapi"
            LAST_SEARCH_COUNT = len(results)
            logging.info(f"WebSearch 使用 provider=serpapi，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
            return results
        else:
            LAST_SEARCH_COUNT = 0
            return []

    if provider == "bing":
        if check_bing_key():
            results = _bing_search(query, num_results)
            LAST_SEARCH_PROVIDER = "bing"
            LAST_SEARCH_COUNT = len(results)
            logging.info(f"WebSearch 使用 provider=bing，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
            return results
        else:
            LAST_SEARCH_COUNT = 0
            return []

    if provider == "google_cse":
        if check_google_cse():
            results = _google_cse_search(query, num_results)
            LAST_SEARCH_PROVIDER = "google_cse"
            LAST_SEARCH_COUNT = len(results)
            logging.info(f"WebSearch 使用 provider=google_cse，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
            return results
        else:
            LAST_SEARCH_COUNT = 0
            return []

    if provider == "duckduckgo":
        results = _ddg_search(query, num_results)
        LAST_SEARCH_PROVIDER = "duckduckgo"
        LAST_SEARCH_COUNT = len(results)
        logging.info(f"WebSearch 使用 provider=duckduckgo，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
        return results

    # auto 模式回退链
    if check_serpapi_key():
        results = _serpapi_search(query, num_results)
        LAST_SEARCH_PROVIDER = "serpapi"
        LAST_SEARCH_COUNT = len(results)
        logging.info(f"WebSearch 使用 provider=serpapi，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
        return results

    if check_bing_key():
        results = _bing_search(query, num_results)
        LAST_SEARCH_PROVIDER = "bing"
        LAST_SEARCH_COUNT = len(results)
        logging.info(f"WebSearch 使用 provider=bing，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
        return results

    if check_google_cse():
        results = _google_cse_search(query, num_results)
        LAST_SEARCH_PROVIDER = "google_cse"
        LAST_SEARCH_COUNT = len(results)
        logging.info(f"WebSearch 使用 provider=google_cse，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
        return results

    results = _ddg_search(query, num_results)
    LAST_SEARCH_PROVIDER = "duckduckgo"
    LAST_SEARCH_COUNT = len(results)
    logging.info(f"WebSearch 使用 provider=duckduckgo，query='{query}'，返回 {LAST_SEARCH_COUNT} 条结果。")
    return results


# 兼容旧接口：保留 serpapi_search 名称，但内部转发到 web_search
def serpapi_search(query: str, num_results: int | None = None) -> list:
    return web_search(query, num_results)
