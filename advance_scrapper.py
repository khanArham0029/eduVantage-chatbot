import asyncio
import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter

async def run_graduate_program_crawler():
    # --- Choose ONE content filter ---
    USE_BM25 = False  # Change to True to use BM25ContentFilter

    if USE_BM25:
        content_filter = BM25ContentFilter(
            user_query="graduate master program courses curriculum degree syllabus",
            bm25_threshold=1.1
        )
    else:
        content_filter = PruningContentFilter(
            threshold=0.4,
            threshold_type="dynamic",
            min_word_threshold=5
        )

    # --- Markdown generator with content filter ---
    md_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
        options={
            "ignore_links": True,
            "escape_html": False,
            "body_width": 80
        }
    )

    # --- Filtering setup ---
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["cam.ac.uk"],
            blocked_domains=[]
        ),
        URLPatternFilter(patterns=[
            "*graduate*",
            "*masters*",
            "*msc*",
            "*postgraduate*",
            "*academics*",
            "*program*",
            "*degree*",
            "*course*",
            "*curriculum*"
        ]),
        ContentTypeFilter(allowed_types=["text/html"])
    ])

    # --- Scoring setup ---
    keyword_scorer = KeywordRelevanceScorer(
        keywords=[
            "graduate", "master", "program", "course", "curriculum",
            "degree", "credit hours", "duration", "admission", "requirements",
            "modules", "syllabus", "track", "thesis", "non-thesis","academics",
        ],
        weight=0.85
    )

    # --- Crawl config ---
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=3,
            include_external=False,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=md_generator,
        stream=True,
        verbose=True
    )

    # --- Run and save results ---
    results = []
    with open("cambridge_2_graduate_programs_markdown.jsonl", "w", encoding="utf-8") as outfile:
        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun("https://www.cam.ac.uk/", config=config):
                score = result.metadata.get("score", 0)
                depth = result.metadata.get("depth", 0)
                print(f"Depth: {depth} | Score: {score:.2f} | {result.url}")

                results.append(result)

                # Save filtered markdown output (fit_markdown)
                json.dump({
                    "url": result.url,
                    "depth": depth,
                    "score": score,
                    "markdown": result.markdown.fit_markdown,
                    "raw_html_excerpt": result.markdown.fit_html[:300]  # Optional: Save HTML snippet
                }, outfile)
                outfile.write("\n")

    # --- Summary ---
    print(f"\n✅ Crawled {len(results)} relevant pages")
    avg_score = sum(r.metadata.get("score", 0) for r in results) / len(results)
    print(f"📊 Average score: {avg_score:.2f}")

    depth_counts = {}
    for result in results:
        d = result.metadata.get("depth", 0)
        depth_counts[d] = depth_counts.get(d, 0) + 1

    print("📁 Pages crawled by depth:")
    for depth, count in sorted(depth_counts.items()):
        print(f"  Depth {depth}: {count} pages")

if __name__ == "__main__":
    asyncio.run(run_graduate_program_crawler())
