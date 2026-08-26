from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from starter.agent import Agent


HELP_TEXT = """Commands:
  /help                 Show this help
  /profile              Show the active preference profile
  /profile fit,comfort  Replace profile tags and start a new session
  /reset                Clear the conversation and start again
  /quit                 Exit the demo

Try messages such as:
  I need women's shoes for trail running, preferably blue and breathable.
  Leather, under $80.
  Actually, switch to a formal black office shoe instead.
"""


@dataclass(frozen=True, slots=True)
class ProductView:
    parent_asin: str
    title: str
    price: str
    rating: str
    rating_number: int
    categories: str
    store: str


def parse_profile_tags(value: str) -> list[str]:
    return list(
        dict.fromkeys(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        )
    )


def build_profile(tags: list[str]) -> dict:
    readable = ", ".join(tags) if tags else "no saved preference tags"
    return {
        "purchase_frequency": "demo session",
        "average_prior_rating": None,
        "rating_style": "not provided",
        "preference_tags": tags,
        "summary": f"Interactive demo profile: {readable}.",
    }


def load_product_views(catalog_path: str | Path) -> dict[str, ProductView]:
    views: dict[str, ProductView] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            raw_price = product.get("price")
            try:
                price = f"${float(raw_price):,.2f}" if raw_price not in (None, "") else "price unavailable"
            except (TypeError, ValueError):
                price = str(raw_price)
            raw_rating = product.get("average_rating")
            try:
                rating = f"{float(raw_rating):.1f}/5" if raw_rating not in (None, "") else "not rated"
            except (TypeError, ValueError):
                rating = str(raw_rating)
            try:
                rating_number = int(product.get("rating_number") or 0)
            except (TypeError, ValueError):
                rating_number = 0
            categories = product.get("categories") or []
            if isinstance(categories, list):
                category_text = " > ".join(str(item) for item in categories[-3:])
            else:
                category_text = str(categories)
            views[parent_asin] = ProductView(
                parent_asin=parent_asin,
                title=str(product.get("title") or "Untitled product"),
                price=price,
                rating=rating,
                rating_number=rating_number,
                categories=category_text or "Uncategorized",
                store=str(product.get("store") or "Unknown store"),
            )
    return views


def format_product(rank: int, product: ProductView) -> str:
    return (
        f"  {rank}. {product.title}\n"
        f"     {product.parent_asin} | {product.price} | {product.rating} "
        f"({product.rating_number:,} ratings) | {product.store}\n"
        f"     {product.categories}"
    )


class InteractiveShoppingDemo:
    def __init__(
        self,
        agent: Agent,
        products: dict[str, ProductView],
        profile_tags: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self.agent = agent
        self.products = products
        self.profile_tags = list(profile_tags or [])
        self.top_k = min(max(int(top_k), 1), 10)
        self.turn = 0
        self.session_id = ""
        self.reset()

    def reset(self) -> None:
        self.turn = 0
        self.session_id = f"interactive_{uuid.uuid4().hex}"
        self.agent.reset(self.session_id, build_profile(self.profile_tags))

    def set_profile(self, tags: list[str]) -> None:
        self.profile_tags = list(tags)
        self.reset()

    def respond(self, user_message: str) -> tuple[dict, list[ProductView]]:
        if self.turn >= 10:
            self.reset()
        self.turn += 1
        response = self.agent.respond(
            self.session_id,
            user_message,
            self.turn,
            self.top_k,
        )
        recommendations: list[ProductView] = []
        for item in response.get("recommendations") or []:
            if not isinstance(item, dict):
                continue
            product = self.products.get(str(item.get("parent_asin") or ""))
            if product is not None:
                recommendations.append(product)
        return response, recommendations

    def run(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        output_fn("\nCompass Interactive Shopping Copilot")
        output_fn("Type a shopping request in natural language. Use /help for commands.")
        if self.profile_tags:
            output_fn("Profile tags: " + ", ".join(self.profile_tags))

        while True:
            try:
                raw = input_fn("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("\nGoodbye.")
                return
            if not raw:
                continue
            command, _, argument = raw.partition(" ")
            command = command.lower()
            if command in {"/quit", "/exit"}:
                output_fn("Goodbye.")
                return
            if command == "/help":
                output_fn(HELP_TEXT.rstrip())
                continue
            if command == "/reset":
                self.reset()
                output_fn("Conversation cleared. Tell me what you are shopping for.")
                continue
            if command == "/profile":
                if argument.strip():
                    self.set_profile(parse_profile_tags(argument))
                    output_fn("Profile updated; a new session has started.")
                output_fn(build_profile(self.profile_tags)["summary"])
                continue
            if command.startswith("/"):
                output_fn(f"Unknown command: {command}. Use /help.")
                continue

            was_last_turn = self.turn == 9
            response, recommendations = self.respond(raw)
            output_fn(f"\nCompass [turn {self.turn}/10] > {response['message']}")
            if recommendations:
                output_fn("\nRecommendations:")
                for rank, product in enumerate(recommendations, start=1):
                    output_fn(format_product(rank, product))
            else:
                output_fn("No catalog matches yet. Add a category, use case, or product detail.")
            if was_last_turn:
                output_fn("\nThe 10-turn limit was reached. Your next message starts a new session.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interactively with the offline shopping copilot")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11), metavar="1-10")
    parser.add_argument(
        "--profile-tags",
        default="fit,comfort,durability",
        help="Comma-separated cold-start preference tags",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        raise SystemExit(
            f"Catalog not found: {catalog_path}. Follow README.md to download the participant-kit catalog."
        )

    print(f"Loading {catalog_path} and building in-memory indexes...")
    products = load_product_views(catalog_path)
    agent = Agent(catalog_path)
    print(f"Ready: {len(products):,} products indexed.")
    demo = InteractiveShoppingDemo(
        agent,
        products,
        profile_tags=parse_profile_tags(args.profile_tags),
        top_k=args.top_k,
    )
    demo.run()


if __name__ == "__main__":
    main()
