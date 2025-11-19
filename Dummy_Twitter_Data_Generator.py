import random, re, string
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ---------- paths ----------
IN_CSV  = r"markets_with_returns.csv"
OUT_TXT = r"twitter_dummy_data.txt"

# ---------- reproducibility ----------
random.seed(7)
np.random.seed(7)

# ---------- load market data ----------
df = (
    pd.read_csv(IN_CSV, parse_dates=["Date"])
)
df = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2024-12-31")].copy()
df["ret"] = df["Nifty_Return"].fillna(0.0)

# ---------- turn daily return into tone bucket ----------
def tone_bucket(r):
    if r >= 0.60:   return "strong_up"
    if r >= 0.15:   return "up"
    if r <= -0.60:  return "strong_down"
    if r <= -0.15:  return "down"
    return "flat"

df["tone"] = df["ret"].apply(tone_bucket)

# ---------- vocabulary / style ----------
ALIAS   = ["Nifty 50", "Nifty50", "#NIFTY50", "Nifty", "NSE50"]
TICKERS = ["$NIFTY", "$NSE", "$NIFTY50"]
TAGS    = ["#India", "#Sensex", "#Markets", "#DalalStreet", "#NSE", "#IndianMarkets", "#Trading", "#StockMarket"]

EMOJIS_POS = ["🚀", "📈", "🔥", "🤑", "✅", "💚", "💪", "🎯", "⚡", "🌟"]
EMOJIS_NEG = ["📉", "😬", "❌", "🥶", "🔻", "💣", "😰", "⚠️", "🩸", "👎"]
EMOJIS_FLAT= ["🤔", "🧭", "⏳", "🤝", "📰", "😐", "🤷", "📊", "⚖️"]

MENTIONS   = ["@traderjoe", "@quantgal", "@macroalpha", "@riskparity", "@valuehacker",
              "@newsbot", "@chartmaster", "@optionsflow", "@marketpulse", "@tradingview"]
URL_BASES  = ["https://t.co/", "http://bit.ly/", "https://tinyurl.com/"]

# ========== EXPANDED VOCABULARY ==========

# Stock-tweet verbs & reactions
OPEN_UP     = ["gaps up", "opens green", "pops at the open", "pushes higher off the bell",
               "ticks up on the open", "strong open", "rockets higher", "melting up", "ripping"]
OPEN_DOWN   = ["gaps down", "opens red", "slips at the open", "pushes lower off the bell",
               "ticks down on the open", "weak open", "selling off", "dumping", "bleeding"]
OPEN_FLAT   = ["opens flat", "quiet open", "unch at the bell", "sideways open",
               "basically flat open", "choppy open", "consolidating", "no direction"]

# Intraday reactions (NEW!)
REACTIONS_BULL = [
    "buying the dip ASAP", "adding here", "shorts getting squeezed",
    "momentum building", "this is the way", "beautiful breakout",
    "told you so", "calling it now", "going parabolic", "to the moon",
    "institutions loading up", "smart money accumulating", "fomo incoming"
]

REACTIONS_BEAR = [
    "selling the rip", "cutting losses", "longs getting wrecked",
    "momentum fading", "red flag", "failing hard", "I warned you",
    "knife catching szn", "dead cat bounce?", "heading lower",
    "smart money exiting", "distribution phase", "panic selling"
]

REACTIONS_NEUTRAL = [
    "just watching", "waiting for confirmation", "no position yet",
    "50/50 here", "coin flip tbh", "unclear picture", "mixed signals",
    "theta gang wins today", "boring day", "snooze fest", "range bound"
]

# Questions/Uncertainty (NEW!)
QUESTIONS_BULL = [
    "new ATH incoming?", "resistance break?", "sustainable?",
    "FOMO time?", "too late to enter?", "how high can it go?"
]

QUESTIONS_BEAR = [
    "support breaking?", "capitulation soon?", "bottom here?",
    "panic selling?", "catching a falling knife?", "dead cat?"
]

QUESTIONS_NEUTRAL = [
    "breakout or fakeout?", "which way?", "choppy or consolidation?",
    "accumulation or distribution?", "bull trap or bear trap?"
]

ADD_BULL    = ["buyers active", "breadth positive", "dip bought", "breakout watch",
               "risk on", "follow-through building", "new highs in sight", "volume surging",
               "all systems go", "bears capitulating", "shorts covering"]

ADD_BEAR    = ["sellers in control", "breadth weak", "sell the rip?", "breakdown watch",
               "risk off", "lower highs", "failed bounce", "volume drying up",
               "all hope lost", "bulls trapped", "death cross forming"]

ADD_NEUTRAL = ["range day?", "inside day vibes", "waiting on data", "no big flows yet",
               "two-way trade", "patience", "chopfest", "theta decay day",
               "consolidation mode", "digesting gains"]

# Sentiment words
POS_WORDS = ["bullish", "constructive", "positive", "accumulating", "bid",
             "support respected", "breakout", "strong hands", "momentum", "trending"]

NEG_WORDS = ["bearish", "negative", "distribution", "offer heavy", "rejected",
             "breakdown", "supply", "weak hands", "reversal", "losing steam"]

MACRO = ["CPI print", "PMI", "GDP", "earnings", "rates", "yields", "crude",
         "FX", "geopolitics", "bond supply", "Fed minutes", "jobs data",
         "inflation worries", "China news", "oil spike", "dollar strength"]

# Tweet styles (NEW!)
STYLES_CASUAL = [
    "lol", "lmao", "ngl", "tbh", "imo", "fwiw", "rn", "literally",
    "bruh", "fr", "no cap", "sheesh", "wild", "crazy"
]

STYLES_ANALYTICAL = [
    "per my analysis", "based on technicals", "fundamentally speaking",
    "chart says", "levels to watch", "key resistance/support", "RSI overbought",
    "moving average cross", "volume profile", "fibonacci retracement"
]

# ---------- helper functions ----------

def rand_url():
    base = random.choice(URL_BASES)
    slug = "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(6, 10)))
    return base + slug

def times_near_open(n):
    # 72% near open (08:00-11:00), 20% mid-day (11:00-14:00), 8% after hours
    out = []
    for _ in range(n):
        r = random.random()
        if r < 0.72:
            h = random.randint(8, 11)
        elif r < 0.92:
            h = random.randint(11, 14)
        else:
            h = random.randint(14, 17)
        m = random.randint(0, 59)
        out.append(f"{h:02d}{m:02d}")
    return out

def choose_alias():  return random.choice(ALIAS)
def choose_ticker(): return random.choice(TICKERS)
def choose_tags(k=2): return " ".join(random.sample(TAGS, k=random.randint(1, min(k, len(TAGS)))))

def build_line(date_str, hhmm, tone):
    alias  = choose_alias()
    ticker = choose_ticker()
    tags   = choose_tags(random.randint(1, 3))  # Variable tag count
    macro  = random.choice(MACRO) if random.random() < 0.4 else ""  # Not always macro
    handle = random.choice(MENTIONS) if random.random() < 0.25 else ""
    url    = rand_url() if random.random() < 0.35 else ""

    # Add casual style sometimes
    casual = random.choice(STYLES_CASUAL) if random.random() < 0.15 else ""
    analytical = random.choice(STYLES_ANALYTICAL) if random.random() < 0.2 else ""

    if tone == "strong_up":
        verb = random.choice(OPEN_UP)
        adds = random.sample(ADD_BULL, k=random.randint(1, 3))
        senti = random.sample(POS_WORDS, k=random.randint(1, 2))
        reaction = random.choice(REACTIONS_BULL) if random.random() < 0.5 else ""
        question = random.choice(QUESTIONS_BULL) if random.random() < 0.3 else ""
        emj  = " " + random.choice(EMOJIS_POS) + (random.choice(EMOJIS_POS) if random.random() < 0.3 else "")
        lead = random.choice(["Morning take:", "Heads up:", "Alert:", "🔥 WATCH:", ""])

    elif tone == "up":
        verb = random.choice(OPEN_UP)
        adds = random.sample(ADD_BULL, k=random.randint(1, 2))
        if random.random() < 0.4:
            adds += random.sample(ADD_NEUTRAL, k=1)
        senti = random.sample(POS_WORDS, k=1)
        reaction = random.choice(REACTIONS_BULL) if random.random() < 0.35 else ""
        question = random.choice(QUESTIONS_BULL) if random.random() < 0.25 else ""
        emj  = " " + random.choice(EMOJIS_POS + EMOJIS_FLAT)
        lead = random.choice(["Quick take:", "Note:", "Update:", ""])

    elif tone == "strong_down":
        verb = random.choice(OPEN_DOWN)
        adds = random.sample(ADD_BEAR, k=random.randint(1, 3))
        senti = random.sample(NEG_WORDS, k=random.randint(1, 2))
        reaction = random.choice(REACTIONS_BEAR) if random.random() < 0.5 else ""
        question = random.choice(QUESTIONS_BEAR) if random.random() < 0.3 else ""
        emj  = " " + random.choice(EMOJIS_NEG) + (random.choice(EMOJIS_NEG) if random.random() < 0.3 else "")
        lead = random.choice(["Caution:", "⚠️ Alert:", "Heads up:", "WARNING:", ""])

    elif tone == "down":
        verb = random.choice(OPEN_DOWN)
        adds = random.sample(ADD_BEAR, k=random.randint(1, 2))
        if random.random() < 0.4:
            adds += random.sample(ADD_NEUTRAL, k=1)
        senti = random.sample(NEG_WORDS, k=1)
        reaction = random.choice(REACTIONS_BEAR) if random.random() < 0.35 else ""
        question = random.choice(QUESTIONS_BEAR) if random.random() < 0.25 else ""
        emj  = " " + random.choice(EMOJIS_NEG + EMOJIS_FLAT)
        lead = random.choice(["Quick take:", "Note:", "Update:", ""])

    else:  # flat
        verb = random.choice(OPEN_FLAT)
        adds = random.sample(ADD_NEUTRAL, k=random.randint(1, 2))
        senti = random.sample(POS_WORDS + NEG_WORDS, k=random.randint(0, 1))
        reaction = random.choice(REACTIONS_NEUTRAL) if random.random() < 0.4 else ""
        question = random.choice(QUESTIONS_NEUTRAL) if random.random() < 0.35 else ""
        emj  = " " + random.choice(EMOJIS_FLAT)
        lead = random.choice(["Morning:", "EOD:", "Update:", ""])

    # Compose tweet with variable structure
    bits = [
        f"[{date_str} {hhmm}]",
        lead,
        f"{alias} {verb}.",
        macro + "." if macro else "",
        " ".join(adds + senti),
        reaction,
        question,
        casual,
        analytical,
        tags,
        ticker + emj,
        handle,
        url,
    ]

    # Remove empty strings and join
    tweet = " ".join(x for x in bits if x).strip()

    # Occasionally make it more casual (remove timestamp, less structured)
    if random.random() < 0.1:
        tweet = tweet.replace(f"[{date_str} {hhmm}] ", "")

    return tweet

# ---------- generate ----------
MEAN_PER_DAY = 60   # Average tweets per day
tweets = []

for _, row in df.iterrows():
    # More variability in daily tweet count
    n = np.random.poisson(lam=MEAN_PER_DAY)
    n = max(n, 10)  # At least 10 tweets per day

    # Spike in tweets on high volatility days
    if abs(row["ret"]) > 1.0:
        n = int(n * 1.5)  # 50% more tweets on volatile days

    dstr = row["Date"].strftime("%Y-%m-%d")
    for t in times_near_open(n):
        tweets.append(build_line(dstr, t, row["tone"]))

# Shuffle to make it more realistic (not all chronological)
random.shuffle(tweets)

# ---------- write ----------
out_path = Path(OUT_TXT)
with out_path.open("w", encoding="utf-8", newline="\n") as f:
    for line in tweets:
        f.write(line + "\n")

print(f"wrote {len(tweets):,} lines → {out_path}")
print(f"Average {len(tweets)/len(df):.1f} tweets per day")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
