"""Safety guardrails and validation for the advisory system."""
import re
from typing import Optional, Any


class AdvisoryGuardrails:
    """Implements safety checks and validation for advisory generation."""

    @staticmethod
    def validate_features(features: dict) -> tuple[bool, list[str]]:
        """Validate input features for safety.

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        critical_fields = ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm"]
        missing = [f for f in critical_fields if f not in features]
        if missing:
            warnings.append(f"Missing critical fields: {missing}")

        if "pH" in features:
            pH = features["pH"]
            if not (0 <= pH <= 14):
                warnings.append(f"pH value {pH} outside valid range [0, 14]")

        if "avg_humidity_percent" in features:
            h = features["avg_humidity_percent"]
            if not (0 <= h <= 100):
                warnings.append(f"Humidity {h}% outside valid range [0, 100]")

        numeric_fields = ["N", "P", "K", "total_rainfall_mm", "area"]
        for field in numeric_fields:
            if field in features and features[field] < 0:
                warnings.append(f"{field} cannot be negative")

        return (len(warnings) == 0, warnings)

    @staticmethod
    def sanitize_advisory_text(text: str) -> str:
        """Remove potentially harmful content from advisory text.

        Args:
            text: Generated advisory text

        Returns:
            Sanitized text
        """
        lines = text.split('\n')
        sanitized = []

        for line in lines:
            if line.strip() and not _contains_harmful_content(line):
                sanitized.append(line)

        return '\n'.join(sanitized)

    @staticmethod
    def is_agricultural_query(query: str, llm: Optional[Any] = None) -> tuple[bool, str]:
        """Validate whether a user query is strictly within the agricultural and farming domain.

        Blocks non-agricultural questions (e.g., coding, sports, movies, politics, cryptocurrency,
        general trivia) while permitting agronomy, crops, soil, pests, weather, farm records,
        and polite farmer conversational greetings.

        Args:
            query: The user query string.
            llm: Optional LLM instance for semantic edge-case verification.

        Returns:
            Tuple of (is_agricultural: bool, reason: str)
        """
        if not query or not query.strip():
            return (False, "Query is empty.")

        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Permitted conversational greetings & assistant identity inquiries
        if _is_farmer_greeting(q_lower):
            return (True, "Conversational farmer greeting or introduction.")

        # 2. Immediate rejection of unambiguous non-agricultural intent
        has_non_agri_trigger, trigger_desc = _matches_non_agricultural_triggers(q_lower)
        if has_non_agri_trigger:
            return (False, f"Non-agricultural domain query detected: {trigger_desc}.")

        # 3. Positive match against agricultural domain vocabulary
        if _contains_agricultural_concepts(q_lower):
            return (True, "Query matches agricultural domain terminology.")

        # 4. Optional zero-shot semantic verification with LLM if available
        if llm is not None:
            llm_verdict = AdvisoryGuardrails._verify_with_llm(q_clean, llm)
            if llm_verdict is not None:
                return llm_verdict

        # 5. Deterministic fallback: Query lacks agricultural context
        return (False, "Query contains no recognized agricultural, farming, or agronomic topics.")

    @staticmethod
    def _verify_with_llm(query: str, llm: Any) -> Optional[tuple[bool, str]]:
        """Verify borderline or complex questions with LLM zero-shot classification."""
        try:
            prompt = (
                "You are a strict domain safety guardrail for a Tamil Nadu precision agriculture advisory platform.\n"
                "Determine if the user query is legitimate and related to: agriculture, farming, crops, soil, "
                "fertilizers, irrigation, weather, agricultural pests/diseases, farm implements, farm economics, "
                "past farm records, or polite farmer greetings.\n\n"
                f'User Query: "{query}"\n\n'
                'Respond with EXACTLY ONE word:\n'
                '"AGRICULTURAL" if the query is related to farming/agriculture/greetings.\n'
                '"NON_AGRICULTURAL" if the query is unrelated (such as coding, entertainment, sports, politics, '
                'general trivia, gaming, cryptocurrencies, non-agri science/cooking).'
            )
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp)).strip().upper()
            if "NON_AGRICULTURAL" in text or "NON-AGRICULTURAL" in text or text.startswith("NO"):
                return (False, "Semantic verification classified query as non-agricultural.")
            elif "AGRICULTURAL" in text or text.startswith("YES"):
                return (True, "Semantic verification confirmed query is within agricultural domain.")
        except Exception:
            pass
        return None

    @staticmethod
    def get_non_agricultural_refusal(farmer_name: str = "Farmer", query: Optional[str] = None) -> str:
        """Generate a polite, farmer-friendly domain restriction response.

        Args:
            farmer_name: The farmer's name for personalized greeting.
            query: The rejected query (optional).

        Returns:
            Formatted Markdown refusal message guiding user to agricultural topics.
        """
        return (
            f"🚫 **Non-Agricultural Question Detected**\n\n"
            f"Vanakkam {farmer_name}! I am your practical **Agricultural Field Assistant & Agronomy Advisor**, "
            f"specifically trained to provide decision support for crop cultivation, soil health, fertilizer scheduling, "
            f"integrated pest management, and farm history in Tamil Nadu.\n\n"
            f"⚠️ **I can only answer questions related to agriculture and farming.**\n\n"
            f"**Please feel free to ask about:**\n"
            f"- 🌾 **Crops & Sowing:** Recommended crops, varieties, seed treatment, and cultivation timetables\n"
            f"- 🌱 **Soil & Fertilizers:** Soil pH, NPK splits, Urea scheduling, biofertilizers, and organic manure\n"
            f"- 🐛 **Pests & Diseases:** Stem borer, leaf blast, fall armyworm, and safe bio-control practices\n"
            f"- 💧 **Water & Weather:** Alternate Wetting and Drying (AWD), irrigation cycles, and rain impact\n"
            f"- 📜 **Farm Records:** Highest past profit, previous harvest yields, and season history"
        )


def _contains_harmful_content(text: str) -> bool:
    """Check if text contains potentially harmful content.

    Args:
        text: Text to check

    Returns:
        True if harmful content detected
    """
    harmful_patterns = [
        "toxic",
        "poison",
        "kill",
        "death",
        "banned",
        "illegal"
    ]

    text_lower = text.lower()
    return any(pattern in text_lower for pattern in harmful_patterns)


# =========================================================================
# DOMAIN GUARDRAILS VOCABULARY & MATCHING LOGIC
# =========================================================================

_GREETING_PATTERNS = [
    r"^(vanakkam|namaste|namaskaram|hello|hi|hey|good morning|good afternoon|good evening|great day)\b",
    r"^(who|what) are you\b",
    r"^what (can|do) you do\b",
    r"^how can you help\b",
    r"^(help|help me)\b",
    r"^(thank you|thanks|nandri)\b"
]

_NON_AGRI_RULES = [
    (r"\b(write\s+(a\s+)?(python|code|program|script|function|class)|debug\s+code|syntax\s+error|leetcode|algorithm)\b", "Software Programming / Coding"),
    (r"\b(python|javascript|typescript|c\+\+|golang|rust|html|css|react|angular|vue|docker|kubernetes|aws|azure|git|github|linux)\b", "Software & Computing"),
    (r"\b(movie|movies|cinema|film|films|hollywood|bollywood|kollywood|tollywood|netflix|spotify|song|songs|album|singer|celebrity|oscar|trailers?|box\s+office)\b", "Cinema & Entertainment"),
    (r"\b(video\s+game|playstation|xbox|nintendo|fortnite|minecraft|gta|anime|manga|marvel|dc\s+comics|batman|superman|spiderman)\b", "Gaming & Pop Culture"),
    (r"\b(cricket|ipl|football|fifa|soccer|premier\s+league|messi|ronaldo|basketball|nba|tennis|badminton|olympics|world\s+cup|super\s+bowl|formula\s+1|f1)\b", "Sports"),
    (r"\b(bitcoin|btc|ethereum|eth|crypto|cryptocurrency|blockchain|nft|stock\s+market|wall\s+street|forex|nasdaq|sensex|nifty|day\s+trading|options\s+trading)\b", "Cryptocurrency & Stock Trading"),
    (r"\b(president\s+of|prime\s+minister\s+of|election\s+results|white\s+house|ukraine\s+war|nato|united\s+nations)\b", "Geopolitics & Non-Agricultural Governance"),
    (r"\b(quantum\s+(physics|computing)|black\s+hole|galaxy|astronomy|spaceship|spacex|nasa)\b", "Astronomy & Theoretical Physics"),
    (r"\b(recipe\s+for\s+(pizza|cake|cookies|burger|pasta|biryani)|bake\s+a\s+cake|dating\s+advice|horoscope|zodiac)\b", "General Lifestyle / Cooking"),
    (r"\bcapital\s+of\s+(france|usa|germany|japan|england|china|italy|australia|canada|brazil|russia)\b", "General Geography Trivia"),
]

# Set of single-token and multi-token agronomic domain terms
_AGRI_DOMAIN_TERMS = {
    # Common Crops & Plantations
    "paddy", "rice", "samba", "kuruvai", "thaladi", "sornavari", "navarai", "wheat", "maize",
    "corn", "sugarcane", "cane", "cotton", "ragi", "millet", "millets", "jowar", "sorghum",
    "bajra", "pulse", "pulses", "gram", "chickpea", "blackgram", "urad", "greengram", "moong",
    "redgram", "tur", "pigeonpea", "cowpea", "groundnut", "peanut", "soybean", "soya", "sunflower",
    "sesame", "gingelly", "castor", "mustard", "turmeric", "ginger", "onion", "potato", "tomato",
    "brinjal", "eggplant", "chilli", "chili", "pepper", "cardamom", "clove", "tapioca", "cassava",
    "banana", "plantain", "coconut", "mango", "guava", "sapota", "papaya", "citrus", "lemon", "lime",
    "jasmine", "marigold", "rose", "crossandra", "rubber", "tea", "coffee", "fodder", "napier",
    "cumbu", "azolla", "daincha", "sunnhemp", "vegetable", "vegetables", "horticulture", "floriculture",
    # Agronomic Stages, Plants & Seeds
    "crop", "crops", "cropping", "plant", "plants", "planting", "planted", "seed", "seeds",
    "seedling", "seedlings", "sapling", "nursery", "sow", "sowing", "transplant", "transplanting",
    "germinate", "germination", "tillering", "panicle", "heading", "flowering", "milking", "dough",
    "maturity", "harvest", "harvests", "harvesting", "harvested", "yield", "yields", "threshing",
    "storage", "post-harvest", "season", "seasons", "kharif", "rabi", "zaid", "rotation", "intercrop",
    "intercropping", "companion", "spacing", "tillage", "plough", "ploughing", "plow", "plowing",
    "rotavator", "tractor", "mulch", "mulching", "pruning", "earthing up", "puddling", "leveling",
    "leaf", "leaves", "stem", "stems", "root", "roots", "pod", "pods", "grain", "grains", "shoot",
    "canopy", "foliage", "biomass", "ratoon", "growth", "grow", "growing", "grown",
    # Soil Health & Nutrients
    "soil", "soils", "loam", "clay", "alluvial", "red soil", "black soil", "delta", "cauvery",
    "ph", "acidity", "alkaline", "alkalinity", "saline", "salinity", "sodic", "fertility", "nutrient",
    "nutrients", "npk", "nitrogen", "phosphorus", "potassium", "potash", "calcium", "magnesium",
    "sulfur", "sulphur", "zinc", "boron", "iron", "copper", "manganese", "micronutrient",
    "micronutrients", "fertilizer", "fertiliser", "fertilizers", "fertilisers", "urea", "dap", "mop",
    "ssp", "fym", "farm yard manure", "manure", "compost", "vermicompost", "green manure",
    "biofertilizer", "biofertilizers", "azospirillum", "phosphobacteria", "rhizobium", "trichoderma",
    "pseudomonas", "neem cake", "dosage", "dosages", "dose", "split", "splits", "top dressing",
    "basal", "fertigation", "foliar", "deficiency", "chlorosis", "humus", "organic matter",
    # Water & Weather
    "water", "irrigate", "irrigating", "irrigation", "awd", "alternate wetting", "drip", "sprinkler",
    "furrow", "basin", "drain", "drainage", "waterlogging", "rain", "rainfall", "precipitation",
    "monsoon", "drought", "dry spell", "weather", "temperature", "humidity", "groundwater",
    "borewell", "well", "canal", "moisture", "evapotranspiration",
    # Pests, Diseases & Crop Protection
    "pest", "pests", "insect", "insects", "bug", "bugs", "borer", "stem borer", "leaf folder",
    "leaf miner", "planthopper", "bph", "glh", "bollworm", "armyworm", "caterpillar", "aphid",
    "aphids", "whitefly", "whiteflies", "thrips", "mite", "mites", "mealybug", "weevil", "nematode",
    "nematodes", "rodent", "rat", "disease", "diseases", "blight", "blast", "sheath blight",
    "leaf spot", "leaf spots", "rust", "rot", "root rot", "soft rot", "wilt", "mildew", "powdery mildew",
    "downy mildew", "smut", "canker", "fungus", "fungal", "bacteria", "bacterial", "virus", "viral",
    "mosaic", "weed", "weeds", "weeding", "herbicide", "herbicides", "weedicide", "pesticide",
    "pesticides", "insecticide", "insecticides", "fungicide", "fungicides", "biopesticide", "ipm",
    "neem", "nske", "azadirachtin", "pheromone", "pheromone trap", "sticky trap", "light trap",
    "spray", "spraying", "sprayer", "knapsack", "ppe", "endosulfan", "paraquat", "monocrotophos",
    "chlorpyrifos", "cibrc", "symptom", "lesion", "infection", "bio-control",
    # Farm Management, Economics, Land & Records
    "farm", "farms", "farmer", "farmers", "farming", "field", "fields", "plot", "plots", "acre",
    "acres", "hectare", "hectares", "cent", "cents", "gunta", "kisan", "profit", "profits",
    "profitable", "revenue", "loss", "quintal", "bag", "bags", "tonne", "tonnes", "mandi", "market",
    "market price", "msp", "crop insurance", "pmfby", "kcc", "subsidy", "scheme", "past harvest",
    "past harvests", "previous harvest", "previous crop", "history", "records", "database",
    "agronomy", "agronomic", "agritech", "tnau", "icar", "kvk",
    # Livestock / Allied Farming
    "livestock", "cattle", "cow", "cows", "buffalo", "goat", "sheep", "poultry", "chicken", "dairy",
    "dung", "silage", "sericulture", "apiculture", "honey bee", "aquaculture", "fish pond"
}


def _is_farmer_greeting(text: str) -> bool:
    """Check if query is a simple greeting or introductory question."""
    cleaned = re.sub(r"[^\w\s]", "", text).strip()
    if not cleaned:
        return False
    return any(re.search(pat, cleaned) for pat in _GREETING_PATTERNS)


def _matches_non_agricultural_triggers(text: str) -> tuple[bool, str]:
    """Check if query matches explicit non-agricultural triggers."""
    for pattern, description in _NON_AGRI_RULES:
        if re.search(pattern, text):
            return (True, description)
    return (False, "")


def _contains_agricultural_concepts(text: str) -> bool:
    """Check if text contains words or phrases from the agricultural domain."""
    # Split text into tokens and normalized phrases
    tokens = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    if tokens & _AGRI_DOMAIN_TERMS:
        return True

    # Check for multi-word or compound domain matches
    for term in _AGRI_DOMAIN_TERMS:
        if " " in term and term in text.lower():
            return True

    return False
