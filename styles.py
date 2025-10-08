# -*- coding: utf-8 -*-
# styles.py — все стили вынесены из main.py

from typing import Any, Dict, List

Style = Dict[str, Any]
STYLE_PRESETS: Dict[str, Style] = { "Портрет у окна": { "desc": "Крупный кинопортрет у большого окна; мягкая тень от рамы, живое боке.", "role": "cinematic window light portrait", "outfit": "neutral top", "props": "soft bokeh, window frame shadow on background", "bg": "large window with daylight glow, interior blur", "comp": "closeup", "tone": "daylight" }, "Портрет 85мм": { "desc": "Классика 85мм — мизерная ГРИП.", "role": "85mm look beauty portrait", "outfit": "minimal elegant top", "props": "creamy bokeh, shallow depth of field", "bg": "neutral cinematic backdrop", "comp": "closeup", "tone": "warm" }, "Бьюти студия": { "desc": "Чистый студийный свет, кожа без «пластика».", "role": "studio beauty portrait", "outfit": "clean minimal outfit", "props": "catchlights, controlled specular highlights", "bg": "seamless studio background with soft light gradients", "comp": "closeup", "tone": "daylight" }, "Кинопортрет": { "desc": "Рембрандтовский свет и мягкая плёнка.", "role": "cinematic rembrandt light portrait", "outfit": "neutral film wardrobe", "props": "subtle film grain", "bg": "moody backdrop with soft falloff", "comp": "closeup", "tone": "cool" }, "Фильм-нуар (портрет)": { "desc": "Дым, жёсткие тени, свет из жалюзи.", "role": "film noir portrait", "outfit": "vintage attire", "props": "venetian blinds light pattern, cigarette smoke curling", "bg": "high contrast noir backdrop", "comp": "closeup", "tone": "noir" }, "Стритвэр город": { "desc": "Уличный лук, город дышит.", "role": "streetwear fashion look", "outfit_f": "crop top and joggers, sneakers", "outfit": "hoodie and joggers, sneakers", "props": "glass reflections, soft film grain", "bg": "daytime city street with graffiti and shop windows", "comp": "half", "tone": "daylight" }, "Вечерний выход": { "desc": "Красная дорожка и блеск.", "role": "celebrity on red carpet", "outfit_f": "elegant evening gown", "outfit": "classic tuxedo", "props": "press lights, velvet ropes, photographers", "bg": "red carpet event entrance", "comp": "half", "tone": "warm" }, "Бизнес": { "desc": "Лобби стеклянного офиса, строгая геометрия.", "role": "corporate executive portrait", "outfit_f": "tailored business suit", "outfit": "tailored business suit", "props": "tablet or folder", "bg": "modern glass office lobby with depth", "comp": "half", "tone": "daylight" }, "Ночной неон": { "desc": "Кибернуар, мокрый асфальт.", "role": "urban night scene", "outfit_f": "long coat, boots", "outfit": "long coat, boots", "props": "colored reflections on wet asphalt, light rain droplets", "bg": "neon signs and steam from manholes", "comp": "half", "tone": "neon" }, "Врач у палаты": { "desc": "Белый халат, стетоскоп, палата за спиной.", "role": "medical doctor", "outfit_f": "white lab coat, scrub cap, stethoscope", "outfit": "white lab coat, scrub cap, stethoscope", "props": "ID badge, clipboard", "bg": "hospital ward interior with bed and monitors", "comp": "half", "tone": "daylight" }, "Хирург операционная": { "desc": "Холодные приборы и блики.", "role": "surgeon in the operating room", "outfit": "surgical scrubs, mask, cap, gloves", "props": "surgical lights and instruments", "bg": "operating theatre with equipment", "comp": "half", "tone": "cool" }, "Шеф-повар кухня": { "desc": "Огонь и пар, энергия ресторана.", "role": "head chef", "outfit": "white chef jacket and apron", "props": "pan with flames, stainless steel counters, copper pans", "bg": "professional restaurant kitchen", "comp": "half", "tone": "warm" }, "Учёный лаборатория": { "desc": "Стекло, приборы, подсветки.", "role": "scientist in a lab", "outfit": "lab coat, safety glasses", "props": "flasks, pipettes, LED indicators", "bg": "modern laboratory benches and glassware", "comp": "half", "tone": "cool" }, "Боксер на ринге": { "desc": "Жёсткий верхний свет, пот, канаты.", "role": "boxer on the ring", "outfit_f": "boxing sports bra and shorts, gloves", "outfit": "boxing shorts and gloves, mouthguard visible", "props": "ring ropes, sweat sheen, tape on wrists", "bg": "boxing ring under harsh top lights", "comp": "half", "tone": "cool" }, "Фитнес зал": { "desc": "Контровый свет между тренажёрами.", "role": "fitness athlete training", "outfit_f": "sports bra and leggings", "outfit": "tank top and shorts", "props": "chalk dust, dumbbells or cable machine", "bg": "gym with machines and dramatic backlight", "comp": "half", "tone": "cool" }, "Приключенец (руины)": { "desc": "Пыльные лучи, древние камни.", "role_f": "tomb raider explorer", "role": "tomb raider explorer", "outfit_f": "tactical outfit, fingerless gloves, utility belt", "outfit": "tactical outfit, fingerless gloves, utility belt", "props": "leather straps patina, map tube", "bg": "ancient sandstone ruins with sun rays and dust motes", "comp": "full", "tone": "warm" }, "Пустынный исследователь": { "desc": "Дюны, песок и жар.", "role": "desert explorer", "outfit": "scarf, cargo outfit, boots", "props": "sand blowing in wind", "bg": "sand dunes and canyon under harsh sun", "comp": "full", "tone": "warm" }, "Горы снег": { "desc": "Суровая красота высокогорья.", "role": "alpinist", "outfit": "mountain jacket, harness, crampons", "props": "ice axe in hand, spindrift", "bg": "snow ridge and blue shadows, cloudy sky", "comp": "full", "tone": "cool" }, "Серфер": { "desc": "Брызги, солнечные блики, доска.", "role": "surfer athlete on a wave", "outfit_f": "black wetsuit", "outfit": "black wetsuit", "props": "a visible surfboard under the subject's arm or feet, water spray, droplets", "bg": "ocean wave breaking, golden backlight", "comp": "full", "tone": "warm" }, "Эльфийская знать": { "desc": "Лесной храм и лучи в тумане.", "role_f": "elven queen in a regal pose", "role": "elven king in a regal pose", "outfit_f": "flowing emerald gown with golden embroidery, delicate crown", "outfit": "ornate armor with emerald cloak, elegant crown", "props": "elven jewelry, filigree filigree patterns", "bg": "ancient forest temple, god rays in mist", "comp": "full", "tone": "candle" }, "Самурай в храме": { "desc": "Лакированные доспехи, фонари, листья.", "role": "samurai warrior in a shrine courtyard", "outfit": "lacquered samurai armor, kabuto helmet", "props": "katana visible in hand", "bg": "Shinto shrine with lanterns, falling leaves", "comp": "full", "tone": "warm" }, "Средневековый рыцарь": { "desc": "Полированный латный доспех, штандарты.", "role": "medieval knight", "outfit": "full plate armor with cloak", "props": "sword and shield", "bg": "castle tournament yard with banners and dust", "comp": "full", "tone": "daylight" }, "Пират на палубе": { "desc": "Треуголка, сабля, мокрая палуба, шторм.", "role_f": "pirate captain", "role": "pirate captain", "outfit_f": "tricorn hat, leather corset, white shirt", "outfit": "tricorn hat, leather vest, white shirt", "props": "cutlass in hand, rope rigging, wet wood highlights", "bg": "ship deck in storm, sails and rigging, sea spray, gulls", "comp": "full", "tone": "cool" }, "Древняя Греция": { "desc": "Белый мрамор и лазурь.", "role_f": "ancient Greek goddess", "role": "ancient Greek hero", "outfit_f": "white chiton with gold trim, diadem", "outfit": "white chiton with gold trim, laurel wreath", "props": "gold accessories", "bg": "white marble colonnade, statues, olive trees, turquoise pool", "comp": "half", "tone": "warm" }, "Королева": { "desc": "Коронованная особа в тронном зале.", "role_f": "queen on a throne", "role": "king on a throne", "outfit_f": "royal gown with long train, jeweled crown, scepter", "outfit": "royal robe with golden embroidery, jeweled crown, scepter", "props": "ornate jewelry, velvet textures", "bg": "grand castle throne room with chandeliers and marble columns", "comp": "half", "tone": "warm" }, "Киберпанк улица": { "desc": "Неон, мокрый асфальт, голограммы.", "role": "cyberpunk character walking in the street", "outfit_f": "leather jacket, high-waist pants, boots", "outfit": "leather jacket, techwear pants, boots", "props": "holographic billboards, overhead cables", "bg": "neon signs, wet asphalt reflections, steam from manholes", "comp": "full", "tone": "neon" }, "Космический скафандр": { "desc": "Хард sci-fi.", "role": "astronaut", "outfit": "realistic EVA spacesuit", "props": "helmet reflections, suit details", "bg": "starfield and spaceship hangar", "comp": "full", "tone": "cool" }, "Арктика": { "desc": "Холодное сияние, айсберги и белый медвежонок.", "role_f": "arctic explorer holding a white polar bear cub", "role": "arctic explorer holding a white polar bear cub", "outfit_f": "white thermal parka with fur hood, knit beanie and mittens", "outfit": "white thermal parka with fur hood, knit beanie and gloves", "props": "polar bear cub cuddled safely in arms, drifting ice, snow crystals in air", "bg": "icebergs and frozen sea, low sun halo, blowing snow", "comp": "half", "tone": "cool" }, "Альпы (гламур)": { "desc": "Гламурный отдых в горах: лыжи/сноуборд, террасы, горное солнце.", "role_f": "alpine fashion vacationer with skis", "role": "alpine fashion vacationer with snowboard", "outfit_f": "sleek white ski suit, fur-trimmed hood, chic goggles", "outfit": "stylish ski jacket and pants, goggles on helmet", "props": "skis or snowboard, steam from mulled wine cup", "bg": "alpine chalet terrace with snowy peaks and cable cars", "comp": "half", "tone": "warm" }, "Франция (Париж)": { "desc": "Берет, багет, круассан и башня на фоне.", "role": "parisian street scene character", "outfit_f": "striped shirt, red beret, trench, scarf", "outfit": "striped shirt, beret, trench, scarf", "props": "baguette and croissant in paper bag, café tables", "bg": "Eiffel Tower in the distance, Haussmann buildings, café awning", "comp": "half", "tone": "daylight" }, "Джунгли (Тарзан)": { "desc": "Густые тропики и дикие звери рядом (безопасно).", "role_f": "jungle adventurer", "role": "jungle adventurer", "outfit_f": "leather jungle top and skirt, rope belt", "outfit": "leather jungle outfit, rope belt", "props": "jungle vines, soft mist, a crocodile or a snake or a panther nearby, not attacking", "bg": "dense tropical jungle, waterfalls and sunbeams through canopy", "comp": "full", "tone": "warm" }, "Детство": { "desc": "Съёмка в детском образе: игрушки, флажки, шарики.", "role": "child portrait in playful setting", "outfit_f": "cute cardigan, skirt with suspenders, bow headband", "outfit": "cute sweater and suspenders", "props": "teddy bear, balloons, crayons, building blocks", "bg": "cozy kids room with garlands and soft daylight", "comp": "closeup", "tone": "daylight", "allow_age_change": True, },"Свадьба": { "desc": "свадебная сцена.", "role_f": "bride in elegant wedding dress", "role": "groom in classic tuxedo", "outfit_f": "white lace wedding gown, veil, bouquet", "outfit": "black tuxedo with boutonnière", "props": "flower petals in air, ring box visible", "bg": "sunlit ceremony arch with flowers", "comp": "half", "tone": "warm" }, "Хаос": { "desc": "Кинематографический бардак: всё рушится, но герой спокоен.", "role": "hero in cinematic disaster scene", "outfit_f": "modern streetwear with dust marks", "outfit": "modern streetwear with dust marks", "props": "embers and sparks in the air, flying papers, cracked glass", "bg": "burning house and collapsing structures in background, dramatic smoke", "comp": "full", "tone": "noir" }, "Инопланетяне": { "desc": "Фантастика: НЛО, лучи, загадочная пыль.", "role": "person confronted by hovering UFOs", "outfit_f": "sleek sci-fi coat", "outfit": "sleek sci-fi coat", "props": "tractor beams, dust motes, floating debris", "bg": "night field with hovering saucers and moody clouds", "comp": "full", "tone": "cool" }, "Фридайвер под водой": { "desc": "Подводная съёмка, пузыри, лучи сквозь толщу воды.", "role_f": "freediver underwater", "role": "freediver underwater", "outfit_f": "apnea wetsuit without tank, long fins, mask", "outfit": "apnea wetsuit without tank, long fins, mask", "props": "air bubbles, sunbeams from surface, small fish around", "bg": "deep blue water with rocky arch or reef", "comp": "full", "tone": "cool" }, "Деревня": { "desc": "Теплая сельская сцена.", "role": "villager in rustic setting", "outfit_f": "linen dress, knitted cardigan, headscarf optional", "outfit": "linen shirt, vest", "props": "basket with apples, wooden fence, hay", "bg": "rural cottage yard with garden and chickens far in background", "comp": "half", "tone": "warm" }, "Россия (зимняя)": { "desc": "Зимний пейзаж, берёзы, снежные сугробы.", "role": "person in Russian winter scene", "outfit_f": "down coat, ushanka hat, woolen scarf, felt boots", "outfit": "down parka, ushanka hat, woolen scarf, felt boots", "props": "steam from breath, snowflakes in air, samovar on wooden table", "bg": "traditional wooden house with ornate window frames and birch trees", "comp": "half", "tone": "cool" }, "Теннис": { "desc": "Теннисный корт и динамика.", "role": "tennis player on court", "outfit_f": "white tennis dress and visor", "outfit": "white tennis kit and headband", "props": "racket in hand, tennis balls mid-air motion blur", "bg": "hard court with service lines and green windscreen", "comp": "half", "tone": "daylight" }, "Дельтаплан": { "desc": "Свобода полёта над горами.", "role": "hang glider pilot running a takeoff", "outfit": "windbreaker, harness, helmet, gloves", "props": "hang glider wings overhead, lines and A-frame visible", "bg": "ridge launch with valley and clouds below", "comp": "full", "tone": "daylight" }, "Космопилот на мостике": { "desc": "Пульт, индикаторы, готовность к гиперпрыжку.", "role": "starship pilot on the bridge", "outfit": "flight suit, helmet under arm", "props": "control panels with glowing indicators", "bg": "spaceship bridge interior", "comp": "half", "tone": "cool" }, }
# === ДОБАВКИ К STYLE_PRESETS ===
STYLE_PRESETS.update({
    # ——— Животные/Существа (безопасно) ———
    "Взрослый медведь рядом": {
        "desc": "Сцена в тайге: рядом взрослый бурый медведь (спокойно, безопасно).",
        "role": "nature explorer",
        "outfit_f": "outdoor jacket, hiking pants, boots",
        "outfit": "outdoor jacket, hiking pants, boots",
        "props": "calm brown bear standing nearby, breath vapor in cool air",
        "bg": "conifer forest clearing with soft fog",
        "comp": "half", "tone": "cool"
    },
    "Со змеёй": {
        "desc": "Экзотическая фотосессия: змея на руках/плече (безопасно).",
        "role": "exotic photo session",
        "outfit_f": "sleeveless top or dress (non-revealing), arm cuff",
        "outfit": "fitted shirt (rolled sleeves)",
        "props": "non-venomous snake gently handled, calm pose",
        "bg": "studio set with tropical leaves and warm backlight",
        "comp": "half", "tone": "warm"
    },
    "С крокодилом": {
        "desc": "Сафари-парк: крокодил на заднем плане (безопасно).",
        "role": "safari visitor",
        "outfit": "light safari outfit, hat",
        "props": "crocodile resting on bank in distance, wooden walkway railing",
        "bg": "wetland lagoon with reeds and sun glare",
        "comp": "half", "tone": "daylight"
    },
    "На драконе в небесах": {
        "desc": "Фэнтези-полет на драконе.",
        "role_f": "dragon rider heroine",
        "role": "dragon rider hero",
        "outfit_f": "wind-swept cloak, leather bracers",
        "outfit": "leather armor, cloak",
        "props": "scales, reins, clouds streaking by",
        "bg": "dramatic sky above mountains",
        "comp": "full", "tone": "cool"
    },

    # ——— Жанры/Поп-культура ———
    "Аладдин и восточная сказка": {
        "desc": "Восточный базар, дворец, лампа-джинн (вайб восточной сказки).",
        "role": "arabian nights adventurer",
        "outfit_f": "embroidered top and harem pants, veil optional",
        "outfit": "vest, sash belt, harem pants",
        "props": "magic lamp glint, patterned carpets",
        "bg": "sandstone palace domes and market stalls",
        "comp": "half", "tone": "warm"
    },
    "Сказка Дисней": {
        "desc": "Светлая сказочная стилизация с крупными выразительными глазами.",
        "role": "fairy-tale protagonist",
        "outfit_f": "pastel dress with subtle sparkles",
        "outfit": "storybook outfit with cape",
        "props": "tiny twinkles around, soft vignette",
        "bg": "storybook castle and garden",
        "comp": "half", "tone": "daylight"
    },
    "Пиксар анимация": {
        "desc": "Тёплая семейная CGI-эстетика, мягкие формы, кинематографичная подсветка.",
        "role": "family animation character",
        "outfit": "casual friendly clothes",
        "props": "subtle subsurface scattering on skin, rim light",
        "bg": "friendly suburban street or cozy room",
        "comp": "half", "tone": "warm"
    },
    "Супергерой": {
        "desc": "Комиксовый супергерой: динамика, контровой свет, цветные неоны.",
        "role": "superhero in action pose",
        "outfit_f": "form-fitting suit with emblem (tasteful, non-revealing)",
        "outfit": "form-fitting suit with emblem",
        "props": "speed lines, comic halftone accents",
        "bg": "night city skyline with spotlights",
        "comp": "full", "tone": "neon"
    },

    # ——— Роли ———
    "Домработница": {
        "desc": "Уютный интерьер, хозяйственные атрибуты.",
        "role": "housekeeper",
        "outfit_f": "modest house dress and apron",
        "outfit": "utility shirt and apron",
        "props": "feather duster, neatly stacked towels",
        "bg": "sunlit living room with plants",
        "comp": "half", "tone": "daylight"
    },
    "Учитель": {
        "desc": "Классная комната, доска, схемы.",
        "role": "teacher in classroom",
        "outfit_f": "blouse and skirt or pantsuit",
        "outfit": "shirt and chinos or suit",
        "props": "chalk or marker, books",
        "bg": "blackboard with formulas/maps, desks",
        "comp": "half", "tone": "daylight"
    },
    "Медсестра (игриво)": {
        "desc": "Игровая стилизация медперсонала (без откровенности).",
        "role_f": "nurse in playful themed outfit",
        "role": "male nurse in playful themed outfit",
        "outfit_f": "short uniform skirt, white stockings, nurse cap (tasteful)",
        "outfit": "partially unbuttoned white coat over scrubs (tasteful)",
        "props": "stethoscope, clipboard, ID badge",
        "bg": "hospital corridor with soft bokeh lights",
        "comp": "half", "tone": "warm"
    },
    "Программист": {
        "desc": "Тёмная комната, мониторы, неон.",
        "role": "software engineer at desk",
        "outfit": "hoodie or tee, headphones",
        "props": "code on monitors, RGB keyboard",
        "bg": "dual-monitor setup with window city glow",
        "comp": "half", "tone": "cool"
    },

    # ——— Этно/Культура ———
    "Бразилия": {
        "desc": "Карнавал/пляжный вайб.",
        "role": "brazilian vibes character",
        "outfit_f": "colorful outfit (non-revealing carnival elements)",
        "outfit": "colorful beach casual",
        "props": "confetti or beach ball, palm shadows",
        "bg": "Rio coastline or carnival street",
        "comp": "half", "tone": "warm"
    },
    "Индия": {
        "desc": "Яркие ткани, узоры, храм/базар.",
        "role": "character in indian setting",
        "outfit_f": "sari-inspired dress (modest), bangles",
        "outfit": "kurta-inspired outfit",
        "props": "marigold garlands, rangoli patterns",
        "bg": "temple gopuram or colorful market",
        "comp": "half", "tone": "warm"
    },

    # ——— Волшебство/праздники ———
    "Фея": {
        "desc": "Эфирные крылья, светлячки, магическая пыль.",
        "role_f": "forest fairy",
        "role": "forest fairy",
        "outfit_f": "sparkly leaf-like dress",
        "outfit": "leaf-like tunic and cloak",
        "props": "tiny glowing fireflies, dust sparkles",
        "bg": "mossy forest with god rays",
        "comp": "half", "tone": "candle"
    },
    "Хэллоуин": {
        "desc": "Тыквы, свечи, паутина.",
        "role": "halloween partygoer",
        "outfit": "costume with playful spooky accents",
        "props": "jack-o-lanterns, candles, paper bats",
        "bg": "porch or party with cobweb decor",
        "comp": "half", "tone": "noir"
    },
    "Новый год": {
        "desc": "Ёлка, гирлянды, бенгальские огни.",
        "role": "new year celebration",
        "outfit": "festive sweater or dress/suit",
        "props": "sparklers, gift boxes, garlands",
        "bg": "cozy living room with decorated tree",
        "comp": "half", "tone": "warm"
    },
    "Рождество": {
        "desc": "Камин, носочки, снежные узоры на окне.",
        "role": "christmas scene",
        "outfit": "cozy knitwear",
        "props": "wreath, candles, cookies on plate",
        "bg": "fireplace and decorated room",
        "comp": "half", "tone": "warm"
    },
    "Пасха": {
        "desc": "Пастельные тона, крашеные яйца, корзинки.",
        "role": "easter scene",
        "outfit": "pastel casual",
        "props": "painted eggs, wicker basket, tulips",
        "bg": "sunny kitchen table with linens",
        "comp": "half", "tone": "daylight"
    },

    # ——— Ночные/клуб ———
    "Дискотека": {
        "desc": "Клуб, лазеры, дым.",
        "role": "club night dancer",
        "outfit": "sparkly party wear",
        "props": "laser beams, fog, mirror ball bokeh",
        "bg": "crowded dance floor, DJ lights",
        "comp": "half", "tone": "neon"
    },

    # ——— Тёмные/социальные ———
    "Тюрьма": {
        "desc": "Камера/коридор с решётками.",
        "role": "inmate or visitor (neutral)",
        "outfit": "plain clothing or jumpsuit",
        "props": "bars, metal door, number plate",
        "bg": "prison corridor with cold light",
        "comp": "half", "tone": "cool"
    },
    "Гранж уличный (сквот)": {
        "desc": "Гранж-стайл, стрит-арт, слегка неряшливо.",
        "role": "grunge street look",
        "outfit": "layered worn clothes, beanie",
        "props": "peeling posters, graffiti, worn textures",
        "bg": "abandoned building entrance with tags",
        "comp": "half", "tone": "noir"
    },

    # ——— Робот/кибер ———
    "Робот": {
        "desc": "Полуробот/андроид — панельки, швы.",
        "role": "humanoid android",
        "outfit": "sleek synthetic suit or plates",
        "props": "glowing seams, panel lines",
        "bg": "clean lab bay or white void",
        "comp": "half", "tone": "cool"
    },

    # ——— Возрастные ———
    "Старый возраст": {
        "desc": "Аккуратное «состаривание» лица (только стилизация).",
        "role": "same person aged up",
        "outfit": "same wardrobe vibe",
        "props": "soft silver hair hints, gentle wrinkles",
        "bg": "neutral portrait backdrop",
        "comp": "closeup", "tone": "daylight",
        "allow_age_change": True
    },
    "Молодой возраст": {
        "desc": "Лёгкое «омоложение» лица (только стилизация).",
        "role": "same person aged down",
        "outfit": "same wardrobe vibe",
        "props": "smoother skin microfeatures",
        "bg": "neutral portrait backdrop",
        "comp": "closeup", "tone": "daylight",
        "allow_age_change": True
    },
})

STYLE_CATEGORIES: Dict[str, List[str]] = { "Портреты": ["Портрет у окна", "Портрет 85мм", "Бьюти студия", "Кинопортрет", "Фильм-нуар (портрет)"], "Современные": ["Стритвэр город", "Вечерний выход", "Бизнес", "Ночной неон"], "Профессии": ["Врач у палаты", "Хирург операционная", "Шеф-повар кухня", "Учёный лаборатория", "Боксер на ринге", "Фитнес зал"], "Приключения": ["Приключенец (руины)", "Пустынный исследователь", "Горы снег", "Серфер"], "Фэнтези/История": ["Эльфийская знать", "Самурай в храме", "Средневековый рыцарь", "Пират на палубе", "Древняя Греция", "Восточная сказка (Аладдин-лайк)", "Сказка студии (диснееподобно)", "Семейная CGI-анимация (пиксар-подобно)","Супергеройский комикс (AAA)" "Королева"], "Sci-Fi": ["Киберпанк улица", "Космический скафандр", "Космопилот на мостике"], "Путешествия": ["Арктика", "Альпы (гламур)", "Франция (Париж)", "Россия (зимняя)", "Деревня"], "Экшен/Адвенчур": ["Джунгли (Тарзан)", "Хаос", "Инопланетяне", "Дельтаплан"], "Спорт/Вода": ["Теннис", "Фридайвер под водой", "Серфер"], "Животные/Существа": [
        "Взрослый медведь рядом", "Со змеёй", "С крокодилом", "На драконе в небесах"
    ],
    "Роли": [
        "Домработница", "Учитель", "Медсестра (игриво)", "Программист"
    ],
    "Этно/Культура": [
        "Бразилия", "Индия"
    ],
    "Волшебство": [
        "Фея"
    ],
    "Праздники": [
        "Хэллоуин", "Новый год", "Рождество", "Пасха"
    ],
    "Ночные/Клуб": [
        "Дискотека"
    ],
    "Тёмные/Социальные": [
        "Тюрьма", "Гранж уличный (сквот)"
    ],
    "Роботы/Кибернетика": [
        "Робот"
    ],
    "Возраст": [
        "Старый возраст", "Молодой возраст"
    ],
    # раз ты просила — определяю и эти:
    "События/Сцены": [
        "Детство", "Свадьба"
    ],
}

THEME_BOOST = { "Пират на палубе": "rope rigging, storm clouds, wet highlights on wood, sea spray, gulls", "Древняя Греция": "ionic capitals, olive trees, turquoise water reflections, gold trim accents", "Ночной неон": "rain droplets on lens, colored reflections on wet asphalt", "Фильм-нуар (портрет)": "venetian blinds light pattern, cigarette smoke curling, deep black shadows", "Приключенец (руины)": "floating dust motes in sunrays, chipped sandstone blocks, leather straps patina", "Горы снег": "spindrift blown by wind, crampon scratches on ice, distant ridge line", "Киберпанк улица":"holographic billboards flicker, cable bundles overhead, neon kanji signs", "Серфер": "rimlight on water droplets, sun flare", "Королева": "subtle film grain, ceremonial ambience", }
THEME_BOOST.update({ "Арктика": "diamond-dust glitter in cold air, low sun halo, frost crystals on clothing", "Альпы (гламур)": "sunflare off snow, chalet wood textures, gondola cables in distance", "Франция (Париж)": "café chalk menu board, wrought iron balcony rails, warm bakery glow", "Джунгли (Тарзан)": "god rays through canopy, wet leaf speculars, mist near ground", "Детство": "soft pastel garlands, shallow dof sparkles, gentle vignette", "Свадьба": "bokeh from fairy lights, soft veil translucency", "Хаос": "embers, flying paper scraps, dramatic smoke layers, slight camera shake feeling", "Инопланетяне": "volumetric beams, dust motes, faint radio-glitch halation", "Фридайвер под водой": "caustic light patterns, particulate backscatter, gentle blue gradient", "Деревня": "warm wood patina, sun dust in air, linen texture details", "Россия (зимняя)": "crisp breath vapor, snow sparkle, frosty window details", "Теннис": "chalk dust from lines, motion blur of ball strings", "Дельтаплан": "wind-rippled jacket, wing fabric texture, valley haze layers" }) # Сцены, где чаще всего уводит лицо → понижаем CFG и принудительно включаем lockface
THEME_BOOST.update({
    "Взрослый медведь рядом": "cool morning haze, pine resin scent implied, damp moss highlights",
    "Со змеёй": "soft speculars on scales, tropical leaf shadows",
    "С крокодилом": "sun glitter on ripples, wooden railing texture",
    "На драконе в небесах": "wind-torn cloak edges, cloud streaks, distant peaks",
    "Аладдин и восточная сказка": "brass lamp glint, patterned textiles, warm dust haze",
    "Сказка Дисней": "gentle bloom, pastel palette, friendly sparkles",
    "Пиксар анимация": "soft subsurface scattering, rim light, friendly palette",
    "Супергерой": "dramatic rim light, neon reflections, halftone micro-texture",
    "Домработница": "sun dust in air, lemon-fresh cleanliness vibe",
    "Учитель": "chalk smudge on fingers, diagram edges on board",
    "Медсестра (игриво)": "clean reflections on tiles, soft corridor bokeh",
    "Программист": "monitor bloom, reflection in pupils, cable clutter hints",
    "Бразилия": "confetti sparkle, beach haze or street banners",
    "Индия": "gold thread glints, marigold petals",
    "Фея": "fairy dust trails, tiny light orbs",
    "Хэллоуин": "candle halos, fog low to ground",
    "Новый год": "string lights bokeh, sparkler trails",
    "Рождество": "fireplace glow, frosted window patterns",
    "Пасха": "pastel paper texture, sun through curtains",
    "Дискотека": "laser volumetrics, mirror-ball shards",
    "Тюрьма": "cold fluorescent falloff, metal scuffs",
    "Гранж уличный (сквот)": "torn poster edges, paint drips",
    "Робот": "micro panel gaps, brushed metal sheen",
    "Старый возраст": "subtle silver hair strands, gentle skin micro-wrinkles",
    "Молодой возраст": "soft youthful glow, smooth micro-contrast",
})
SCENE_GUIDANCE = { "Киберпанк улица": 3.2, "Космический скафандр": 3.2, "Самурай в храме": 3.2, "Средневековый рыцарь": 3.2, }
SCENE_GUIDANCE.update({ "Джунгли (Тарзан)": 3.2, "Инопланетяне": 3.2, "Хаос": 3.2, "Фридайвер под водой": 3.0, "Дельтаплан": 3.2, "Арктика": 3.2, "Детство": 3.0, })
SCENE_GUIDANCE.update({
    "Супергерой": 3.2,
    "Аладдин и восточная сказка": 3.4,
    "Сказка Дисней": 3.4,
    "Пиксар анимация": 3.4,
    "На драконе в небесах": 3.0,
    "Фея": 3.2,
    "Дискотека": 3.2,
    "Тюрьма": 3.2,
    "Гранж уличный (сквот)": 3.0,
    "Робот": 3.2,
    "Взрослый медведь рядом": 3.2,
    "Со змеёй": 3.2,
    "С крокодилом": 3.0,
    "Бразилия": 3.2,
    "Индия": 3.2,
    "Старый возраст": 3.0,
    "Молодой возраст": 3.0,
})

RISKY_PRESETS = set(SCENE_GUIDANCE.keys())
RISKY_PRESETS.update(set([
  "Супергерой", "Восточная сказка (Аладдин-лайк)",
  "Сказка Дисней", "Пиксар анимация",
  "На драконе в небесах", "Фея", "Дискотека", "Тюрьма",
  "Гранж уличный (сквот)", "Робот", "Взрослый медведь рядом",
  "Со змеёй", "С крокодилом", "Бразилия", "Индия",
  "Старый возраст", "Молодой возраст"
]))