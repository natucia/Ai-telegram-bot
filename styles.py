# -*- coding: utf-8 -*-
# styles.py — коммерческие стили, яркие описания, расширенные категории

from typing import Any, Dict, List

Style = Dict[str, Any]

# =========================
#        ПРЕСЕТЫ
# =========================
STYLE_PRESETS: Dict[str, Style] = {
    # ===== ПОРТРЕТЫ / МОДА =====
    "Портрет у окна": {
        "desc": "Киношный крупный портрет у большого окна; мягкая тень от рамы, живое боке.",
        "role": "cinematic window light portrait",
        "outfit": "neutral top",
        "props": "soft bokeh, window frame shadow on background",
        "bg": "large window with daylight glow, interior blur",
        "comp": "closeup", "tone": "daylight"
    },
    "85 мм": {
        "desc": "Классика 85 мм: микроскопическая ГРИП, акцент на глаза.",
        "role": "85mm look beauty portrait",
        "outfit": "minimal elegant top",
        "props": "creamy bokeh, shallow depth of field",
        "bg": "neutral cinematic backdrop",
        "comp": "closeup", "tone": "warm"
    },
    "Бьюти-студия": {
        "desc": "Чистый студийный свет, текстура кожи сохранена (никакого «пластика»).",
        "role": "studio beauty portrait",
        "outfit": "clean minimal outfit",
        "props": "catchlights, controlled specular highlights",
        "bg": "seamless studio background with soft light gradients",
        "comp": "closeup", "tone": "daylight"
    },
    "Кинопортрет Рембрандта": {
        "desc": "Рембрандтовский свет, мягкая «плёнка», благородные тени.",
        "role": "cinematic rembrandt light portrait",
        "outfit": "neutral film wardrobe",
        "props": "subtle film grain",
        "bg": "moody backdrop with soft falloff",
        "comp": "closeup", "tone": "cool"
    },
    "Фильм-нуар": {
        "desc": "Нуар: свет из жалюзи, дым, резкий контраст.",
        "role": "film noir portrait",
        "outfit": "vintage attire",
        "props": "venetian blinds light pattern, cigarette smoke curling",
        "bg": "high contrast noir backdrop",
        "comp": "closeup", "tone": "noir"
    },
    "Стритвэр мегаполис": {
        "desc": "Уличный лук, отражения витрин, городской драйв.",
        "role": "streetwear fashion look",
        "outfit_f": "crop top and joggers, sneakers",
        "outfit": "hoodie and joggers, sneakers",
        "props": "glass reflections, soft film grain",
        "bg": "daytime city street with graffiti and shop windows",
        "comp": "half", "tone": "daylight"
    },
    "Вечерний выход": {
        "desc": "Глянцевый блеск, софиты, фотокорреспонденты.",
        "role": "celebrity on red carpet",
        "outfit_f": "elegant evening gown",
        "outfit": "classic tuxedo",
        "props": "press lights, velvet ropes, photographers",
        "bg": "red carpet event entrance",
        "comp": "half", "tone": "warm"
    },
    "Бизнес-портрет C-suite": {
        "desc": "Строгая геометрия стеклянного лобби, лидерский вайб.",
        "role": "corporate executive portrait",
        "outfit_f": "tailored business suit",
        "outfit": "tailored business suit",
        "props": "tablet or folder",
        "bg": "modern glass office lobby with depth",
        "comp": "half", "tone": "daylight"
    },
    "Ночной неон": {
        "desc": "Кибернуар: мокрый асфальт, пар из люков, неоновые вывески.",
        "role": "urban night scene",
        "outfit_f": "long coat, boots",
        "outfit": "long coat, boots",
        "props": "colored reflections on wet asphalt, light rain droplets",
        "bg": "neon signs and steam from manholes",
        "comp": "half", "tone": "neon"
    },

    # ===== ПРОФЕССИИ / СПОРТ =====
    "Доктор у палаты": {
        "desc": "Белый халат, стетоскоп, палата за спиной.",
        "role": "medical doctor",
        "outfit_f": "white lab coat, scrub cap, stethoscope",
        "outfit": "white lab coat, scrub cap, stethoscope",
        "props": "ID badge, clipboard",
        "bg": "hospital ward interior with bed and monitors",
        "comp": "half", "tone": "daylight"
    },
    "Хирург в операционной": {
        "desc": "Холодные приборы, яркие лампы, стерильная атмосфера.",
        "role": "surgeon in the operating room",
        "outfit": "surgical scrubs, mask, cap, gloves",
        "props": "surgical lights and instruments",
        "bg": "operating theatre with equipment",
        "comp": "half", "tone": "cool"
    },
    "Шеф-повар в огне": {
        "desc": "Горящие сковороды, пар и медь — энергия ресторана.",
        "role": "head chef",
        "outfit": "white chef jacket and apron",
        "props": "pan with flames, stainless steel counters, copper pans",
        "bg": "professional restaurant kitchen",
        "comp": "half", "tone": "warm"
    },
    "Учёный в лаборатории": {
        "desc": "Стекло, индикаторы, чистые линии.",
        "role": "scientist in a lab",
        "outfit": "lab coat, safety glasses",
        "props": "flasks, pipettes, LED indicators",
        "bg": "modern laboratory benches and glassware",
        "comp": "half", "tone": "cool"
    },
    "Боксёр на ринге": {
        "desc": "Жёсткий верхний свет, пот, канаты — момент силы.",
        "role": "boxer on the ring",
        "outfit_f": "boxing sports bra and shorts, gloves",
        "outfit": "boxing shorts and gloves, mouthguard visible",
        "props": "ring ropes, sweat sheen, tape on wrists",
        "bg": "boxing ring under harsh top lights",
        "comp": "half", "tone": "cool"
    },
    "Фитнес-зал — контровый свет": {
        "desc": "Тренажёры, меловая пыль, выраженный рельеф.",
        "role": "fitness athlete training",
        "outfit_f": "sports bra and leggings",
        "outfit": "tank top and shorts",
        "props": "chalk dust, dumbbells or cable machine",
        "bg": "gym with machines and dramatic backlight",
        "comp": "half", "tone": "cool"
    },
    "Теннис — динамика корта": {
        "desc": "Сервисная линия, ракетка в движении, лёгкий смаз.",
        "role": "tennis player on court",
        "outfit_f": "white tennis dress and visor",
        "outfit": "white tennis kit and headband",
        "props": "racket in hand, tennis balls mid-air motion blur",
        "bg": "hard court with service lines and green windscreen",
        "comp": "half", "tone": "daylight"
    },

    # ===== ПРИКЛЮЧЕНИЯ / ПУТЕШЕСТВИЯ / ПРИРОДА =====
    "Руины — охотник за артефактами": {
        "desc": "Пыльные лучи, древние камни, дух авантюры.",
        "role_f": "tomb raider explorer",
        "role": "tomb raider explorer",
        "outfit_f": "tactical outfit, fingerless gloves, utility belt",
        "outfit": "tactical outfit, fingerless gloves, utility belt",
        "props": "leather straps patina, map tube",
        "bg": "ancient sandstone ruins with sun rays and dust motes",
        "comp": "full", "tone": "warm"
    },
    "Пустынные дюны — исследователь": {
        "desc": "Жар, ветер и бесконечные дюны.",
        "role": "desert explorer",
        "outfit": "scarf, cargo outfit, boots",
        "props": "sand blowing in wind",
        "bg": "sand dunes and canyon under harsh sun",
        "comp": "full", "tone": "warm"
    },
    "Высокие горы — снег и лёд": {
        "desc": "Суровая красота, скайлайн гребня, иней и спиндрифт.",
        "role": "alpinist",
        "outfit": "mountain jacket, harness, crampons",
        "props": "ice axe in hand, spindrift",
        "bg": "snow ridge and blue shadows, cloudy sky",
        "comp": "full", "tone": "cool"
    },
    "Серфер — брызги и солнце": {
        "desc": "Золотая подсветка, капли воды, доска подмышкой или на волне.",
        "role": "surfer athlete on a wave",
        "outfit_f": "black wetsuit",
        "outfit": "black wetsuit",
        "props": "a visible surfboard, water spray, droplets",
        "bg": "ocean wave breaking, golden backlight",
        "comp": "full", "tone": "warm"
    },
    "Арктика — сияние и айсберги": {
        "desc": "Ледяное солнце, айсберги, снежная позёмка.",
        "role_f": "arctic explorer",
        "role": "arctic explorer",
        "outfit_f": "white thermal parka with fur hood, knit beanie and mittens",
        "outfit": "white thermal parka with fur hood, knit beanie and gloves",
        "props": "drifting ice, snow crystals in air",
        "bg": "icebergs and frozen sea, low sun halo, blowing snow",
        "comp": "half", "tone": "cool"
    },
    "Альпы — гламурный отпуск": {
        "desc": "Шале, терраса, лыжи/сноуборд, пар от глинтвейна.",
        "role_f": "alpine fashion vacationer with skis",
        "role": "alpine fashion vacationer with snowboard",
        "outfit_f": "sleek white ski suit, fur-trimmed hood, chic goggles",
        "outfit": "stylish ski jacket and pants, goggles on helmet",
        "props": "skis or snowboard, steam from mulled wine cup",
        "bg": "alpine chalet terrace with snowy peaks and cable cars",
        "comp": "half", "tone": "warm"
    },
    "Париж — кофе, берет и багет": {
        "desc": "Кафе, хаусмановские фасады и башня вдалеке.",
        "role": "parisian street scene character",
        "outfit_f": "striped shirt, red beret, trench, scarf",
        "outfit": "striped shirt, beret, trench, scarf",
        "props": "baguette and croissant in paper bag, café tables",
        "bg": "Eiffel Tower in the distance, Haussmann buildings, café awning",
        "comp": "half", "tone": "daylight"
    },
    "Россия — зимняя сказка": {
        "desc": "Берёзы, сугробы, пар от дыхания, тёплый уют.",
        "role": "person in Russian winter scene",
        "outfit_f": "down coat, ushanka hat, woolen scarf, felt boots",
        "outfit": "down parka, ushanka hat, woolen scarf, felt boots",
        "props": "steam from breath, snowflakes in air, samovar on table",
        "bg": "wooden house with ornate frames and birch trees",
        "comp": "half", "tone": "cool"
    },
    "Деревня — тёплый уют": {
        "desc": "Сельская идиллия: сад, яблоки, деревянный забор.",
        "role": "villager in rustic setting",
        "outfit_f": "linen dress, knitted cardigan, headscarf optional",
        "outfit": "linen shirt, vest",
        "props": "basket with apples, wooden fence, hay",
        "bg": "rural cottage yard with garden; chickens far behind",
        "comp": "half", "tone": "warm"
    },
    "Дикий Запад": {
        "desc": "Пыль, солнце, салун и шляпы-стетсоны.",
        "role": "western hero",
        "outfit_f": "cowgirl hat, suede jacket, boots",
        "outfit": "cowboy hat, leather vest, boots",
        "props": "lasso, spurs, wooden saloon doors, tumbleweed",
        "bg": "old western town street, desert horizon",
        "comp": "full", "tone": "warm"
    },
    "Конная прогулка": {
        "desc": "Лошади — рядом/в кадре, загар и ветер в волосах.",
        "role": "equestrian rider",
        "outfit_f": "riding jacket, breeches, boots, helmet optional",
        "outfit": "riding jacket, breeches, boots, helmet optional",
        "props": "saddle, reins, gentle horse nuzzling",
        "bg": "sunlit field or paddock, wooden fence",
        "comp": "half", "tone": "daylight"
    },

    # ===== ФЭНТЕЗИ / СКАЗКИ / ЛОР =====
    "Эльфы, гномы и тролли": {
        "desc": "Лесной храм, царственная поза, украшения.",
        "role_f": "elven queen in a regal pose",
        "role": "elven king in a regal pose",
        "outfit_f": "flowing emerald gown with golden embroidery, delicate crown",
        "outfit": "ornate armor with emerald cloak, elegant crown",
        "props": "elven jewelry, filigree patterns",
        "bg": "ancient forest temple, god rays in mist",
        "comp": "full", "tone": "candle"
    },
    "Самурай в храме": {
        "desc": "Синто-огни, лакированные доспехи, осенние листья.",
        "role": "samurai warrior in a shrine courtyard",
        "outfit": "lacquered samurai armor, kabuto helmet",
        "props": "katana visible in hand",
        "bg": "Shinto shrine with lanterns, falling leaves",
        "comp": "full", "tone": "warm"
    },
    "Средневековый рыцарь": {
        "desc": "Полированный латный доспех и штандарты.",
        "role": "medieval knight",
        "outfit": "full plate armor with cloak",
        "props": "sword and shield",
        "bg": "castle tournament yard with banners and dust",
        "comp": "full", "tone": "daylight"
    },
    "Пират капитан": {
        "desc": "Треуголка, мокрая палуба, чайки и шквалы.",
        "role_f": "pirate captain",
        "role": "pirate captain",
        "outfit_f": "tricorn hat, leather corset, white shirt",
        "outfit": "tricorn hat, leather vest, white shirt",
        "props": "cutlass, rope rigging, sea spray",
        "bg": "ship deck in storm, sails and rigging",
        "comp": "full", "tone": "cool"
    },
    "Древняя Греция": {
        "desc": "Мраморные колоннады, оливы и бирюза воды.",
        "role_f": "ancient Greek goddess",
        "role": "ancient Greek hero",
        "outfit_f": "white chiton with gold trim, diadem",
        "outfit": "white chiton with gold trim, laurel wreath",
        "props": "gold accessories",
        "bg": "white marble colonnade, statues, olive trees, turquoise pool",
        "comp": "half", "tone": "warm"
    },
    "Матерь драконов": {
        "desc": "Полёт на реальном драконе — чешуя, облака, шквал ветра.",
        "role_f": "dragon rider heroine",
        "role": "dragon rider hero",
        "outfit_f": "wind-swept cloak, leather bracers",
        "outfit": "leather armor, cloak",
        "props": "majestic dragon clearly visible, reins, cloud streaks",
        "bg": "dramatic sky above mountains",
        "comp": "full", "tone": "cool"
    },
    "Магическая школа": {
        "desc": "Мантии, шрамы судьбы, волшебные палочки и залы с парящими свечами.",
        "role": "young wizard or witch",
        "outfit": "wizard robe, scarf in house colors",
        "props": "wand with sparks, floating candles, spell books, owl far away",
        "bg": "grand gothic hall or castle corridor with moving portraits",
        "comp": "half", "tone": "candle"
    },
    "Хоббит": {
        "desc": "Круглая дверь, холмы, чайник, трубка — уют Шира.",
        "role": "hobbit in the Shire",
        "outfit": "vest, shirt, bare feet look",
        "props": "round green door, tiny garden tools, loaf of bread",
        "bg": "lush rolling hills, small burrow houses",
        "comp": "half", "tone": "daylight"
    },
    "Красная Шапочка": {
        "desc": "Красный плащ, корзинка, сказочный лес (безопасно).",
        "role": "little red riding hood character",
        "outfit_f": "red hooded cloak, rustic dress",
        "outfit": "red hooded cloak, rustic outfit",
        "props": "basket with pastries, flowers, distant wolf silhouette (non-threatening)",
        "bg": "mossy forest path with god rays",
        "comp": "half", "tone": "candle"
    },
    "Белоснежка": {
        "desc": "Сказочная невинность: яблоко, лесные друзья, замок вдали.",
        "role": "snow white inspired character",
        "outfit_f": "classic fairytale dress (modest), headband",
        "outfit": "storybook prince attire (optional)",
        "props": "red apple, tiny birds, woodland animals hints",
        "bg": "storybook forest clearing, castle in distance",
        "comp": "half", "tone": "daylight"
    },
    "Спящая красавица": {
        "desc": "Лепестки роз, заколдованный сад, нежный свет.",
        "role": "sleeping beauty inspired character",
        "outfit_f": "elegant pastel gown (modest)",
        "outfit": "royal attire (optional)",
        "props": "rose vines, soft petals in air",
        "bg": "enchanted garden with ivy-covered stone",
        "comp": "half", "tone": "warm"
    },

    # ===== КИНО / КОМИКСЫ / ПОП-КУЛЬТУРА (вдохновение) =====
    "Готэм-ночь (Бэтмен-вайб)": {
        "desc": "Ночной мегаполис, прожектора, готическая крыша.",
        "role": "dark vigilante silhouette style",
        "outfit": "matte suit vibe (no logos)",
        "props": "dramatic cape motion, rooftop gargoyles, signal in clouds",
        "bg": "gothic city skyline in rain",
        "comp": "half", "tone": "noir"
    },
    "Халк-эффект": {
        "desc": "Мощная зеленая трансформация: огромное мускулистое тело, зеленная кожа, разорванная одежда.",
        "role": "hulk-style transformed hero with green skin and massive body",
        "outfit": "torn purple pants (classic Hulk reference)",
        "props": "veins popping, cracked ground under feet, green skin texture",
        "bg": "urban destruction or lab wreckage",
        "comp": "half", "tone": "cool",
        "allow_body_change": True,
        "required_attributes": ["green skin", "massive muscular body", "hulk-like proportions", "green face"]
    },
    "Женщина-Кошка": {
        "desc": "Лаконичный кошачий силуэт, городское небо.",
        "role": "cat-burglar heroine",
        "outfit_f": "sleek catsuit (modest), goggles",
        "outfit": "sleek dark outfit, mask",
        "props": "whip silhouette, rooftop antennae",
        "bg": "neon skyline with moon",
        "comp": "half", "tone": "neon"
    },
    "Харли-Квинн": {
                                       "desc": "Аутентичный образ Харли Квинн: красно-черное комбинезон, грим арлекина, бейсбольная бита, хулиганский стиль.",
                                       "role": "harley quinn character with full authentic appearance",
                                       "outfit_f": "red and black harlequin jumpsuit, fishnet stockings, combat boots",
                                       "outfit": "harley quinn inspired outfit with red and black colors",
                                       "props": "baseball bat, giant mallet, playing cards, confetti explosion, ""Daddy's Little Monster"" shirt visible",
                                       "bg": "graffiti-covered asylum or carnival alley",
                                       "comp": "half", "tone": "neon",
                                       "required_attributes": ["classic harley quinn makeup", "red and black color scheme", "harlequin pattern", "baseball bat or mallet"]
                                   },
    "Супергерой": {
        "desc": "Комиксовая динамика, контровой свет, гоночные неоны.",
        "role": "superhero in action pose",
        "outfit_f": "form-fitting suit with emblem (non-revealing)",
        "outfit": "form-fitting suit with emblem",
        "props": "speed lines, comic halftone accents",
        "bg": "night city skyline with spotlights",
        "comp": "full", "tone": "neon"
    },

    # ===== SCI-FI =====
    "Космический скафандр — EVA": {
        "desc": "Хард sci-fi: шлем с отражениями, стыковочный отсек.",
        "role": "astronaut",
        "outfit": "realistic EVA spacesuit",
        "props": "helmet reflections, suit details",
        "bg": "starfield and spaceship hangar",
        "comp": "full", "tone": "cool"
    },
    "Космопилот на мостике": {
        "desc": "Пульт, индикаторы, режим гиперпрыжка.",
        "role": "starship pilot on the bridge",
        "outfit": "flight suit, helmet under arm",
        "props": "control panels with glowing indicators",
        "bg": "spaceship bridge interior",
        "comp": "half", "tone": "cool"
    },
    "Киборг": {
        "desc": "Половина лица повреждена — под кожей блестящий металлический эндоскелет, красный кибер-глаз, детали из титана и проводов. Атмосфера фильма о будущем: дым, холодный свет, напряжение.",
        "role": "half-terminator portrait, realistic cinematic sci-fi",
        "outfit": (
            "tactical or futuristic outfit with metallic seams, "
            "visible armored collar and cable harness on the neck"
        ),
        "props": (
            "mandatory detailed half-face endoskeleton visible under torn skin, "
            "metallic skull structure, hydraulic pistons and micro servos, "
            "chrome cheekbones, glowing red cybernetic eye, "
            "thin steel cables along the jawline, exposed neck wiring, "
            "small bolts and seams in the skin, steam coming from the wound, "
            "soft sparks and reflections on the metal surface"
        ),
        "bg": (
            "futuristic research corridor or destroyed lab, "
            "flickering red lights, smoke haze, volumetric backlight, "
            "broken glass, scattered sparks and glowing wires"
        ),
        "comp": "closeup",
        "tone": "dramatic",
        "cyborg_split": True,
        "cyborg_mode": "mask",
        "cyborg_side": "right",
        "cyborg_feather": 64,
        "force_keywords": [
            "metal on the face", 
            "cybernetic implants visible", 
            "half mechanical skull", 
            "glowing red eye", 
            "exposed circuits", 
            "chrome cheek and jaw"
        ]
    },

    # ===== НОЧНЫЕ / УЖАСЫ / ГОТИКА =====
    "Вампирский бал": {
        "desc": "Готическая элегантность с обязательными вампирскими чертами: острые клыки и красные глаза. Атмосфера – мистический бал под лунным светом и свечами.",
        "role": "vampire aristocrat attending a gothic masquerade ball",
        "outfit_f": (
            "dark velvet or lace gown with corset, silver jewelry, "
            "delicate gothic accessories"
        ),
        "outfit": (
            "black or crimson tailcoat with embroidered collar, cravat, "
            "leather gloves, subtle sheen fabric"
        ),
        "props": (
            "slightly elongated canine teeth visible even when lips closed, "
            "crimson glowing eyes reflecting candlelight, "
            "goblet with dark red drink, candelabra, roses, wax drips"
        ),
        "bg": (
            "lavish gothic ballroom illuminated by candles, "
            "crimson drapes, moonlight through stained glass, "
            "mist and golden dust in the air"
        ),
        "comp": "half",
        "tone": "noir",
        "required_attributes": [
            "sharp elongated fangs",
            "glowing crimson eyes",
            "pale aristocratic skin"
        ],
        "force_keywords": [
            "sharp visible vampire fangs",
            "slightly open mouth showing fangs",
            "glowing red eyes under candlelight"
        ]
    },
    "Зомби-апокалипсис (кино)": {
        "desc": "Выжженный мир после катастрофы. Вы — заражённый человек: бледная кожа, стеклянный взгляд, следы деградации, разорванная одежда. Атмосфера — киношный хоррор, напряжение и пыль.",
        "role": "infected survivor in cinematic apocalypse setting",
        "outfit": (
            "tattered, dirty, weathered clothing with faded stains, "
            "rough texture, survival accessories, worn boots"
        ),
        "props": (
            "pale uneven skin tone, dark eye shadows, cracked lips, "
            "slow posture, empty eyes, dust and grime on the face, "
            "fog and smoke in the background, cinematic debris"
        ),
        "bg": (
            "destroyed city street with dust clouds, abandoned vehicles, "
            "dim orange light through smoke, cinematic haze"
        ),
        "comp": "half",
        "tone": "noir",
        "required_attributes": [
            "pale skin", 
            "dark eye circles", 
            "lifeless gaze", 
            "weathered clothing"
        ]
    },
    "Ведьма — чары и луна": {
        "desc": "Травы, чары, ночная поляна и огромная луна.",
        "role": "mystic witch",
        "outfit_f": "dark boho dress, hat",
        "outfit": "dark cloak",
        "props": "spellbook, candles, ravens in distance",
        "bg": "forest clearing with moon halo",
        "comp": "half", "tone": "candle"
    },
    "Монашка": {
        "desc": "Сдержанный религиозный вайб, драматический свет.",
        "role": "mystic nun style",
        "outfit_f": "habit (modest)",
        "outfit": "monastic cloak",
        "props": "candle, old book, stained glass glow",
        "bg": "old chapel interior",
        "comp": "half", "tone": "noir"
    },
    "Клоун — цирковой сюр": {
        "desc": "Сюрреалистичный грим, шары, разноцветные огни.",
        "role": "circus clown (friendly)",
        "outfit": "playful costume (modest)",
        "props": "balloons, confetti, stage curtains",
        "bg": "circus ring with spotlights",
        "comp": "half", "tone": "warm"
    },

    # ===== РЕТРО / СЦЕНЫ ЖИЗНИ =====
    "Золотые 50-е": {
        "desc": "Неон «DINER», винил, пастель, блеск хрома.",
        "role": "50s diner scene character",
        "outfit_f": "polka-dot dress or waitress vibe (modest)",
        "outfit": "letterman jacket or retro shirt",
        "props": "milkshake, jukebox, checker floor",
        "bg": "retro diner booth and neon sign",
        "comp": "half", "tone": "warm"
    },
    "Бал": {
        "desc": "Хрустальные люстры, шелк, вальс и золотая подсветка.",
        "role": "ball attendee",
        "outfit_f": "evening ball gown (modest)",
        "outfit": "tailcoat",
        "props": "chandeliers, marble columns, soft bokeh",
        "bg": "grand ballroom",
        "comp": "half", "tone": "warm"
    },
    "Романтик 🥰": {
        "desc": "Эстетичный boudoir: мягкий свет, воздушные ткани, без откровенностей.",
        "role": "tasteful lingerie scene",
        "outfit_f": "delicate lingerie set with robe (modest, tasteful)",
        "outfit": "silk robe / loungewear (modest)",
        "props": "sheer curtains, soft bed linen, candle glow",
        "bg": "cozy bedroom with fairy lights",
        "comp": "half", "tone": "warm"
    },
    "Свадьба": {
        "desc": "Белая арка, лепестки в воздухе, коробочка с кольцом.",
        "role_f": "bride in elegant wedding dress",
        "role": "groom in classic tuxedo",
        "outfit_f": "white lace wedding gown, veil, bouquet",
        "outfit": "black tuxedo with boutonnière",
        "props": "flower petals in air, ring box",
        "bg": "sunlit ceremony arch with flowers",
        "comp": "half", "tone": "warm"
    },
    "Детство": {
        "desc": "Игрушки, пастель, тёплый дневной свет.",
        "role": "child portrait in playful setting",
        "outfit_f": "cute cardigan, skirt with suspenders, bow headband",
        "outfit": "cute sweater and suspenders",
        "props": "teddy bear, balloons, crayons, blocks",
        "bg": "cozy kids room with garlands",
        "comp": "closeup", "tone": "daylight",
        "allow_age_change": True
    },

    # ===== ТРАНСПОРТ / ДРАЙВ =====
    "Суперкары — хром и скорость": {
        "desc": "Автосъёмка: блики лака, тоннели, световые дорожки.",
        "role": "driver / model with car",
        "outfit": "smart casual / leather jacket",
        "props": "sleek supercar body, wheel reflections",
        "bg": "underground parking or city tunnel",
        "comp": "half", "tone": "cool"
    },
    "Мото-культура — байк и дым": {
        "desc": "Кастомы, кожа, лёгкий burnout-дым.",
        "role": "biker style",
        "outfit": "leather jacket, boots, gloves",
        "props": "motorcycle close, headlight flare, chain hints",
        "bg": "industrial yard or night street",
        "comp": "half", "tone": "noir"
    },

    # ===== ПАРОДИИ / СЦЕНЫ ФИЛЬМОВ (вдохновение) =====
    "Красотка — красное платье": {
        "desc": "Гламур из 90-х: красное платье, белые перчатки, роскошный отель.",
        "role": "glamorous heroine 90s vibe",
        "outfit_f": "iconic red evening dress (modest), gloves",
        "outfit": "sleek suit",
        "props": "pearl necklace vibe, hotel lobby",
        "bg": "grand hotel entrance or staircase",
        "comp": "half", "tone": "warm"
    },

    # ===== НОЧНЫЕ / КЛУБ =====
    "Дискотека — лазеры и туман": {
        "desc": "Клубный вайб: лазеры, дым, зеркальный шар.",
        "role": "club night dancer",
        "outfit": "sparkly party wear",
        "props": "laser beams, fog, mirror ball bokeh",
        "bg": "crowded dance floor",
        "comp": "half", "tone": "neon"
    },

    # ===== ТЁМНЫЕ / СОЦИАЛЬНЫЕ =====
    "Тюрьма — холодный коридор": {
        "desc": "Металл, номера камер, холодный свет.",
        "role": "inmate or visitor (neutral)",
        "outfit": "plain clothing or jumpsuit",
        "props": "bars, metal door, number plate",
        "bg": "prison corridor with cold light",
        "comp": "half", "tone": "cool"
    },
    "Гранж-сквот — рваные постеры": {
        "desc": "Гранж, стрит-арт, лёгкая небрежность.",
        "role": "grunge street look",
        "outfit": "layered worn clothes, beanie",
        "props": "peeling posters, graffiti, paint drips",
        "bg": "abandoned entrance with tags",
        "comp": "half", "tone": "noir"
    },

    # ===== ТРАНСФОРМАЦИИ =====
    "Старость": {
        "desc": "Деликатное «состаривание» как стилизация.",
        "role": "same person aged up",
        "outfit": "same wardrobe vibe",
        "props": "soft silver hair hints, gentle wrinkles",
        "bg": "neutral portrait backdrop",
        "comp": "closeup", "tone": "daylight",
        "allow_age_change": True
    },
    "Молодость": {
        "desc": "Лёгкое «омоложение» как стилизация.",
        "role": "same person aged down",
        "outfit": "same wardrobe vibe",
        "props": "smoother skin microfeatures",
        "bg": "neutral portrait backdrop",
        "comp": "closeup", "tone": "daylight",
        "allow_age_change": True
    },
    "Пухлый герой": {
                                       "desc": "Реалистичная стилизация полноты: человек за столом с едой, мягкие складки, естественные пропорции.",
                                       "role": "heavier person enjoying meal at table",
                                       "outfit": "comfortable casual clothes that fit larger body",
                                       "props": "food on table (pizza, burger, or home meal), drink, napkins",
                                       "bg": "cozy kitchen or restaurant setting",
                                       "comp": "half", "tone": "warm",
                                       "allow_body_change": True,
                                       "required_attributes": ["plus-size body", "eating at table", "realistic body folds", "natural weight distribution"]
    },

    # ===== ИНФО/РОЛИ =====
    "Домашний уют": {
        "desc": "Солнечная гостиная, порядок и зелень.",
        "role": "housekeeper",
        "outfit_f": "modest house dress and apron",
        "outfit": "utility shirt and apron",
        "props": "feather duster, neatly stacked towels",
        "bg": "sunlit living room with plants",
        "comp": "half", "tone": "daylight"
    },
    "Учитель": {
        "desc": "Классная доска, схемы, маркер/мел.",
        "role": "teacher in classroom",
        "outfit_f": "blouse and skirt or pantsuit",
        "outfit": "shirt and chinos or suit",
        "props": "chalk or marker, books",
        "bg": "blackboard with formulas/maps, desks",
        "comp": "half", "tone": "daylight"
    },
    "Медсестра": {
        "desc": "Игровой вайб без откровенности.",
        "role_f": "nurse in playful themed outfit",
        "role": "male nurse in playful themed outfit",
        "outfit_f": "short uniform skirt, stockings, nurse cap (tasteful)",
        "outfit": "white coat over scrubs (tasteful)",
        "props": "stethoscope, clipboard, ID badge",
        "bg": "hospital corridor with soft bokeh",
        "comp": "half", "tone": "warm"
    },
    "Программист": {
        "desc": "Код на экранах, RGB-клавиатура, ночь за окном.",
        "role": "software engineer at desk",
        "outfit": "hoodie or tee, headphones",
        "props": "code on monitors, RGB keyboard",
        "bg": "dual-monitor setup with city glow",
        "comp": "half", "tone": "cool"
    },

    # ===== ДИЗНЕЕВСКИЕ ВАЙБЫ / PIXAR (стилизация) =====
    "Светлая сказка": {
        "desc": "Сказочная палитра, крупные выразительные глаза.",
        "role": "fairy-tale protagonist",
        "outfit_f": "pastel dress with subtle sparkles",
        "outfit": "storybook outfit with cape",
        "props": "tiny twinkles around, soft vignette",
        "bg": "storybook castle and garden",
        "comp": "half", "tone": "daylight"
    },
    "Пиксар-семья": {
        "desc": "Мягкие формы, кинематографичный rim light.",
        "role": "family animation character",
        "outfit": "casual friendly clothes",
        "props": "subsurface scattering skin hint, rim light",
        "bg": "friendly suburban street or cozy room",
        "comp": "half", "tone": "warm"
    },

    # ===== ДЖУНГЛИ / ЭКШЕН =====
    "Джунгли — Тарзан-вайб": {
        "desc": "Густая зелень, туман у земли, дикие звери на дистанции.",
        "role_f": "jungle adventurer",
        "role": "jungle adventurer",
        "outfit_f": "leather jungle top and skirt, rope belt",
        "outfit": "leather jungle outfit, rope belt",
        "props": "vines, soft mist, crocodile/snake/panther nearby (safe)",
        "bg": "dense tropical jungle, waterfalls and sunbeams",
        "comp": "full", "tone": "warm"
    },
    "Хаос-кинематограф": {
        "desc": "Всё летит, но герой спокоен — драматический эпик.",
        "role": "hero in cinematic disaster scene",
        "outfit_f": "modern streetwear with dust marks",
        "outfit": "modern streetwear with dust marks",
        "props": "embers, flying papers, cracked glass",
        "bg": "burning house and collapsing structures",
        "comp": "full", "tone": "noir"
    },
    "Контакт НЛО — лучи и пыль": {
        "desc": "Парящие тарелки, лучи и загадочная пыль.",
        "role": "person confronted by hovering UFOs",
        "outfit_f": "sleek sci-fi coat",
        "outfit": "sleek sci-fi coat",
        "props": "tractor beams, floating debris",
        "bg": "night field with moody clouds",
        "comp": "full", "tone": "cool"
    },
    "Фридайвер — синие бездны": {
        "desc": "Голубая толща, лучи сверху, пузыри.",
        "role_f": "freediver underwater",
        "role": "freediver underwater",
        "outfit_f": "apnea wetsuit without tank, long fins, mask",
        "outfit": "apnea wetsuit without tank, long fins, mask",
        "props": "air bubbles, sunbeams, small fish",
        "bg": "deep blue water with rocky arch or reef",
        "comp": "full", "tone": "cool"
    },

    # ===== КОСТЮМНЫЕ / ОБРАЗЫ =====
    "Кукла в коробке": {
        "desc": "Ты — кукла в бренд-коробке: прозрачное окно, стикеры, аксессуары.",
        "role": "boxed doll character",
        "outfit": "theme outfit matching box art (modest)",
        "props": "plastic-like gloss, accessory tray, printed slogans",
        "bg": "toy store display box interior",
        "comp": "half", "tone": "daylight"
    },
    "Зена-королева воинов": {
        "desc": "Амазонка с круглыми латами, браслеты, боевой плащ.",
        "role_f": "warrior princess style",
        "role": "warrior style",
        "outfit_f": "leather armor dress, bracers",
        "outfit": "leather armor and cloak",
        "props": "round buckler, sword, braid in hair",
        "bg": "ancient battlefield ridge",
        "comp": "full", "tone": "warm"
    }
}

# =========================
#    ДОБАВКИ/УЛУЧШЕНИЯ
# =========================
# Дополнительные бусты сцены (визуальные детали)
THEME_BOOST = {
    "Пират капитан": "rope rigging, storm clouds, wet highlights on wood, sea spray, gulls",
    "Древняя Греция": "ionic capitals, olive trees, turquoise water reflections, gold trim accents",
    "Ночной неон — мокрый асфальт": "rain droplets on lens, colored reflections on wet asphalt",
    "Фильм-нуар — жалюзи и дым": "venetian blinds light pattern, cigarette smoke curling, deep black shadows",
    "Руины — охотник за артефактами": "floating dust motes in sunrays, chipped sandstone blocks, leather straps patina",
    "Высокие горы — снег и лёд": "spindrift blown by wind, crampon scratches on ice, distant ridge line",
    "Готэм-ночь (Бэтмен-вайб)": "roof gargoyles, rain streaks, spotlight haze",
    "Серфер — брызги и солнце": "rimlight on water droplets, sun flare",
    "Бал": "golden bokeh, polished parquet reflections",
    "Арктика — сияние и айсберги": "diamond-dust glitter, low sun halo, frost crystals on clothing",
    "Альпы — гламурный отпуск": "sunflare off snow, chalet wood textures, gondola cables in distance",
    "Париж — кофе, берет и багет": "café chalk menu board, wrought iron balcony rails, warm bakery glow",
    "Джунгли — Тарзан-вайб": "god rays through canopy, wet leaf speculars, ground mist",
    "Детство": "soft pastel garlands, shallow dof sparkles, gentle vignette",
    "Свадьба": "fairy lights bokeh, soft veil translucency",
    "Хаос-кинематограф": "embers, paper scraps, dramatic smoke, slight camera shake feel",
    "Контакт НЛО — лучи и пыль": "volumetric beams, dust motes, faint radio-glitch halation",
    "Фридайвер — синие бездны": "caustic light patterns, particulate backscatter, gentle blue gradient",
    "Деревня — тёплый уют": "warm wood patina, sun dust in air, linen texture details",
    "Россия — зимняя сказка": "crisp breath vapor, snow sparkle, frosty window details",
    "Теннис — динамика корта": "chalk dust from lines, motion blur of ball strings",
    "Конная прогулка": "mane motion highlights, dust sparkles in backlight",
    "Космический скафандр — EVA": "helmet micro-scratches, HUD reflections",
    "Киборг": "subsurface skin vs brushed metal micro-contrast",
    "Вампирский бал": "sharp fangs detail, crimson eye glow, pale complexion, gothic architecture shadows",
    "Золотые 50-е": "checker floor reflections, chrome sparkle",
    "Кукла в коробке": "plastic gloss, cardboard print grain",
    "Зена-королева воинов": "wind-torn cloak edges, sun flare",
    "Гранж-сквот — рваные постеры": "torn poster edges, paint drips",
    "Мото-культура — байк и дым": "headlight bloom, tire smoke whirls",
    "Суперкары — хром и скорость": "light trails, glossy reflections",
    "Готика: Ведьма — чары и луна": "moon halo, drifting sparks",
    "Халк-эффект": "green skin texture, massive muscle definition, torn clothing fibers, destruction debris",
    "Пухлый герой": "appetizing food details, table setting, natural body language, comfortable seating",
    "Харли-Квинн": "diamond pattern details, smudged makeup, weapon props, chaotic confetti bursts",
    "Зомби-апокалипсис (кино)": "decaying skin details, blood splatters, tattered clothing, post-apocalyptic debris"
}

# Настройки «сценовой» направляющей (чуть выше — меньше уводит лицо)
SCENE_GUIDANCE = {
    "Джунгли — Тарзан-вайб": 3.2,
    "Контакт НЛО — лучи и пыль": 3.2,
    "Хаос-кинематограф": 3.2,
    "Фридайвер — синие бездны": 3.0,
    "Арктика — сияние и айсберги": 3.2,
    "Детство": 3.0,
    "Самурай в храме": 3.2,
    "Средневековый рыцарь": 3.2,
    "Космический скафандр — EVA": 3.2,
    "Киборг": 3.2,
    "Вампирский бал": 3.2,
    "Зомби-апокалипсис (кино)": 3.3,
    "Монашка": 3.0,
    "Клоун — цирковой сюр": 3.0,
    "Кукла в коробке": 3.0,
    "Готэм-ночь (Бэтмен-вайб)": 3.2,
    "Халк-эффект": 3.5,
    "Женщина-Кошка": 3.2,
    "Харли-квинн": 3.4,
    "Магическая школа": 3.2,
    "Хоббит": 3.0,
    "Матерь драконов": 3.0,
    "Дикий Запад": 3.2,
    "Конная прогулка": 3.0,
    "Пухлый герой": 3.0
}

# Сцены, где чаще уводит лицо / агрессивная стилизация — держим под контролем
RISKY_PRESETS = set(SCENE_GUIDANCE.keys())

# =========================
#       КАТЕГОРИИ
# =========================
STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Портреты и Мода": [
        "Портрет у окна", "85 мм",
        "Бьюти-студия", "Кинопортрет Рембрандта",
        "Фильм-нуар", "Стритвэр мегаполис",
        "Вечерний выход", "Бизнес-портрет C-suite",
        "Ночной неон"
    ],
    "Профессии и Спорт": [
        "Доктор у палаты", "Хирург в операционной",
        "Шеф-повар в огне", "Учёный в лаборатории",
        "Боксёр на ринге", "Фитнес-зал — контровый свет",
        "Теннис — динамика корта"
    ],
    "Приключения и Путешествия": [
        "Руины — охотник за артефактами", "Пустынные дюны — исследователь",
        "Высокие горы — снег и лёд", "Серфер — брызги и солнце",
        "Арктика — сияние и айсберги", "Альпы — гламурный отпуск",
        "Париж — кофе, берет и багет", "Россия — зимняя сказка",
        "Деревня — тёплый уют", "Дикий Запад",
        "Конная прогулка"
    ],
    "Фэнтези и Сказки": [
        "Эльфы, гномы и тролли", "Самурай в храме",
        "Средневековый рыцарь", "Пират капитан",
        "Древняя Греция", "Матерь драконов",
        "Магическая школа", "Хоббит",
        "Красная Шапочка", "Белоснежка",
        "Спящая красавица"
    ],
    "Кино и Комиксы (вдохновение)": [
        "Готэм-ночь (Бэтмен-вайб)", "Халк-эффект",
        "Женщина-Кошка", "Харли-Квинн",
        "Супергерой"
    ],
    "Sci-Fi": [
        "Космический скафандр — EVA", "Космопилот на мостике",
        "Киборг", "Контакт НЛО — лучи и пыль"
    ],
    "Ночные / Готика / Ужасы": [
        "Вампирский бал", "Зомби-апокалипсис (кино)",
        "Ведьма — чары и луна", "Монашка",
        "Клоун — цирковой сюр"
    ],
    "Ретро и Сцены жизни": [
        "Золотые 50-е", "Бал",
        "Романтик 🥰", "Свадьба",
        "Детство", "Красотка — красное платье"
    ],
    "Транспорт и Драйв": [
        "Суперкары — хром и скорость", "Мото-культура — байк и дым"
    ],
    "Костюмные и Ролевые": [
        "Кукла в коробке", "Зена-королева воинов",
        "Домашний уют", "Учитель",
        "Медсестра", "Программист"
    ],
    "Трансформации": [
        "Старость", "Молодость",
        "Пухлый герой"
    ]
}
