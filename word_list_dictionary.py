

#%%
replacement_dict = {
    "allœh": "Allah",
    "ramadhœn": "ramadhan",
    "ibrœheem": "ibrahim",
    "abraham": "ibrahim",
    "shawwœl": "shawwal",
    "kaffœrah": "kaffarah",
    "ghœfir": "ghafir",
    "MunŒfiq´n": "munafiqun",
    "ôuzzœ": "uzza",
    "barœõah": "baraah",
    "wœqiôah": "waqiah",
    "minœ": "mina",
    "shuôarœõ": "shuara",
    "salœmu": "salamu",
    "îijjah": "hijjah",
    "tabœrak": "tabarak",
    "îajj": "Hajj",
    "sœõibah": "saibah",
    "iúrœm": "ihram",
    "subúœnahu": "subhanahu",
    "ôabdullœh": "Abdullah",
    "sulaymœn": "sulayman",
    "hœshim": "hashim",
    "rakôahs": "rakahs",
    "miôrœj": "miraj",
    "saôeed": "saeed",
    "KŒfir´n": "kafirun",
    "kœfir´n": "kafirun",
    "kœfir": "kafir",
    "mumtaúanah": "mumtahanah",
    "îœkim": "Hakim",
    "ôarsh": "Arsh",
    "YaÕj´j": "Yajuj",
    "MaÕj´j": "Majuj",
    "îœrith": "Harith",
    "Ban´": "Banu",
    "Mu‹‹alib": "Muttalib",
    "BukhŒr¥": "Bukhari",
    "janœbah": "janabah",
    "firôawn": "firawn",
    "inshirœh": "inshirah",
    "‹Œgh´t": "taghut",
    "êœliú": "Salih",
    "Tham´d": "Thamud",
    "MuÕmin´n": "Muminun",
    "muõmin": "mumin",
    "-DhŒriyŒt": "Dhariyat",
    "iúsœn": "ihsan",
    "muúarram": "muharram",
    "jumuôah": "jumuah",
    "J´diyy": "Judiyy",
    "ôumar": "Umar",
    "wa§¥lah": "wasilah",
    "îarŒm": "haram",
    "mashôar": "mashar",
    "taghœbun": "taghabun",
    "isrœõeel": "israeel",
    "ÔAad": "Aad",
    "ÔAa§": "Aas",
    "sabœõ": "Saba",
    "islœmi": "islami",
    "zaqq´m": "zaqqum",
    "DhŒriyŒt": "Dhariyat",
    "HŒr´t": "Harut",
    "MŒr´t": "Marut",
    "thihŒr": "thihar",
    "HŒr´n": "Harun",
    "hœr": "har",
    "ilyœseen": "ilyaseen",
    "s´rah": "surah",
    "qurõn": "quran",
    "qurõœn": "quran",
    "al-fœtiúah": "al-fatihah",
    "li ôimrœn": "Ali Imrœn",
    "Khal¥fah": "Khalifah",
    "juzõ": "juz",
    "raúeem": "raheem",
    "raúmœnir": "rahmanir",
    "bismillœhir": "bismillahir",
    "taôœlœ": "taala",
    "islœm": "islam",
    "zakœh": "zakah",
    "muúammad": "muhammad",
    "qurõœn": "quran",
    "nisœõ": "nisa",
    "ÔImrŒn": "Imran",
    "imrœn": "Imran",
    "anfœl": "anfal",
    "y´nus": "yunus",
    "LŒm": "Lam",
    "isrœõ":    "isra",
    "hœ": "ha",
    "naúl": "nahl",
    "aúzœb": "ahzab",
    "y´suf":"yusuf",
    "mœõidah":"maidah",
    "anôœm": "anam",
    "aôrœf": "araf",
    "h´d": "hud",
    "anbiyœõ": "anbiya",
    "shuôayb": "shuayb",
    "furqœn": "furqan",
    "a§-êŒffŒt": "as-saffat",
    "êœffœt": "saffat",
    "ÔAnkab´t": "Ankabut",
    "kaôbah": "kabah",
    "ôumrah": "umrah",
    "îajjŒj":            "hajjaj",
    "hajjœj":            "hajjaj",
    "úajj": "hajj",
    "Fu§§ilat": "Fussilat",
    "îijr": "hijr",
    "úad¥th": "hadith",
    "maúf´th": "mahfuth",
    "raúmœn": "rahman",
    "sabaõ": "saba",
    "n´r": "nur",
    "êœd": "Saad",
    "r´m": "rum",
    "aúqœf": "ahqaf",
    "Sh´rŒ": "Shura",
    "luqmœn": "luqman",
    "raôd": "rad",
    "lawú": "lawh",
    "ôazeez": "azeez",
    "qœf": "qaf",
    "îadeed": "Hadeed",
    "mujœdilah": "mujadilah",
    "jœthiyah": "jathiyah",
    "§aúŒbah": "sahabah",
    "nœs": "nas",
    "fatú": "fath",
    "taúreem": "tahreem",
    "îujurœt": "hujurat",
    "qiyœmah": "qiyamah",
    "a‹-$Œriq": "at-tariq",
    "$Œriq": "tariq",
    "AÔlŒ": "ala",
    "nœziôœt": "naziat",
    "aúmad": "ahmad",
    "mumtaúinah": "mumtahinah",
    "insœn": "insan",
    "mursalœt": "mursalat",
    "nabaõ": "naba",
    "IkhlŒ§": "ikhlas",
    "ôabasa":     "abasa",
    "sharú":    "sharh",
    "n´ú":      "nuh",
    "bur´j":    "buruj",
    "ôalaq":    "alaq",
    "Infi‹Œr":    "infitar",
    "takœthur":    "takathur",
    "uúud":    "uhud",
    "qœriôah":    "qariah",
    "aadiyœt":    "aadiyat",
    "dhuúœ":    "dhuha",
    "mœô´n":    "maun",
    "taqwœ":    "taqwa",
    "ôuthmœn":    "uthman",
    "qœr´n":    "qarun",

    "$Œ HŒ":    "Ta ha ",
    "FŒ‹ir":    "Fatir",
    "YŒ Seen":    "Ya Seen",
    "MaÔŒrij":    "maarij",
    "$alŒq":    "talaq",
    "yat":    "ayat",
    "Mu‹affifeen":    "Mutaffifeen",
    "§adaqah":    "sadaqah",
    "Khudr¥":    "Khudri",
    "ismœôeel":        "ismaeel",
    "îashr":    "Hashr",
    "îœqqah":    "Haqqah",
    "inshiqœq":      "inshiqaq",
    "êaff":      "Saff",
    "mœn":        "man",
    "ôiddah":    "iddah",
    "rœôinœ":    "raina",
    "õishah":    "aishah",
    "îaf§ah":        "hafsah",
    "qaôdah":    "qadah",
    "SŒmir¥":      "samiri",
    "sœmir":      "samir",
    "AsbŒ‹":      "Asbat",
    "îaram":      "Haram",
    "tab´k":      "tabuk",
    "mujœhideen":   "mujahideen",
    "Kurs¥":      "Kursi",
    "masô´d":       "masud",
    "ôilliyy´n":    "illiyyun",
    "iqraõ":       "iqra",
    "ôayn":       "ayn",
    "ôalaykum":       "alaykum",
    "dœw´d":       "dawud",
    "tœbiô´n":       "tabiun",
    "§uúuf":       "suhuf",
    "êaúeeú":       "saheeh",
    "mu§úaf":       "mushaf",
    "Tirmidh¥":       "Tirmidhi",
    "zab´r":       "zabur",
    "mœlik":       "malik",
    "ôaqeedah":       "aqeedah",
    "muntadœ":       "muntada",
    "ahl":       "ahl",
    "dahr":       "dahr",
    "daôwah":       "dawah",
    "$uwŒ":       "tuwa",
    "îudaybiyyah":    "hudaybiyyah",
    "unthurnœ":    "unthurna",
    "mannœô":      "manna",
    "muôallœ":      "mualla",
    "ôurwah":      "urwah",
    "jamœôah":      "jamaah",
    "êœkhkhah":      "sakhkhah",
    "manœt":      "manat",
    "awlœ":      "awla",
    "wudh´õ":      "wudhu",
    "sçrahs":      "surahs",
    "raúmah":      "rahmah",
    "ôaffœn":      "affan",
    "dajjœl":      "dajjal",
    "shawwœl":    "shawwal",
    "shawwœl":    "shawwal",
    "shawwœl":    "shawwal",
    "shawwœl":    "shawwal",
    "rœfiô":      "rafi",
    "ayy´b":      "ayyub",
    "úœm":      "ham",
    "yaôq´b":      "yaqub",
    "zakariyyœ":      "zakariyya",
    "sufyœn":      "sufyan",
    "‹awŒf":      "tawaf",
    "qubœõ":      "quba",
    "êafœ":      "safa",
    "naôeem":      "naeem",
    "dardœõ":      "darda",
    "n´n":      "nun",
    "kœf":      "kaf",
    "kœna":      "kana",
    "saÔ¥":      "sai",
    "ôabbœs":      "abbas",
    "jilbœb":      "jilbab",
    "ôul´m":      "ulum",
    "jœmiôu":      "jamiu",
    "Bayúaq¥":      "bayhaqi",
    "úasan":      "hasan",
    "ras´l":      "rasul",
    "yagh´th":      "yaghuth",
    "yaúyœ":      "yahya",
    "ôabdur":     "abdur",
    "§alŒh":      "salah",
    "alœh":      "alah",
    "baú¥rah":   "bahirah",
    "malœõikah":     "malaikah",
    "tubbaô":      "tubba",
    "ilyasaô":     "ilyasa",
    "êiddeeq":     "siddeeq",
    "jœl´t":      "jalut",
    "kitœb":      "kitab",
    "aúkœmil":     "ahkamil",
    "muôœwiyah":  "muawiyah",
    "meen":      "ameen",
    "m´sœ":      "musa",
    "ilœha":     "ilaha",
    "muôawwidhat":    "muawwidhat",
    "rakôah":       "rakah",
    "îayy":       "Hayy",
    "isúœq":       "ishaq",
    "makt´m":       "maktum",
    "inshirœú":       "inshirah",
    "fatœwœ":       "fatawa",
    "wadd":       "wadd",
    "yaô´q":       "yauq",
    "FarŒheed¥":         "faraheedi",
    "farœheed":      "faraheed",
    "yaôm´r":      "yamur",
    "ilyœs":      "ilyas",
    "ôarafœt":     "arafat",
    "ôuzayr":     "uzayr",
    "fœtiúat":     "fatihat",
    "kaôb":      "kab",
    "ôubayy":    "ubayy",
    "marwœn":    "marwan",
    "barœõ":    "bara",
    "umœmah":    "umamah",
    "lœt":       "lat",
    "mabœúith":       "mabahith",
    "tawúeed":   "tawheed",
    "îunayn":      "Hunayn",
    "ôabdul":      "abdul",
    "ôuqbah":      "uqbah",
    "$ŒÕif":      "taif",
    "œõif":      "aif",
    "tawrœh":      "tawrah",
    "kœf´r":      "kafur",
    "kaf´r":      "kafur",
    "DuÕl¥":      "duli",
    "duõl":      "dul",
    "r´ú":      "ruh",
    "JŒl´t":      "jalut", 
    "œl´t":      "alut", 
    "muôœdh":      "muadh",
    "suwœô":      "suwa",
    "bœhil":      "bahil",
    "baôl":      "bal",
    "taúreef":      "tahreef",
    "ô¡sœ":      "isa",
    "shay‹œn":      "shaytan",
    "îœ":      "Ha",
    "ôzib":      "azib",
    "ôa§r":      "asr",
    "a§r":      "asr",
    "ôishaõ":      "isha",
    "in-shaõ-allah":      "in-sha-allah",
    "kha‹‹œb":      "khattab",
    "ab´ bakr a§-siddeeq":      "abu bakr as-siddeeq",
    "ab´ bakr":      "abu bakr",
    "mad¥nah":      "madinah",
    "an§œr":      "ansar",
    "lœ ilaha":      "la ilaha",
    "a‹-$´r":      "at-tur",
    "qa‹‹œn":      "qattan",
    "fi‹rah":      "fitrah",
    "f¥l":      "fil",
    "taô‹eel":      "tateel",
    "qur‹ub¥'s":      "qurtubi's",
    "$œ ha":      "Ta ha",
    "qa§a§":      "qasas",

    #Mohsin Khan Translation
    "Allahs":     "Allah's",
    "Moosa":      "musa",
    "salat":      "salah",
    "firaun":     "firawn",
    "shaitan":    "shaytan",
    "iesa":      "isa",
    "taurat":      "torah",
    "yoosuf":      "yusuf",
    "aal-e-imran":      "ali imran",
    "baôl":      "bal",
    "baôl":      "bal",
    "baôl":      "bal",
    "baôl":      "bal", 



    # Removing Surah Names matching with Names
    "surah ali imran" : "",
    "surah 3 – ali imran" : "",
    "surah 9 – at-tawbah" : "",
    "surah at-tawbah" : "",
    "surah 10 – yunus" : "",
    "surah yunus" : "",
    "surah 11 – hud" : "",
    "surah hud" : "",
    "surah 12 – yusuf" : "",
    "surah yusuf" : "",
    "surah 14 – ibrahim" : "",
    "surah ibrahim" : "",
    "surah 19 – maryam" : "",
    "surah maryam" : "",
    "surah 31 – luqman" : "",
    "surah luqman" : "",
    "surah 34 – saba" : "",
    "surah saba" : "",
    "surah 47 – muhammad" : "",
    "surah muhammad": "",
    "surah ar-rahman" : "",
    "surah 55 – ar-rahman" : "",
    "surah 71 – nuh" : "",
    "surah nuh" : "",
    "surah 72 – al-jinn" : "",
    "surah al-jinn" : "",
    "surah 76 – al-insan" : "",
    "surah al-insan" : "",


    "ﺪ": "",
    "ﺘ":"",
    "ﻳ":"",
    "ﺮ":"",
    "ﻭ":"",
    "ﺑ":"",
    "ﻟ":"",
    "ﻭﻥﹶ":"",
    "ﺍﻟﻘﹸﺮﺁﻥﹶ":"",
    "ﻋ":"",
    "ﻢ": "",
    "ﺃﹶﻓﹶﻼﹶ": "",
    "ﻣ": "",
    "ﻭﺍ": "",
    "ﻴ": "",
    "ﻮﺍ": "",
    "ﺍﷲ": "",
    "ﻠﹶﻰ": "",
    "ﺍﻷ": "",
    "ﺎﺭ": "",
    "ﺑﹺﻲ": "",
    "ﻥﹸ": "",
    "ﺎﺏﹺ": "",
    "ﻟﹶﻮ": "",
    "ﻓ": "",
    "ﺇﹺﻟﹶﻴﻚ": "",
    "ﺁﻳ": "",
    "ﺎﺗ": "",
    "ﻟﺒ": "",
    "ﺍﻟﻘﹸﺮ": "",
    "ﻠﱠﻬ": "",
    "ﺟ": "",
    "ﺴﺘ": "",
    "ﺆﻣ": "",
    "ﻟﻴ": "",
    "ﻨ": "",
    "ﺍ": "",
    "ﻥﹶ": "",
    "ﺍﻘﹸﺁﻥﹶ": "",
    "ﻪ": "",
    "ﺴ": "",
    "ﺒ": "",
    "ﺁ": "",
    "ﹶﻮ": "",
    "ﹶﻠ": "",
    "ﻏﹶﲑﹺ": "",
    "ﺃﹶﻡ": "",
    "ﺠﹺﺒ": "",
    "ﺍﺣ": "",
    "ﻲ": "",
    "ﻛﹶﺜ": "",
    "ﻦ": "",
    "ﺍﻜﹶﹺ": "",
    "ﺎﻩ": "",
    "ﻛ": "",
    "ﻼﹰﺎ": "",
    "ﺆ": "",
    "ﹶﻌ": "",
    "ﺷ": "",
    "ﹶﻮﻛﹶﺎﻥﹶ": "",
    "ﹺﻲ": "",
    "ﺇﹺﹶﻚ": "",
    "ﲑ": "",
    "ﺎﺏ": "",
    "ﻙ": "",
    "ﻗﹸﻠﹸﻮﺏﹴ": "",
    "ﺃﹶﻗﻔﹶﺎﹸﻬ": "",
    "ﺍﲪﻦ": "",
    "ﺍﻘﹸ": "",
    "ﺃﹸﹸ": "",
    "ﺃﹶﻧﺰ": "",
    "ﺍﺧ": "",
    "ﺬﹶﻛﱠ": "",
    "ﻜﹶﹺ": "",
    "ﺠﹺ": "",
    "'ﺧ'": "",
    "ﺣ": "",
    "ﲪ": "",
    "ﻘﹸ": "",

}

 

#  'kaf´r',
# "kha":            "kha",
# "tme":            "tme",
# "shihab":            "shihab": 
# "shaõ":             "shaõ": 
# "thihar":             "thihar": 
# "taô":            "taô",
# "ôishaõ":            "ôishaõ",


#%%
additional_english_words = ['uncloven', 'speciality','jewish','warners','david','attainers',
            'quran','effecter','reliers','abbasine','warner','outrunners','arabic',
'accommodators', 'arabs','israelite','abyssinian','repeller', 'all-encompassing','well-knowing','pre-islamic',
'guardian-angels', 'withholder', 'all-encompassing', 'non-existent', 'withholders', 'deviators',
 'messenger-angels', 'self-deluded', 'self-deluded','darknesses','prophethood','etc','corrupters',
 'bedouins','bestower','friday', 'islamically','sirius','byzantines','affirmers', 'indulgement','consisely',
  'compeller', 'israelites', 'headcovers', 'exaltedness', 'christianity', 'establishers','associators',  'amenders',
   'praisers', 'untrellised', 'copt', 'knowledgable', 'subjugators', 'hinderers', 'michael', 'misleader',
'avoided', 'unmindfulness', 'misguiders', 'disacknowledging', 'romaeans', 'christ', 'pre','unevident',
"handclapping", "milkings",


]

names = ['taymiyyah','malik','khidhr','naml','abrahah','kahf','qarnayn','zubayr','sulayman','mugheerah', 'az-zubayr',
 'Salih', 'idrees','moses','satan','jacob','isaac','gabriel','aaron','christians','adam','thamud',
 'adam','aad',"yusuf","hud",'iblees','ishmael','maryam','yunus','shuayb','rahman',"saba","luqman",
"azeez",'elias','abul','goliath','bakr','sabeans','saeed','lote','hurayrah', 'akhnas','hisham', 'thabit',
 'aas', 'marut', 'muttalib', 'qasim', 'hashim', 'zaynab', 'harun', 'elisha', 'kifl', 'quraythah', 'dhurriyyah',
'ismaeel', 'saul',"Khudri",'lahab','ezra', "iqra","masud","aishah", "hafsah","samiri", "hajjaj",
"zabur","malik","affan","dajjal","dawud","Tirmidhi",'harut','ubayy','thuhr',"kab", "ubayy",
"urwah",'banun',"Hunayn", "abdul", "uqbah", "taif","jalut",  "lat","bal",  "ilyas", "arafat", "uzayr",
 "marwan", "bara", "umamah","muadh","faraheedi", "faraheed", 'ilyasa', 'sufyan', 'darda', 'naeem',
'yaqub', 'abdur','bayhaqi','muawiyah','musa', 'rasul','siddeeq', 'bahirah','haman',"isa",'khattab', 'azib',

         ]


islamic_words = ['Allah','jizyah','muzdalifah','subhanahu',' Dhul-Hijjah', 'hijjah','sunnah', 'mulk',
"Ta ha ", "Mutaffifeen", "Fatir", "Ya Seen", "Hashr", "Haqqah", "inshiqaq", "Saff","man", "iddah", "raina",
'khandaq','persians','khaybar', 'israeel', 'tafseer', 'marwah', 'kafirun', 'takweer', 'muhajireen',
"takathur","uhud","qariah", "aadiyat", "dhuha", "maun", "taqwa","uthman", "qarun", 'magians', 'damascus',
'layl','banu','katheer','zalzalah','fitnah','saqar','hadiths',"abasa", "sharh", "nuh", "buruj", "alaq","infitar",
"ala","naziat","ahmad", "mumtahinah","insan","mursalat","naba", "ikhlas",'kawthar','balad','ghashiyah',
"jathiyah","sahabah","nas",'adhdhariyat',"fath","tahreem","hujurat","qiyamah","at-tariq", 'madinah',
'Abdullah','iram','ramadhan', 'islamic', 'qurnayn', 'al-Arsh', 'az-zalzalah', 'firawn', 'kaffarah', 'abu',
 'al-miraj', 'al-firdaus', 'as-sajdah','rajab', 'yathrib', 'al-kahf', 'al-kawthar',  'al-akhnas',
'satan','muslims','baqarah','noah','joseph','jesus','mary','jews','raheem','juz','anbiya', 'ihsan',
'saffat','kabah','ankabut','hijr','fussilat','mahfuth','waqiah','zechariah','dukhan', 'masad',
"nur","rum","Saad","ahqaf", "Shura", "rad","lawh",'badr',"qaf","Hadeed","mujadilah",'qamar', 'jumuah', 'janabah',
'rahmanir','bismillahir','taala','nisa','Imran','meem','quraysh',"maidah","anam","araf", 'firdaus',
"nahl","ahzab",'dhul','shuara','madyan','zumar','muminun','ghafir','isra','anfal', 'tamtheel',
'Saheeh','sheba','ghusl','humazah','zukhruf','muddaththir','injeel','gog','khamr', 'khaleel',
    'Jeddah', 'USA', 'jibreel','waleed','alif','fidyah','mahr','sajdah','maghrib', 'lahum',
    "Abd", "Ameen", "Ahlul Bayt", "Ajr", "Alhamdulillah", "Al-Qur'an", "Akhirah",  'shurayq',  "qadah",
    "Akhlaq", "Amal", "Arafah", "Asr", "Awrah", "Bismillah", "Dajjal", "Dhuhr",  'mundhir',
    "Du'a", "Fard", "Fajr", "Fitrah", "Ghazwa", "Hadith", "Hajj", "Halal", "Haram",  'surahs',
    "Hasan", "Ibn", "Ijma", "Ijtihad", "Ilah", "Iman", "Imam", "Islami", "Jahannam",  'majuj',
    "Jannah", "Jihad", "Kafir", "Khilafah", "Khadijah", "Khutbah", "Kufr", "La ilaha illallah", 
    "Lailahailallah", "Masjid", "Mawlid", "Mizan", "Nafl", "Niyyah", "Qada", "Qadr",  'ilyaseen',
    "Qiblah", "Raka'ah", "Ramadan", "Sadaqah", "Salah", "Sawm", "Sincerity", "Siraat",  'tasneem','rakahs',
    "Sunnah", "Surah", "Tafsir", "Tahajjud", "Tawbah", "Tawheed", "Tawakkul", "Ummah",  'shawwal',
    "Zakat", "Zikr", "Zuhd", "Adhan", "Alif", "Baqara", "Barakah", "Dar al-Islam",  'takyeef',
    "Dhimmi", "Dua", "Fajr", "Farz", "Fitra", "Ghusl", "Halal", "Haraam", "Hikmah",  'inshirah',
    "Ilm", "Jazaa", "Khalifah", "Khatib", "Khutbah", "Masjid", "Mubah", "Mu'min",  'mumtahanah',
    "Mushrik", "Naskh", "Qiyas", "Rijz", "Sahaba", "Sahih", "Sani", "Sattak", "Sura",  'jonah', 'jew',
    "Tabi'in", "Tafwid", "Taqwa", "Tawassul", "Thawab", "Waqf", "Zamzam", "Zina",  'sijjeen',
    "Adab", "Adalat", "Al-Baqarah", "Al-Fatiha", "Al-Hud", "Al-Maidah", "Al-Mulk",  'salamu',
    "Al-Nisa", "Al-Tawbah", "Al-Zumar", "An-Nahl", "An-Nas", "Asbab", "Awal",  'magog', 'miraj',
    "Barzakh", "Bashar", "Da'wah", "Darajat", "Dunya", "Fasad", "Fitan", "Ghaib",  'tayammum', 'salsabeel',
    "Hakam", "Hikma", "Ilahiyat", "Inam", "Ittihad", "Khayr", "Khilafah", "Mahdiyyah",  'wasilah',
    "Malik", "Maslahat", "Mu'minah", "Muhajir", "Munafiqun", "Mustahabb", "Mutahhir",  'qalam',
    "Nafs", "Nasiha", "Nasr", "Qadir", "Quran", "Quranic", "Razi", "Ruh", "Sadaqat",  'harith',
    "Shahid", "Shari'ah", "Shirk", "Sihah", "Sulh", "Sunah", "Tafkik", "Tarikh",  'saibah',
    "Taqwa", "Taqlid", "Ubudiyyah", "Usul", "Zina", "Alim", "Hujjah", "Sadr",  'yajuj',
    "Safa", "Safar", "Shahr", "Sunnat", "Suyuti", "Tasfiyah", "Umrah", "Witr",  'arsh', 'talaq',
    "Zawaj", "Zawj", "Zikrullah", "Adhan", "Amir", "Ansar", "At-Tawbah", "Ayat",  'aykah',
    "Barakah", "Fudul", "Furqan", "Ghadir", "Hawzah", "Ilm al-Kalam", "Kalam",  'baraah',
    "Khalq", "Makruh", "Mawaddah", "Mujahid", "Munafiq", "Mut'ah", "Nabi", "Nizami",  'harithah',
    "Qadariyyah", "Sadr", "Salah", "Salawat", "Sama", "Sibghah", "Sijjin", "Siyah", 
    "Tafsir al-Jalalayn", "Tazkiyah", "Ummah", "Wali", "Yahudi", "Zahir", "Ala",  'muzzammil',
    "Azan", "Adab", "Aqidah", "Al-Azhar", "Al-Farabi", "Al-Ghazali", "Al-Maturidi", 
    "Al-Razi", "Ali", "Asma", "Asr", "Awrah", "Bilal", "Dhikr", "Ibadah",  'falaq', 'makkans',
    "Iftah", "Ikhlas", "Ilahi", "Isnad", "Isr", "Maqam", "Mufti", "Mujtahid", 'zaqqum', 'taghabun',
    "Mursaleen", "Qir'at", "Ramadan", "Sham", "Shaykh", "Tafakkur", "Tafsir",  'nadheer',
    "Ta'zir", "Taqi", "Tasbih", "Uthman", "Yaum", "Zakat", "Zar", "Ziyarat", 
    "Al-Ikhlas", "Al-Bukhari", "Al-Nasai", "Al-Qadr", "Al-Tahrir", "Al-Umrah", 
    "Dar al-Hikmah", "Dar al-Harb", "Fiqh", "Hijab", "Kalamullah", "Mahr", 
    "Maqasid", "Mahr", "Muharram", "Nawafil", "Qurashi", "Rabia", "Rashidun", 
    "Shahadah", "Siraat al-Mustaqim", "Sunni", "Zaroorat", "Zawji", "Ayyub", "tabiun","suhuf",
    "Al-Bari", "Al-Hakam", "Al-Ma'arij", "Al-Quraish", "Anwar", "Bukhari", "saheeh","mushaf",
    "Fatihah", "Ibn Abbas", "Ikhwah", "Islamiat", "Jibrail", "Lailahailallah",  'tabarak',
    "Lailat al-Miraj", "Mawlana", "Mu'minun", "Murshid", "Ruh al-Quds",  'najm',
    "Sadiq", "Shahr Ramadan", "Surah Al-Fatiha", "Tawakkul", "Ulum", "Uthman", 
    "Wali", "Yusuf", "Zayd", "Zina", "Abbas", "Abdul", "Ahl al-Kitab", "Al-Ghafir", 
    "Ali", "Al-Hadi", "Al-Jabbar", "Al-Ma'arij", "Al-Mundhir", "Al-Rahman", 
    "Al-Rahim", "Al-Razzaq", "Al-Wadud", "An-Najm", "Ar-Rahman", "Ar-Rahim", 
    "Asbab al-Nuzul", "Baitul Mal", "Baitul Muqaddas", "Bayyinah", "Bukhari", 
    "Daru al-Islam", "Daru al-Harb", "Durood", "Fatimah", "Ghaib", "Ghulam", 
    "Hadhrat", "Hadi", "Huda", "Kahf", "Kufr", "Makki", "Manzil", "Mumin", 
    "Murad", "Nabi", "Nashid", "Nawafil", "Qisas", "Qur'an", "Qur'an al-Karim", 
    "Sahaba", "Salaf", "Sawa", "Shaykh", "Shura", "Sura", "Tabi'in", "Tauba", 
    "Wahid", "Yahya", "Ziya", "Zuhd", "Zikr", "Zindagi", "Zuhd", "Al-Salamu Alaykum", 
    "Al-Khaatam", "As-salam", "Asmaul Husna", "Fatiha", "Hadith Qudsi", 
    "Halim", "Hasan", "Ibrahim", "Ismail", "Khadijah", "Masjid al-Haram", 
    "Masjid al-Nabawi", "Muhammad", "Nabi Yunus", "Nisa", "Rabia al-Adawiya", 
    "Rida", "Sa'ad", "Salah al-Janazah", "Saqifah", "Shahada", "Tawhid", 
    "Umar", "Zaid", "Ayyub", "Bashir", "Fatih", "Fuqaha", "Hafiz", 'sai', 'wal', 'tubba', 'zakariyya', 
    "Hanafi", "Maqbul", "Maqdis", "Maliki", "Naseehah", "Qadar", "Salah", 'thihar', 'quba',  'jilbab', 
    "Shafi", "Sufi", "Surah", "Zamzam", "Zainab", "Ahl al-Hadith", 'taghut','malaikah', 
    "Ahl al-Sunnah", "Al-Farouk", "Al-Kabir", "Al-Mahdi", "Al-Maqdis", 
    "Al-Mu'min", "Al-Rahman", "Al-Rahim", "Al-Zuhd", "Amir al-Mu'minin",  "shaytan",
    "Asbab al-Nuzul", "Azaan", "Da'wa", "Dar al-Islam", "Dhikrullah", 'kaf',
    "Dua", "Fiqh al-Aqeedah", "Fiqh al-Mu'amalat", "Fitr", "Ghusl", "alut", "isha",
    "Ilah", "Jahiliyyah", "Khiyar", "Mafhum", "Mu'min", "Mushrik", "ishaq", "maktum", "inshirah", "fatawa", "wadd",
    "Nasr", "Rabita", "Salafi", "Salah", "Shahadah", "Tafsir", "muawwidhat", "rakah", "Hayy", 
    "Wudhu", "Zakat", "Zakah", "Adab", "Adl", "Al-Baqarah",  'alaykum',
    "Al-Jannah", "An-Nas", "Azzak", "Dhuhar", "Hijrah", "Ihtiram", 'ahkamil', 'rafi', 'shihab', 
    "Khutbah", "Mahr", "Maqsad", "Muhafiz", "Munkar", "Musawwir", "yauq",  "yamur",   
    "Qal", "Qadr", "Ramadan", "Razaq", "Ruh", "Saqifah", "Salah al-Tarawih",  'tawaf', 'jamiu', 'yaghuth',
    "Sariyah", "Sijjin", "Sukoon", "Sunnat al-Fajr", "Taaziyah", "Tafsir Ibn Kathir", 
    "Taqdeer", "Tawheed", "Umrah", "Uthman", "Witr", "Zamzam", "aif", "tawrah", "kafur", "duli",    
    "Zina", "Al-Ma'arij", "Al-Rahman", "Al-Tawbah", "Asr", 'jibt','diyah', "dul",     "ruh", 
    "At-Tahreem", "Barakah", "Firdaws", "Ghafoor", "Hidayah", 'aswad',"tahreef", "fatihat",
    "Ilahi", "Imam al-Mahdi", "Jahiliyyah", "Khalil", "Makruh", "manna","mualla",
    "aqeedah","muntada","ahl","dahr","dawah","tuwa",    "hudaybiyyah","unthurna",
    "Maturidiyyah", "Mu'tazilah", "Qadiriyyah", "Sahih Muslim", "Salah al-Eid", 
    "Sahabah", "Sunan", "Tahaarah", "Thawab", "Waqf", "Zakat al-Fitr", 
    "Zakat al-Maal", "Zamzam", "Zuhd", "Ahl al-Kitab", "Ahl al-Sunnah",  "Al-Amin", "Al-Kareem", "Al-Latif",
      "Al-Malik", "Al-Quddus", "Walid", "Yawm", "Zayd", "Zina", "Zuhd"
    "Al-Salam", "Al-Wahid", "Al-Wadud", "Amir", "Asbab", "Bari",  "Burda", "Da'wah", "Dar al-Hikmah", "Dunya", "Fasal", 
    "Fatwa", "Fujur", "Gaflah", "Hikmat", "Ibadah",  "Isha", "Islam", "Khalkh", "Maklumat", "Maslahat", 
    "Mawqif", "Mu'tamar", "Nasihah", "Nazar", "Rabiyah", "Raqiq", "Sahih", "Sujud", "Sura", "Tariq", 
    "Tashawwur", "Tawbah", "Umm", "Wasat", "Zarurat", "Asbat", "Haram", "tabuk", "mujahideen", "Kursi","illiyyun", "ayn", "alaykum"
    "Zina", "Al-Adab", "Al-Dhul-Qarnayn", "Al-Hudud", "Al-Ittihad",  "mabahith", "tawheed",     
    "Al-Jaheem", "Al-Khalkh", "Al-Khiraaf", "Al-Mu'azzin", "Al-Rahim", "suwa",     "bahil",
    "Al-Sajdah", "Awwal", "Bahr", "Bariyah", "Burhan",  'maarij', 'uzza',
    "Furqan", "Ghadir", "Hakam", "Haqiqah", "Ijtihad",  "Ikhlas", "Ilaha", "Khalq", "Marah", "Mujahid", 
    "Mushrif", "Nasira", "Qabilah", "Qadar", "Salah al-Tahajjud", 
    "Sama", "Sibghah", "Sura al-Baqarah", "Tabi'i", "Ubudiyyah",  "Wajh", "Zuhd", "Abdurrahman", "Adl", "Ameen", 
    "Amal", "Ayat al-Kursi", "Al-Azhar", "Al-Farooq", "Al-Muhajir",  "Al-Qur'an", "Asbab", "Bashar", "Dar al-Salam", "Dhalim", 
    "Dua", "Fatihah", "Ghurabah", "Hijab", "Ibn Taymiyyah",  "Ikhwan", "Kafir", "Kitab", "Mahr", "Mujahid", 
    "Muslim", "Nasara", "Qadi", "Qibla", "Rabb", "Ruqyah", "Sadaqah", "Sijjin", "Tawheed", "Umrah", 
    "jamaah","sakhkhah","manat","awla","wudhu","surahs","rahmah",
    
]

locations = [
    "United States", "California", "Los Angeles",'asia','makkah','bakkah','samaria','tannur', 'sinai',
    "California", "San Francisco", "California", "San Diego", 'nubian', 'babylon',
    "Texas", "Houston", "Texas", "Dallas", "Texas", "Austin", 'jerusalem','Judiyy',
    "New York", "New York City", "New York", "Buffalo", "New York", "Rochester",
    "Canada", "Ontario", "Toronto", "Ontario", "Ottawa", "Ontario", "Hamilton",
    "Quebec", "Montreal", "Quebec", "Quebec City", "Quebec", "Gatineau", 'mashar',
    "British Columbia", "Vancouver", "British Columbia", "Victoria", "British Columbia", "Surrey",
    "Australia", "New South Wales", "Sydney", "New South Wales", "Newcastle", "New South Wales", "Wollongong",
    "Victoria", "Melbourne", "Victoria", "Geelong", "Victoria", "Ballarat", 'persia',
    "Queensland", "Brisbane", "Queensland", "Gold Coast", "Queensland", "Cairns",
    "United Kingdom", "England", "London", "England", "Manchester", "England", "Birmingham",
    "Scotland", "Edinburgh", "Scotland", "Glasgow", "Scotland", "Aberdeen",
    "Wales", "Cardiff", "Wales", "Swansea", "Wales", "Newport",
    "India", "Maharashtra", "Mumbai", "Maharashtra", "Pune", "Maharashtra", "Nagpur",
    "Karnataka", "Bangalore", "Karnataka", "Mysore", "Karnataka", "Mangalore",
    "Delhi", "New Delhi", "Delhi", "Delhi",
    "Germany", "Bavaria", "Munich", "Bavaria", "Nuremberg", "Bavaria", "Augsburg",
    "Berlin", "Berlin",
    "North Rhine-Westphalia", "Cologne", "North Rhine-Westphalia", "Düsseldorf", "North Rhine-Westphalia", "Dortmund",
    "France", "Île-de-France", "Paris", "Île-de-France", "Versailles", "Île-de-France", "Boulogne-Billancourt",
    "Provence-Alpes-Côte d'Azur", "Marseille", "Provence-Alpes-Côte d'Azur", "Nice", "Provence-Alpes-Côte d'Azur", "Toulon",
    "Auvergne-Rhône-Alpes", "Lyon", "Auvergne-Rhône-Alpes", "Grenoble", "Auvergne-Rhône-Alpes", "Saint-Étienne",
    "Brazil", "São Paulo", "São Paulo", "São Paulo", "Campinas", "São Paulo", "Santos",
    "Rio de Janeiro", "Rio de Janeiro", "Rio de Janeiro", "Niterói", "Rio de Janeiro", "Petrópolis",
    "Minas Gerais", "Belo Horizonte", "Minas Gerais", "Ouro Preto", "Minas Gerais", "Uberlândia",
    "China", "Beijing", "Beijing", "Shanghai", "Shanghai", "Guangdong", "Guangzhou", "Guangdong", "Shenzhen", "Guangdong", "Dongguan",
    "Japan", "Tokyo", "Tokyo", "Osaka", "Osaka", "Hokkaido", "Sapporo",
    "Russia", "Moscow", "Moscow", "Saint Petersburg", "Saint Petersburg", "Tatarstan", "Kazan",
    "Mexico", "Mexico City", "Mexico City", "Jalisco", "Guadalajara", "Nuevo León", "Monterrey",
    "South Africa", "Gauteng", "Johannesburg", "Gauteng", "Pretoria",
    "Western Cape", "Cape Town", "Western Cape", "Paternoster",
    "KwaZulu-Natal", "Durban", "KwaZulu-Natal", "Pietermaritzburg",
    "Italy", "Lazio", "Rome", "Lombardy", "Milan", "Campania", "Naples", "Campania", "Salerno",
    "Argentina", "Buenos Aires", "Buenos Aires", "Cordoba", "Cordoba", "Santa Fe", "Santa Fe",
    "Egypt", "Cairo", "Cairo", "Alexandria", "Alexandria", "Giza", "Giza",
    "Turkey", "Istanbul", "Istanbul", "Ankara", "Ankara", "Izmir", "Izmir",
    "Saudi Arabia", "Riyadh", "Riyadh", "Jeddah", "Jeddah", "Mecca", "Mecca"
]
