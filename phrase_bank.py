"""
phrase_bank.py
==============

The phrase-level event bank for codebook training.

Each concept maps to a list of (phrase, image_search_query) pairs.
- phrase        : input to the LLM (Mistral / ST) — disambiguates physical sense
- image_query   : Wikimedia/web search query to fetch an image that matches
                  the same physical sense for CLIP/MAE/V-JEPA 2

Design principles:
  1. Each phrase describes a specific physical EVENT, not an abstract category.
     "water running down a drain" not "water"
  2. Phrases should span the distinct physical senses of polysemous concepts.
     "run" needs locomotion, fluid, mechanical senses covered separately.
  3. Phrases should be usable in a PIQA-style question without ambiguity.
  4. 5–9 phrases per concept; more for high-polysemy concepts.
  5. image_query should find a photo, not a diagram — world models need naturalistic images.

Total events: ~343 (49 concepts × ~7 each)
"""

PHRASE_BANK = {

    # ── ORIGINAL 17 CONCEPTS ──────────────────────────────────────────────────

    "apple": [
        ("an apple falling from a tree branch",                 "apple falling from tree"),
        ("an apple rolling across a wooden table",              "apple rolling on table"),
        ("an apple being cut with a knife",                     "cutting apple with knife"),
        ("an apple floating in a bucket of water",              "apple floating water"),
        ("an apple bruising after impact with the floor",       "bruised apple drop"),
        ("an apple being crushed underfoot",                    "apple crushed underfoot"),
    ],
    "chair": [
        ("a chair tipping over backwards",                      "chair tipping over"),
        ("a chair bearing the weight of a seated person",       "person sitting chair weight"),
        ("a wooden chair breaking under excessive load",        "broken wooden chair collapse"),
        ("a chair sliding across a polished floor",             "chair sliding floor"),
        ("stacking chairs on top of each other",                "stacking chairs"),
        ("a chair leg sinking into soft ground",                "chair leg sinking ground"),
    ],
    "water": [
        ("water flowing downhill over rocks",                   "water flowing stream rocks"),
        ("water running down a drain",                          "water going down drain"),
        ("water filling a container from a tap",                "water filling container tap"),
        ("water evaporating from a heated surface",             "water evaporating hot surface"),
        ("water freezing into ice in a cold environment",       "water freezing ice"),
        ("water spreading across a flat surface",               "water spreading flat surface"),
        ("water pressure bursting through a weak seal",         "water pressure burst pipe"),
    ],
    "fire": [
        ("fire spreading across dry leaves on a forest floor",  "fire spreading dry leaves"),
        ("fire consuming a wooden plank",                       "fire burning wood plank"),
        ("a campfire burning steadily in a fire pit",           "campfire burning steadily"),
        ("fire going out when deprived of oxygen",              "fire extinguished no oxygen"),
        ("fire melting a wax candle from above",                "fire melting candle wax"),
        ("flames bending in the wind",                          "flames bending wind"),
    ],
    "stone": [
        ("a stone sinking to the bottom of a pond",             "stone sinking water"),
        ("a stone rolling down a steep hillside",               "stone rolling downhill"),
        ("a stone being skipped across a water surface",        "skipping stone water surface"),
        ("a large stone crushing a wooden crate",               "heavy stone crushing crate"),
        ("a stone remaining immobile when pushed lightly",      "large stone immobile push"),
        ("a stone breaking when struck with a hammer",          "stone breaking hammer strike"),
    ],
    "rope": [
        ("a rope under tension pulling two objects apart",      "rope under tension pulling"),
        ("a rope being knotted and pulled tight",               "rope knot pulled tight"),
        ("a rope fraying where it contacts a sharp edge",       "rope fraying sharp edge"),
        ("a rope coiled and thrown across a gap",               "throwing rope across gap"),
        ("a rope snapping under excessive load",                "rope snapping breaking"),
        ("a rope supporting a hanging weight",                  "rope supporting hanging weight"),
    ],
    "door": [
        ("a door swinging open on its hinges",                  "door swinging open hinges"),
        ("a door held shut by a strong wind",                   "door held shut by wind"),
        ("a door being forced open by pushing",                 "forcing door open push"),
        ("a sliding door rolling along a track",                "sliding door track"),
        ("a door slamming against its frame",                   "door slamming frame"),
        ("a door wedged open with a stopper",                   "door wedge stopper"),
    ],
    "container": [
        ("a container being filled with liquid from above",     "pouring liquid into container"),
        ("a sealed container building up internal pressure",    "sealed container pressure buildup"),
        ("a container tipping over and spilling its contents",  "container tipping spilling"),
        ("a container floating in water when empty",            "empty container floating water"),
        ("a container sinking when filled with water",          "container sinking when filled"),
        ("a lidded container being opened by prying",           "prying open lidded container"),
    ],
    "shadow": [
        ("a shadow moving as the sun changes position",         "shadow moving with sun"),
        ("an object casting a long shadow in low sunlight",     "long shadow low sunlight"),
        ("a shadow disappearing when a cloud covers the sun",   "shadow disappearing cloud"),
        ("overlapping shadows creating a darker region",        "overlapping shadows darker area"),
        ("a shadow shrinking at midday",                        "shadow shrinking midday sun"),
    ],
    "mirror": [
        ("a mirror reflecting light onto the opposite wall",    "mirror reflecting light wall"),
        ("a mirror showing a reversed image of a face",         "mirror reversed reflection face"),
        ("a mirror shattering when struck",                     "mirror shattering impact"),
        ("a curved mirror distorting a reflected image",        "curved mirror distorted reflection"),
        ("light bouncing off a mirror at an angle",             "light reflecting mirror angle"),
    ],
    "knife": [
        ("a knife slicing through a ripe tomato",               "knife slicing tomato"),
        ("a knife cutting rope by drawing along the blade",     "knife cutting rope"),
        ("a knife losing sharpness after cutting hard material","dull knife hard material"),
        ("a blade concentrating force on a thin edge",          "blade thin edge force"),
        ("a knife bending under lateral force",                 "knife bending lateral force"),
    ],
    "wheel": [
        ("a wheel rolling along a flat road",                   "wheel rolling flat road"),
        ("a wheel spinning in mud without traction",            "wheel spinning mud no traction"),
        ("a wheel decelerating due to braking friction",        "wheel braking friction deceleration"),
        ("a large wheel providing mechanical advantage",        "large wheel mechanical advantage"),
        ("a wheel wobbling when its axle is misaligned",        "wobbly wheel misaligned axle"),
    ],
    "hand": [
        ("a hand gripping a jar lid and twisting",              "hand gripping jar lid twist"),
        ("a hand catching a thrown object",                     "hand catching thrown ball"),
        ("a hand pressing down on a surface to test firmness",  "hand pressing surface firmness"),
        ("a hand squeezing water from a wet cloth",             "hand squeezing wet cloth"),
        ("fingers pinching a small object off a flat surface",  "fingers pinching small object"),
        ("a hand pushing a heavy box across a floor",           "hand pushing heavy box floor"),
    ],
    "wall": [
        ("a wall bearing the load of a ceiling above it",       "wall bearing ceiling load"),
        ("a wall deflecting sound from one room to another",    "wall deflecting sound"),
        ("a wall cracking under ground movement",               "wall cracking ground movement"),
        ("a thin wall transmitting vibration",                  "thin wall transmitting vibration"),
        ("a wall stopping the movement of a rolling object",    "wall stopping rolling ball"),
    ],
    "hole": [
        ("an object falling through a hole in a floor",         "object falling through floor hole"),
        ("water draining through a small hole in a container",  "water draining through hole"),
        ("a hole in a wall allowing light to pass through",     "hole in wall letting light through"),
        ("an object too large to pass through a hole",          "object too large to fit hole"),
        ("a hole deepening as material is removed",             "hole being dug deeper"),
    ],
    "bridge": [
        ("a bridge flexing under the weight of a passing truck","bridge flexing under truck"),
        ("a bridge transferring load to supports at each end",  "bridge load transfer supports"),
        ("a rope bridge swaying in the wind",                   "rope bridge swaying wind"),
        ("a bridge failing under excessive central load",       "bridge collapse center load"),
        ("a stone arch bridge distributing weight outward",     "stone arch bridge weight"),
    ],
    "ladder": [
        ("a ladder leaning against a wall at a safe angle",     "ladder leaning wall safe angle"),
        ("a ladder sliding when its base slips on smooth floor","ladder base sliding smooth floor"),
        ("a person climbing a ladder rung by rung",             "person climbing ladder rungs"),
        ("a ladder bearing weight distributed across its rungs","ladder bearing weight rungs"),
        ("a ladder tipping sideways when weight is uncentered", "ladder tipping sideways"),
    ],

    # ── NEW CONCEPTS ──────────────────────────────────────────────────────────

    # High polysemy / high sensorimotor
    "spring": [
        ("a coiled spring compressing under load",              "coiled spring compressing load"),
        ("a compressed spring releasing and launching an object","spring releasing launching object"),
        ("a spring stretching under tension",                   "spring stretching tension"),
        ("a spring returning to its rest length",               "spring returning rest length"),
        ("a stiff spring requiring more force to compress",     "stiff spring compression force"),
    ],
    "bark": [
        ("tree bark protecting a trunk from abrasion",          "tree bark protecting trunk"),
        ("bark peeling off a dead tree trunk",                  "bark peeling dead tree"),
        ("bark burning on the outside of a log in a fire",      "bark burning log fire"),
        ("rough bark scratching skin on contact",               "rough bark scratching skin"),
        ("bark absorbing water after rain",                     "bark absorbing rainwater"),
    ],
    "wave": [
        ("a wave breaking onto a rocky shore",                  "ocean wave breaking shore"),
        ("a wave transferring energy without moving water forward","wave energy transfer water"),
        ("a large wave lifting a floating object",              "large wave lifting boat"),
        ("wave height increasing in shallow water",             "wave height increasing shallow"),
        ("a wave reflecting off a solid barrier",               "wave reflecting barrier"),
    ],
    "charge": [
        ("static charge building up on a balloon surface",      "static charge balloon buildup"),
        ("a charged object attracting small paper pieces",      "charged object attracting paper"),
        ("electric charge jumping as a spark",                  "electric spark discharge"),
        ("charge dissipating when grounded through a conductor","charge dissipating grounding"),
        ("two like charges repelling each other",               "like charges repelling"),
    ],
    "field": [
        ("a ball rolling across a flat open field",             "ball rolling flat grassy field"),
        ("a field of grass bending under wind",                 "grass field bending wind"),
        ("an object thrown across an open field following a parabola","object thrown parabola field"),
        ("rain falling uniformly across a flat field",          "rain falling flat open field"),
        ("a flat field providing no cover from wind",           "flat field no wind cover"),
    ],
    "light": [
        ("light refracting through a glass prism",              "light refracting glass prism"),
        ("light casting a sharp shadow behind an object",       "light sharp shadow object"),
        ("light intensity decreasing with distance from source","light intensity distance falloff"),
        ("light reflecting off a polished metal surface",       "light reflecting polished metal"),
        ("light passing through a translucent material",        "light translucent material transmission"),
        ("light blocked by an opaque barrier creating shadow",  "light blocked opaque barrier"),
    ],
    "strike": [
        ("a hammer striking a nail and driving it into wood",   "hammer striking nail into wood"),
        ("a bat striking a ball and sending it forward",        "bat striking ball impact"),
        ("a fist striking a surface and transferring momentum", "fist striking surface impact"),
        ("a stone striking glass and causing it to crack",      "stone striking glass cracking"),
        ("a hard object striking a soft one and deforming it",  "hard object striking soft deformation"),
    ],
    "press": [
        ("pressing two flat surfaces together to seal them",    "pressing surfaces together seal"),
        ("pressing a soft material and leaving an impression",  "pressing soft material impression"),
        ("hydraulic press crushing a metal can",                "hydraulic press crushing metal can"),
        ("pressing juice out of a fruit by squeezing",          "pressing juice from fruit"),
        ("pressing down on a spring to compress it",            "pressing down spring compress"),
    ],
    "shoot": [
        ("a plant shoot growing upward toward light",           "plant shoot growing toward light"),
        ("a new shoot emerging from a cut stem",                "new shoot emerging cut stem"),
        ("a bamboo shoot pushing through soil",                 "bamboo shoot pushing through soil"),
        ("a fragile shoot bending under its own weight",        "fragile shoot bending weight"),
    ],
    "run": [
        ("a person running at full speed across a track",       "person running full speed track"),
        ("water running in a fast stream over pebbles",         "water running stream pebbles"),
        ("a machine running continuously at high speed",        "machine running high speed"),
        ("paint running down a vertical surface",               "paint running down wall"),
        ("a crack running across a surface under stress",       "crack running across material stress"),
        ("a river running between two steep banks",             "river running between steep banks"),
    ],

    # Low polysemy / high sensorimotor
    "hammer": [
        ("a hammer driving a nail flush into wood",             "hammer driving nail flush wood"),
        ("a hammer breaking apart a concrete block",            "hammer breaking concrete block"),
        ("a rubber mallet striking without damaging a surface", "rubber mallet gentle strike"),
        ("a hammer claw pulling a nail out of wood",            "hammer claw removing nail wood"),
        ("missing a nail with a hammer and striking wood",      "hammer missing nail hitting wood"),
    ],
    "scissors": [
        ("scissors shearing through a sheet of paper",          "scissors cutting paper"),
        ("scissors cutting fabric along a marked line",         "scissors cutting fabric line"),
        ("scissors failing to cut thick cardboard",             "scissors struggling thick cardboard"),
        ("the pivot point of scissors concentrating force",     "scissors pivot point force"),
        ("scissors slipping on a smooth material",              "scissors slipping smooth material"),
    ],
    "bowl": [
        ("a bowl holding liquid without spilling",              "bowl holding liquid"),
        ("a bowl tipping when weight is placed on its rim",     "bowl tipping weight rim"),
        ("a heavy bowl staying stable on a flat surface",       "heavy bowl stable surface"),
        ("a bowl being spun and the contents rising up the sides","bowl spinning contents rising"),
        ("a shallow bowl unable to hold deep liquid",           "shallow bowl liquid limit"),
    ],
    "bucket": [
        ("a bucket being filled with water from a hose",        "bucket filling with water hose"),
        ("a full bucket being lifted by its handle",            "full bucket lifted by handle"),
        ("a bucket tipping and spilling when unbalanced",       "bucket tipping spilling"),
        ("a bucket sinking into water when filled",             "filled bucket sinking water"),
        ("a bucket swung in a circle keeping water inside",     "bucket swung circle centripetal"),
    ],
    "bench": [
        ("a park bench supporting the weight of two people",    "park bench two people seated"),
        ("a bench bowing slightly under a heavy load",          "bench bowing heavy load"),
        ("a bench being used as a step to reach a high shelf",  "standing on bench reach shelf"),
        ("a workbench providing a stable surface for hammering","workbench stable hammering surface"),
    ],
    "fence": [
        ("a wooden fence stopping a rolling ball",              "fence stopping rolling ball"),
        ("a fence bending under lateral force from wind",       "fence bending wind pressure"),
        ("a person climbing over a fence",                      "person climbing over fence"),
        ("a fence post being driven into the ground",           "fence post driven into ground"),
        ("a fence providing a boundary but not a barrier to air","fence boundary not air barrier"),
    ],
    "needle": [
        ("a needle piercing fabric by concentrating force",     "needle piercing fabric"),
        ("a needle threading through a narrow eye",             "needle threading narrow eye"),
        ("a needle bending when forced laterally",              "needle bending lateral force"),
        ("a sharp needle making a small clean hole",            "needle small clean hole"),
        ("a needle scratching a hard surface lightly",          "needle scratching hard surface"),
    ],
    "drum": [
        ("a drumhead vibrating when struck with a stick",       "drumhead vibrating struck"),
        ("a tightly tuned drumhead producing a high pitch",     "tight drumhead high pitch"),
        ("a drum skin denting under a hard blow",               "drum skin denting hard blow"),
        ("resonance building inside a hollow drum body",        "drum body resonance hollow"),
    ],
    "clock": [
        ("clock hands rotating at a steady rate",               "clock hands rotating steady"),
        ("a pendulum swinging at a fixed period",               "pendulum swinging fixed period"),
        ("a clock mechanism unwinding a coiled spring",         "clock mechanism spring unwinding"),
        ("a clock running slow when its battery is low",        "clock running slow low battery"),
    ],
    "telescope": [
        ("a telescope gathering and focusing light from a star","telescope focusing star light"),
        ("a telescope lens magnifying a distant object",        "telescope lens magnification"),
        ("dew forming on a cold telescope lens outdoors",       "dew forming telescope lens"),
        ("a telescope being aimed precisely at a target",       "telescope precise aiming"),
    ],

    # Low polysemy / low sensorimotor
    "cloud": [
        ("a cloud forming as rising air cools and condenses",   "cloud forming condensation"),
        ("a cloud casting a large moving shadow on the ground", "cloud shadow moving ground"),
        ("a dark cloud releasing rain below it",                "dark cloud raining below"),
        ("a cloud dissipating as air warms",                    "cloud dissipating warming air"),
        ("wind driving a cloud across the sky",                 "wind moving cloud across sky"),
    ],
    "sand": [
        ("sand flowing through the neck of an hourglass",       "sand flowing hourglass"),
        ("sand shifting underfoot and reducing traction",       "sand shifting underfoot"),
        ("wet sand holding a shape when compressed",            "wet sand holding shape"),
        ("dry sand collapsing from a pile into a cone",         "dry sand collapsing pile cone"),
        ("sand abrasion wearing down a surface",                "sand abrasion wearing surface"),
    ],
    "ice": [
        ("ice melting when warm water is poured over it",       "ice melting warm water"),
        ("ice forming a slippery surface underfoot",            "ice slippery surface underfoot"),
        ("ice expanding as it freezes and cracking a container","ice expanding cracking container"),
        ("ice floating on liquid water",                        "ice cube floating water glass"),
        ("ice fracturing when struck sharply",                  "ice fracturing sharp impact"),
    ],
    "feather": [
        ("a feather drifting slowly downward in still air",     "feather drifting down still air"),
        ("a feather being deflected by a light breeze",         "feather deflected by breeze"),
        ("a feather floating on water without sinking",         "feather floating on water"),
        ("a feather being crushed flat under light pressure",   "feather crushed flat"),
    ],
    "leaf": [
        ("a dry leaf crumbling when squeezed",                  "dry leaf crumbling hand"),
        ("a leaf bending in the wind without breaking",         "leaf bending in wind"),
        ("a leaf floating on the surface of water",             "leaf floating water surface"),
        ("a wet leaf sticking to a smooth surface",             "wet leaf sticking smooth surface"),
        ("a leaf burning quickly when exposed to flame",        "leaf burning fire quickly"),
    ],
    "thread": [
        ("a thread breaking when pulled beyond its tensile limit","thread breaking tensile limit"),
        ("a thread being pulled tight around an object",        "thread pulled tight around object"),
        ("multiple threads twisted together forming a cord",    "threads twisted forming cord"),
        ("a thread tangling when coiled loosely",               "thread tangling loose coil"),
        ("a thread being threaded through a needle eye",        "thread through needle eye"),
    ],
    "glass": [
        ("a glass shattering into fragments on a hard floor",   "glass shattering hard floor"),
        ("light passing straight through a clear glass pane",   "light through clear glass pane"),
        ("a glass cracking from thermal shock",                 "glass cracking thermal shock"),
        ("a glass cutting skin along its broken edge",          "broken glass cutting skin"),
        ("a glass pane flexing slightly under wind pressure",   "glass pane flexing wind"),
    ],
    "coin": [
        ("a coin rolling on its edge and falling flat",         "coin rolling on edge falling"),
        ("a coin sinking immediately when dropped in water",    "coin sinking water dropped"),
        ("coins stacking when placed on top of each other",     "coins stacking pile"),
        ("a coin conducting heat away from a fingertip",        "coin conducting heat finger"),
        ("a coin being flipped and landing randomly",           "coin flip landing"),
    ],
    "shelf": [
        ("a shelf bowing in the middle under heavy books",      "shelf bowing heavy books middle"),
        ("books sliding off an overloaded shelf",               "books sliding off overloaded shelf"),
        ("a shelf bracket pulling out of a wall under load",    "shelf bracket pulling out wall"),
        ("weight distributed across a shelf staying stable",   "weight distributed shelf stable"),
        ("a shelf mounted too high to reach without a step",    "high shelf unreachable without step"),
    ],
    "pipe": [
        ("water flowing through a pipe under pressure",         "water flowing pipe pressure"),
        ("a pipe bursting when internal pressure exceeds wall strength","pipe bursting pressure"),
        ("a blocked pipe causing pressure to build upstream",   "blocked pipe pressure buildup"),
        ("a pipe bending without breaking when forced",         "pipe bending without breaking"),
        ("water hammer shock when flow is suddenly stopped",    "water hammer shock pipe"),
    ],
    "net": [
        ("a net trapping fish while letting water through",     "fishing net trapping fish"),
        ("a net catching a falling object",                     "safety net catching falling object"),
        ("a net tearing when the load exceeds its strength",    "net tearing under load"),
        ("a net stretching to absorb impact",                   "net stretching absorb impact"),
        ("an object passing through mesh that is too large",    "object too large for net mesh"),
    ],
    "chain": [
        ("a chain transmitting tension between two objects",    "chain transmitting tension"),
        ("a chain wrapped around a capstan to multiply force",  "chain capstan force multiply"),
        ("a chain snapping at its weakest link under load",     "chain snapping weakest link"),
        ("a chain failing to push — buckling under compression","chain buckling compression"),
        ("a chain swinging as a pendulum",                      "chain swinging pendulum"),
    ],
}

# Sanity check
ALL_CONCEPTS = list(PHRASE_BANK.keys())
total_events = sum(len(v) for v in PHRASE_BANK.values())

if __name__ == "__main__":
    print(f"Concepts: {len(ALL_CONCEPTS)}")
    print(f"Total phrase/image pairs: {total_events}")
    print(f"Mean per concept: {total_events/len(ALL_CONCEPTS):.1f}")
    print()

    # Show polysemy-heavy concepts
    poly = ["run", "strike", "press", "light", "charge", "bark", "spring"]
    print("Polysemous concepts — sense coverage:")
    for c in poly:
        phrases = [p for p, _ in PHRASE_BANK[c]]
        print(f"\n  {c}:")
        for p in phrases:
            print(f"    - {p}")
